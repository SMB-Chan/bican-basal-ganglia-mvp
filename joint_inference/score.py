"""Evaluate fixed measurement panels on unseen graphs and mechanism strengths."""
import argparse
import csv
import gzip
import itertools
import json
import platform
from pathlib import Path

import numpy as np

from partial_block.run import read_json, write_rows

HERE = Path(__file__).resolve().parent


def quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((x, x[:, i] * x[:, j]))


def predict(train, test, features, penalty):
    def measurements(rows):
        return np.array([[float(r[key]) for key in features] for r in rows])
    x, xt = measurements(train), measurements(test)
    labels = np.array([[float(r["h1"]), float(r["h2"])] for r in train])
    average, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    basis, held = quadratic((x - average) / scale), quadratic((xt - average) / scale)
    center = basis.mean(axis=0)
    basis -= center
    coef = np.linalg.solve(basis.T @ basis + penalty * np.eye(basis.shape[1]), basis.T @ (labels - labels.mean(axis=0)))
    return np.clip(labels.mean(axis=0) + (held - center) @ coef, 0, 1)


def read_observations(out, setup):
    with gzip.open(out / "observations.csv.gz", "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = ("h1", "h2", "eff_true")
    models = list(read_json(HERE.parent / "partial_block/protocol.json")["models"])
    expected = {(split, m, s, a, b, e) for split in ("train", "test")
                for m, s, a, b, e in itertools.product(models, setup[split + "_seeds"],
                    setup[split + "_strengths"], setup[split + "_strengths"], setup[split + "_block"])}
    observed = {(r["split"], r["model"], int(r["seed"]), *(float(r[k]) for k in labels)) for r in rows}
    if expected != observed or len(observed) != len(rows):
        raise ValueError("Missing, duplicate or unexpected observations")
    return rows, models


def score(out):
    setup = read_json(HERE / "protocol.json")
    rows, models = read_observations(out, setup)
    seeds = setup["test_seeds"]
    bootstrap = np.random.default_rng(setup["bootstrap_seed"]).integers(len(seeds), size=(setup["bootstrap_replicates"], len(seeds)))
    seed_scores, summary, effects, proxy, decisions = [], [], [], [], {}
    def interval(values):
        low, high = np.quantile(values[bootstrap].mean(axis=1), [0.025, 0.975])
        return dict(mean=float(values.mean()), low=float(low), high=float(high))

    for model in models:
        test = [r for r in rows if r["split"] == "test" and r["model"] == model]
        y = np.array([[float(r["h1"]), float(r["h2"])] for r in test])
        ids = np.array([int(r["seed"]) for r in test])
        proxy_error = np.array([abs(float(r["eff_proxy"]) - float(r["eff_true"])) for r in test])
        proxy.append(dict(model=model, **interval(np.array([proxy_error[ids == s].mean() for s in seeds]))))
        for mode in ("known_model", "unseen_model"):
            train = [r for r in rows if r["split"] == "train" and ((r["model"] == model) == (mode == "known_model"))]
            panels = {}
            for name, features in {"constant": [], **setup["features"]}.items():
                if ({"seed", "h1", "h2"} & set(features)
                        or ("eff_true" in features and name != "rates_known_efficacy")):
                    raise ValueError("Ground truth used as an observable")
                estimate = predict(train, test, features, setup["ridge"]) if features else np.full_like(y, 0.5)
                abs_error = np.abs(estimate - y)
                per_seed = np.array([abs_error[ids == s].mean(axis=0) for s in seeds])
                panels[name] = per_seed
                for s, (e1, e2) in zip(seeds, per_seed):
                    seed_scores.append(dict(model=model, mode=mode, panel=name, seed=s, h1_mae=e1, h2_mae=e2))
                for metric, vals in [("h1", per_seed[:, 0]), ("h2", per_seed[:, 1]), ("joint", per_seed.mean(axis=1))]:
                    summary.append(dict(model=model, mode=mode, panel=name, target=metric, **interval(vals)))
            gain = interval((panels["rates"] - panels["joint"]).mean(axis=1))
            errors = panels["joint"].mean(axis=0)
            passed = bool(max(errors) <= setup["primary_mae_max"] and gain["low"] > setup["primary_improvement_min"])
            effects.append(dict(model=model, mode=mode, improvement=gain["mean"], low=gain["low"], high=gain["high"]))
            if mode == "known_model":
                decisions[model] = dict(h1_mae=float(errors[0]), h2_mae=float(errors[1]), improvement=gain, passed=passed)
    write_rows(out / "by_seed.csv", seed_scores)
    write_rows(out / "summary.csv", summary)
    write_rows(out / "improvement.csv", effects)
    write_rows(out / "efficacy.csv", proxy)
    manifest = dict(parent=setup["parent"], python=platform.python_version(), numpy=np.__version__,
                    train_rows=sum(r["split"] == "train" for r in rows), test_rows=sum(r["split"] == "test" for r in rows),
                    decisions=decisions)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    score(parser.parse_args().folder)
