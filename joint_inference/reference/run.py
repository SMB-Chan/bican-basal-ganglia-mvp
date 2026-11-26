"""Use independent zero-change references to estimate GPi bias and pathway gain."""
import argparse
import csv
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from joint_inference.run import HERE as STUDY, init_worker, simulate_model
from partial_block.run import read_json, write_rows
from atlas import ROOT

HERE = Path(__file__).resolve().parent
OBS = ["isol_low", "isol_high", "peak_pre"]


def measure(task):
    cohort, seed = task
    design = read_json(HERE / "protocol.json")
    setup = read_json(STUDY / "protocol.json")
    setup["test_strengths"] = [0.0] if cohort == "reference" else design["strengths"]
    setup["test_block"] = [design["block"]]
    rows = []
    for model in read_json(ROOT / "partial_block/protocol.json")["models"]:
        for row in simulate_model(model, seed, "test", setup):
            rows.append(dict(cohort=cohort, **{k: row[k] for k in ["model", "seed", "h1", "h2", *OBS]}))
    return rows


def estimate(control, measurements, cap):
    slope = (control[..., 1] - control[..., 0]) / 4
    if np.any(slope <= 0) or np.any(control[..., 2] <= 0):
        raise ValueError("Reference slope and peak must be positive")
    physical = (control[..., :2].mean(axis=-1) - measurements[..., :2].mean(axis=-1)) / slope
    gain = measurements[..., 2] / control[..., 2] - 1
    return np.stack((np.clip(physical, 0, cap), np.clip(gain, 0, 1)), axis=-1)


def score(out):
    design = read_json(HERE / "protocol.json")
    with (out / "measurements.csv").open() as f:
        rows = list(csv.DictReader(f))
    extents = read_json(ROOT / "partial_block/results/manifest.json")["interpolation_mv"]
    models = list(extents)
    def index(row):
        return row["model"], int(row["seed"]), float(row["h1"]), float(row["h2"])
    values = {index(r): [float(r[p]) for p in OBS] for r in rows}
    refs, tests = design["reference_seeds"], design["evaluation_seeds"]
    grid = list(itertools.product(design["strengths"], repeat=2))
    if len(values) != len(rows) or len(rows) != len(models) * (len(refs) + len(tests) * len(grid)):
        raise ValueError("Wrong number of unique measurements")
    ref = {m: np.array([values[m, s, 0.0, 0.0] for s in refs]) for m in models}
    cap = design["h1_physical_cap_mv"]
    rng = np.random.default_rng(design["bootstrap_seed"])
    n = design["bootstrap_replicates"]
    half = len(refs) // 2
    first = rng.integers(half, size=(n, half))
    last = half + rng.integers(half, size=(n, half))
    test_draw = rng.integers(len(tests), size=(n, len(tests)))
    effects, summary, controls, verdict = [], [], [], {}
    for m in models:
        others = [v for k, v in ref.items() if k != m]
        other = (others[0][:half] + others[1][:half]) / 2
        target = abs(extents[m])
        truth = np.array(grid) * [target, 1.0]
        x = np.array([[values[m, s, a, b] for a, b in grid] for s in tests])
        centers = {"matched": ref[m].mean(axis=0), "other": other.mean(axis=0)}
        boot = {"matched": (ref[m][first].mean(axis=1) + ref[m][last].mean(axis=1)) / 2,
                "other": other[first].mean(axis=1)}
        metric = {}
        for label, center in centers.items():
            controls.append(dict(model=m, reference=label, **dict(zip(OBS, center))))
            def errors(pred):
                e = np.abs(pred - truth)
                return np.stack((e[..., 0], e[..., 0]/target, e[..., 1],
                                 (e[..., 0]/target + e[..., 1])/2), axis=-1)
            point = errors(estimate(center, x, cap)).mean(axis=(0, 1))
            chunks = []
            for start in range(0, n, 100):
                take = slice(start, start + 100)
                pred = estimate(boot[label][take, None, None, :], x[test_draw[take]], cap)
                chunks.append(errors(pred).mean(axis=(1, 2)))
            distribution = np.concatenate(chunks)
            lo, hi = np.quantile(distribution, [0.025, 0.975], axis=0)
            for j, key in enumerate(("h1_mv", "h1_fraction", "h2_fraction", "joint_fraction")):
                summary.append(dict(model=m, reference=label, target=key, mean=point[j], low=lo[j], high=hi[j]))
            metric[label] = point, distribution
        change = metric["other"][0][-1] - metric["matched"][0][-1]
        low, high = np.quantile(metric["other"][1][:, -1] - metric["matched"][1][:, -1], [0.025, 0.975])
        effects.append(dict(model=m, improvement=change, low=low, high=high))
        verdict[m] = bool(max(metric["matched"][0][1:3]) <= design["max_normalized_mae"]
                          and low > design["min_improvement"])
    write_rows(out / "summary.csv", summary)
    write_rows(out / "improvement.csv", effects)
    write_rows(out / "references.csv", controls)
    (out / "decisions.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    design = read_json(HERE / "protocol.json")
    old = read_json(STUDY / "protocol.json")
    ids = design["reference_seeds"] + design["evaluation_seeds"]
    if len(ids) != len(set(ids)) or set(ids) & set(old["train_seeds"] + old["test_seeds"]):
        raise ValueError("Reference and evaluation cohorts must be new and disjoint")
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = [("reference", s) for s in design["reference_seeds"]] + [("evaluation", s) for s in design["evaluation_seeds"]]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), initializer=init_worker) as pool:
        write_rows(args.out / "measurements.csv", [row for group in pool.map(measure, jobs) for row in group])
    score(args.out)
