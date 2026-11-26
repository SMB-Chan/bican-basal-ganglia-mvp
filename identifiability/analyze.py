"""Summarize graph collisions and test held-out discrimination of pathway hypotheses."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from identifiability.run import CONTINUOUS, SIGNS
from run import write_csv


def summarize_ensembles(rows, protocol):
    summaries = []
    for membrane in protocol["membrane_conditions"]:
        for hypothesis in protocol["hypotheses"]:
            samples = [row for row in rows if row["membrane"] == membrane and row["hypothesis"] == hypothesis]
            statistics = {}
            for metric in ("tonic_rate_mae_hz", "pulse_early_hz", "pulse_late_hz", "blocked_late_hz", "stn_to_gpe_fraction"):
                values = np.array([float(row[metric]) for row in samples])
                statistics.update({metric + "_mean": values.mean(), metric + "_sd": values.std(ddof=1),
                                   metric + "_min": values.min(), metric + "_max": values.max()})
            summaries.append({"membrane": membrane, "hypothesis": hypothesis, "graphs": len(samples),
                              "coarse_candidates": sum(int(row["external_coarse"]) for row in samples),
                              "strict_candidates": sum(int(row["external_strict"]) for row in samples), **statistics})
    return summaries


def summarize_pairs(pairs, protocol):
    summary = []
    hypotheses = list(protocol["hypotheses"])
    for membrane in protocol["membrane_conditions"]:
        for i, left in enumerate(hypotheses):
            for right in hypotheses[i:]:
                samples = [p for p in pairs if p["membrane"] == membrane and
                           {p["left_hypothesis"], p["right_hypothesis"]} == {left, right}]
                matched = [p for p in samples if p["continuous_close"] == "1"]
                close = [float(p["edge_jaccard"]) for p in matched]
                partners = Counter((p["placement"], p[f"{side}_hypothesis"], p[f"{side}_graph_seed"])
                                   for p in matched for side in ("left", "right"))
                overlaps = [float(p["edge_jaccard"]) for p in samples if p["edge_jaccard"] != ""]
                summary.append({"membrane": membrane, "left": left, "right": right, "pairs": len(samples),
                                "sign_matches": sum(int(p["signs_equal"]) for p in samples),
                                "continuous_matches": len(close), "graphs_with_a_continuous_match": len(partners),
                                "max_close_partners_for_one_graph": max(partners.values(), default=0),
                                "median_edge_jaccard_for_continuous_matches": float(np.median(close)) if close else None,
                                "min_edge_jaccard_for_continuous_matches": min(close) if close else None,
                                "median_edge_jaccard_measured": float(np.median(overlaps)) if overlaps else None,
                                "median_abs_stn_gpe_fraction_gap": np.median([float(p["stn_to_gpe_fraction_gap"]) for p in samples])})
    return summary


def predict_nearest_centroid(train, labels, test):
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    classes = np.unique(labels)
    means = np.array([train[labels == name].mean(axis=0) for name in classes])
    distances = np.sum(((test[:, None] - means[None]) / std) ** 2, axis=2)
    # Equal nearest distances occur with discrete sign features. Credit ties uniformly.
    nearest = np.isclose(distances, distances.min(axis=1, keepdims=True), rtol=1e-10, atol=1e-10)
    return classes, nearest / nearest.sum(axis=1, keepdims=True)


def discrimination(rows, responses, protocol):
    placement = np.array([int(row["placement"]) for row in rows])
    rank_split = np.array([int(row["graph_seed"]) % 2 for row in rows])
    membrane = np.array([row["membrane"] for row in rows])
    label = np.array([row["hypothesis"] for row in rows])
    signs = np.array([[float(row[key]) for key in SIGNS] for row in rows])
    continuous = np.array([[float(row[key]) for key in CONTINUOUS] for row in rows])
    waveform = responses.reshape(len(rows), -1)
    features = {"signs": signs, "continuous": continuous, "waveform_and_continuous": np.concatenate([waveform, continuous], axis=1)}
    folds, confusion = [], []
    placements = sorted(set(placement))
    if len(placements) < 2:
        return [], []
    for condition in protocol["membrane_conditions"]:
        for method, x in features.items():
            for p in placements:
                for split in (0, 1):
                    test = (placement == p) & (membrane == condition) & (rank_split == split)
                    train = (placement != p) & (membrane == condition) & (rank_split != split)
                    classes, probabilities = predict_nearest_centroid(x[train], label[train], x[test])
                    outcomes = label[test]
                    accuracy = np.sum(probabilities * (classes[None] == outcomes[:, None]), axis=1).mean()
                    key = {"membrane": condition, "features": method, "held_placement": p, "held_seed_parity": split}
                    folds.append({**key, "train": int(train.sum()), "test": int(test.sum()), "accuracy": accuracy})
                    for true in classes:
                        matched = probabilities[outcomes == true]
                        for predicted, credit in zip(classes, matched.sum(axis=0)):
                            confusion.append({**key, "true": true, "predicted": predicted, "credit": credit, "test_in_class": len(matched)})
    return folds, confusion


def make_figure(out, pairs, folds):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), layout="constrained")
    fig.suptitle("How much wiring can these measurements distinguish?", fontsize=15, fontweight="bold")
    ax = axes[0]
    between = [p for p in pairs if p["left"] != p["right"] and p["membrane"] == "F"]
    labels = {"baseline": "Balanced", "toward_GPe": "GPe-favoring", "toward_GPi": "GPi-favoring"}
    x = np.arange(len(between))
    for i, (key, name, color) in enumerate((("sign_matches", "Same four signs", "#647b91"),
                                            ("continuous_matches", "All continuous tolerances", "#a34b31"))):
        values = [p[key] / p["pairs"] for p in between]
        bars = ax.bar(x + (i - 0.5) * 0.3, values, width=0.3, label=name, color=color)
        ax.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=3, fontsize=8)
    ax.set_xticks(x, [f"{labels[p['left']]}\nvs {labels[p['right']]}" for p in between])
    ax.set(title="Different pathway hypotheses | F", ylabel="Fraction of matched graph draws", ylim=(0, 1.05))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax = axes[1]
    methods = ("signs", "continuous", "waveform_and_continuous")
    ax.axhline(1 / 3, color="#778595", linestyle="--", lw=1, label="Chance")
    for i, (condition, color) in enumerate((("E", "#647b91"), ("F", "#207c74"))):
        values = [np.mean([r["accuracy"] for r in folds if r["features"] == name and r["membrane"] == condition]) for name in methods]
        ax.bar(x + (i - 0.5) * 0.3, values, width=0.3, label=condition, color=color)
    ax.set_xticks(x, ["Four signs", "Continuous\nmeasures", "Continuous\n+ waveform"])
    ax.set(title="Identify pathway hypothesis on held-out graphs", ylabel="Classification accuracy", ylim=(0, 1.05))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "distinguishing_wiring.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    raw = read_csv(out / "evaluations.csv")
    pairs = summarize_pairs(read_csv(out / "pairs.csv"), protocol)
    with np.load(out / "response_arrays.npz") as file:
        folds, confusion = discrimination(raw, file["responses"], protocol)
    write_csv(out / "ensembles.csv", summarize_ensembles(raw, protocol))
    write_csv(out / "pair_summary.csv", pairs)
    if folds:
        write_csv(out / "discrimination_folds.csv", folds)
        write_csv(out / "discrimination_confusion.csv", confusion)
        figure_style()
        make_figure(out, pairs, folds)
