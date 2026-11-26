"""Summarize paired network-seed and held-out donor outcomes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from heterogeneity.patchseq import RHESUS
from run import write_csv


NETWORK_METRICS = ("tonic_rate_mae_hz", "tonic_cv_mae", "pulse_features", "pulse_early_hz", "pulse_late_hz",
                   "blocked_late_hz", "late_attenuation_hz", "choice_weak_accuracy", "choice_strong_accuracy")
DONOR_METRICS = ("mae_pa", "signed_error_pa")
DONOR_MODELS = ("train_median", "cell_passive", "calibrated_cell")
NETWORK_LABELS = {"original": "Original", "median": "Median τ", "tau": "Variable τ", "shuffled": "Shuffled traits",
                  "joint": "Paired traits", "rhesus_joint": "Rhesus only"}


def bootstrap(values, rng, repetitions):
    samples = rng.integers(0, len(values), size=(repetitions, len(values)))
    low, high = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def summarize_network(rows, protocol, seeds):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    repetitions = protocol["bootstrap_replicates"]
    index = {(row["wiring"], row["mode"], int(row["seed"])): row for row in rows}
    values = {(wiring, mode, metric): np.array([float(index[wiring, mode, seed][metric]) for seed in seeds])
              for wiring in protocol["wiring"] for mode in protocol["conditions"] for metric in NETWORK_METRICS}
    summary = [{"wiring": wiring, "mode": mode, "metric": metric, "seeds": len(seeds), **bootstrap(data, rng, repetitions)}
               for (wiring, mode, metric), data in values.items()]
    pairs = []
    for wiring in protocol["wiring"]:
        for after, before in protocol["comparisons"]:
            for metric in NETWORK_METRICS:
                diff = values[wiring, after, metric] - values[wiring, before, metric]
                pairs.append({"wiring": wiring, "after": after, "before": before, "metric": metric,
                              "seeds": len(seeds), **bootstrap(diff, rng, repetitions)})
    return summary, pairs


def donor_values(rows, subset, metric):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        if not subset(row):
            continue
        key = (int(row["fold"]), row["model"])
        n = int(row["held_cells"])
        totals[key] += n * float(row[metric])
        counts[key] += n
    folds = sorted({fold for fold, model in totals})
    values = {model: np.array([totals[fold, model] / counts[fold, model] for fold in folds]) for model in DONOR_MODELS}
    return values, sum(counts[fold, "train_median"] for fold in folds)


def summarize_donors(rows, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"] + 1)
    repetitions = protocol["bootstrap_replicates"]
    subsets = {"all": lambda row: True, "D1": lambda row: row["kind"] == "D1",
               "D2": lambda row: row["kind"] == "D2", "rhesus": lambda row: row["species"] == RHESUS}
    summary, pairs = [], []
    for cohort, subset in subsets.items():
        for metric in DONOR_METRICS:
            values, n_cells = donor_values(rows, subset, metric)
            keys = {"cohort": cohort, "metric": metric, "donors": len(values["train_median"]), "cells": n_cells}
            for model, data in values.items():
                summary.append({**keys, "model": model, **bootstrap(data, rng, repetitions)})
            for model in ("cell_passive", "calibrated_cell"):
                diff = values[model] - values["train_median"]
                pairs.append({**keys, "model": model, **bootstrap(diff, rng, repetitions)})
    return summary, pairs


def plot_results(out, network, donors, protocol):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), layout="constrained")
    fig.suptitle("What do measured cell properties add?", fontsize=16, fontweight="bold")
    color = {"cell_type": "#387990", "projection": "#a64e30"}
    mode = protocol["conditions"]
    labels = {"cell_type": "Distance + cell type", "projection": "With projection priors"}
    x = np.arange(len(mode))
    specifications = (("tonic_rate_mae_hz", "Tonic firing rate", "Mean absolute error (Hz)"),
                      ("tonic_cv_mae", "Spike-interval variability", "Mean absolute CV error"),
                      ("blocked_late_hz", "GPi after an STN pulse and pathway blockade", "Late response vs sham (Hz)"))
    index = {(row["wiring"], row["mode"], row["metric"]): row for row in network}
    for ax, (metric, title, ylabel) in zip(axes.flat, specifications):
        ax.axhline(0, lw=0.8, color="#929ca9", linestyle="--")
        for i, wiring in enumerate(protocol["wiring"]):
            data = [index[wiring, name, metric] for name in mode]
            values = np.array([row["mean"] for row in data])
            lows = np.array([row["ci_low"] for row in data])
            highs = np.array([row["ci_high"] for row in data])
            errors = np.maximum(0, [values - lows, highs - values])
            ax.errorbar(x + (i * 2 - 1) * 0.07, values, yerr=errors, color=color[wiring], marker="o", ms=4, capsize=3,
                        lw=1.2, label=labels[wiring])
        ax.set_xticks(x, [NETWORK_LABELS[name] for name in mode], rotation=25, ha="right")
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1, 1]
    labels = {"train_median": "Training type median", "cell_passive": "Cell measurements",
              "calibrated_cell": "Cell measurements + calibration"}
    colors = ["#6d7e8f", "#a64e30", "#397960"]
    for i, model in enumerate(DONOR_MODELS):
        row = next(row for row in donors if row["cohort"] == "all" and row["model"] == model and row["metric"] == "mae_pa")
        mean = row["mean"]
        ax.barh(i, mean, xerr=[[max(0, mean - row["ci_low"])], [max(0, row["ci_high"] - mean)]], color=colors[i], capsize=3)
    ax.set_yticks(range(3), [labels[model] for model in DONOR_MODELS])
    ax.invert_yaxis()
    ax.set(title="Rheobase prediction on held-out donors", xlabel="Mean absolute error (pA)")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    save(fig, out / "measured_cell_properties.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    network, network_pairs = summarize_network(read_csv(out / "network.csv"), protocol, manifest["seeds"])
    donors, donor_pairs = summarize_donors(read_csv(out / "donor_validation.csv"), protocol)
    for name, rows in (("network_summary", network), ("network_paired", network_pairs), ("donor_summary", donors), ("donor_paired", donor_pairs)):
        write_csv(out / f"{name}.csv", rows)
    figure_style()
    plot_results(out, network, donors, protocol)
