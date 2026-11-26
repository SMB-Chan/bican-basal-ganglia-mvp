"""Report seed-level uncertainty and paired effects without pooling dependent trials."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from run import write_csv


OUTCOMES = ("tonic_rate_mae_hz", "tonic_cv_mae", "pulse_features", "pulse_early_hz", "pulse_late_hz",
            "blocked_early_hz", "blocked_late_hz", "late_attenuation_hz", "early_excitation", "late_inhibition",
            "blocked_early_excitation", "late_attenuation", "choice_weak_accuracy", "choice_strong_accuracy",
            "choice_weak_margin_hz", "choice_strong_margin_hz")
LABELS = {"A_random": "A Random", "B_distance": "B Distance", "C_cell_type": "C Cell type",
          "D_mask": "D Mask", "E_odds": "E Odds", "F_patchseq": "F Patch-seq", "G_channel": "G Channel",
          "E_channel": "E + channel", "E_gaussian": "E Gaussian", "F_gaussian": "F Gaussian"}
COLORS = ["#66788a", "#437fb4", "#277c78", "#937638", "#a44d2b", "#663f82", "#1c494c",
          "#938686", "#be886c", "#9980ad"]


def aggregate(metrics, protocol, seeds):
    index = {(row["condition"], int(row["seed"])): row for row in metrics}
    n = len(seeds)
    rng = np.random.default_rng(protocol["resample_seed"])
    bootstrap = rng.integers(0, n, size=(protocol["bootstrap_replicates"], n))
    by_condition = {name: np.array([[float(index[name, seed][key]) for key in OUTCOMES] for seed in seeds])
                    for name in protocol["conditions"]}
    summary = []
    pairs = []
    for name, data in by_condition.items():
        for i, outcome in enumerate(OUTCOMES):
            mean = data[:, i].mean()
            low, high = np.quantile(data[bootstrap, i].mean(axis=1), [0.025, 0.975])
            sd = data[:, i].std(ddof=1) if n > 1 else None
            summary.append({"condition": name, "outcome": outcome, "seeds": n, "mean": mean, "sd": sd,
                            "ci_low": low, "ci_high": high})
    for after, before in protocol["comparisons"]:
        differences = by_condition[after] - by_condition[before]
        for i, outcome in enumerate(OUTCOMES):
            values = differences[:, i]
            low, high = np.quantile(values[bootstrap].mean(axis=1), [0.025, 0.975])
            pairs.append({"after": after, "before": before, "outcome": outcome, "seeds": n,
                          "delta_mean": values.mean(), "ci_low": low, "ci_high": high,
                          "seeds_positive": int((values > 0).sum()), "seeds_negative": int((values < 0).sum())})
    return summary, pairs


def plot_scores(directory, summary, protocol):
    stats = {(row["condition"], row["outcome"]): row for row in summary}
    conditions = list(protocol["conditions"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained")
    fig.suptitle("Constraint ladder | Functional validation", fontsize=16, fontweight="bold")
    outcomes = (("tonic_rate_mae_hz", "Tonic firing-rate error", "Mean absolute error (Hz)", 0),
                ("tonic_cv_mae", "Spike-interval variability error", "Mean absolute CV error", 0),
                ("pulse_features", "GPi response to STN stimulation", "Matched experimental features (of 4)", 0),
                ("choice_strong_accuracy", "Two-choice task (simulation diagnostic)", "Accuracy at 5 mV input difference", 0.5))
    x = np.arange(len(conditions))
    for ax, (key, title, ylabel, ref) in zip(axes.flat, outcomes):
        ax.axhline(ref, color="#738297", linewidth=0.9, linestyle="--")
        for i, name in enumerate(conditions):
            row = stats[name, key]
            y = row["mean"]
            ax.errorbar(i, y, yerr=[[max(0, y - row["ci_low"])], [max(0, row["ci_high"] - y)]],
                        color=COLORS[i], marker="o", ms=6, capsize=3, linewidth=1.5)
        ax.axvline(6.5, color="#ccd3da", lw=0.8)
        ax.set_xticks(x, [LABELS[name] for name in conditions], rotation=35, ha="right")
        ax.set(xlabel="", ylabel=ylabel, title=title, xlim=(-0.5, len(x) - 0.5))
        ax.grid(axis="y", alpha=0.2)
        if key == "pulse_features":
            ax.set(ylim=(-0.12, 4.17), yticks=range(5))
        if key == "choice_strong_accuracy":
            ax.set_ylim(0, 1.05)
    save(fig, directory / "functional_scores.png")


def plot_pulse(directory, rows, protocol):
    conditions = ["A_random", "C_cell_type", "D_mask", "E_odds", "F_patchseq", "G_channel"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.1), layout="constrained", sharex=True, sharey=True)
    fig.suptitle("GPi response to a single STN pulse", fontsize=16, fontweight="bold")
    spec = protocol["stn_pulse"]
    for ax, name in zip(axes.flat, conditions):
        subset = [row for row in rows if row["condition"] == name]
        time = sorted({float(row["time_ms"]) for row in subset})
        ax.axvspan(*spec["early_window_ms"], color="#81b7be", alpha=0.24)
        ax.axvspan(*spec["late_window_ms"], color="#deac8d", alpha=0.23)
        ax.axhline(0, color="#7d8a9b", linestyle="--", lw=0.85)
        for key, label, color in (("intact_hz", "Intact", "#1b6274"), ("blocked_hz", "STN→GPe blocked", "#ac532a")):
            values = np.array([np.mean([float(row[key]) for row in subset if float(row["time_ms"]) == tick]) for tick in time])
            ax.plot(time, values, color=color, lw=1.4, label=label)
        ax.set_title(LABELS[name], loc="left")
        ax.set_xlim(0, max(time))
    for ax in axes[:, 0]:
        ax.set_ylabel("GPi rate minus sham (Hz)")
    for ax in axes[-1]:
        ax.set_xlabel("Time after pulse (ms)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, directory / "pulse_response.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    directory = parser.parse_args().results
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    summary, pairs = aggregate(read_csv(directory / "metrics.csv"), protocol, manifest["seeds"])
    write_csv(directory / "summary.csv", summary)
    write_csv(directory / "paired.csv", pairs)
    figure_style()
    plot_scores(directory, summary, protocol)
    plot_pulse(directory, read_csv(directory / "pulse_curves.csv"), protocol)
