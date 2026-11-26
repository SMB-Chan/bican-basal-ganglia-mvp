"""Seed-paired effects of measured composition, and the census figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from run import write_csv


HERE = Path(__file__).resolve().parent
OUTCOMES = ("tonic_rate_hz_mae", "tonic_isi_cv_mae", "rate_GPe_hz", "rate_STN_hz", "rate_GPi_hz",
            "pulse_features", "pulse_early_hz", "pulse_late_hz", "blocked_late_hz", "late_attenuation_hz",
            "fixed60_pulse_features", "fixed60_pulse_late_hz", "fixed60_late_attenuation_hz",
            "in_degree_GPe", "in_degree_STN", "in_degree_GPi", "edges", "excitatory_edges")
LABELS = {"assumed": "Assumed\n(config)", "striatal_measured": "Striatum\nmeasured",
          "pallidal_measured": "GPe/STN/GPi\nmeasured", "both_measured": "Both\nmeasured",
          "striatal_literature": "Striatum\nliterature 14%"}
BLUE, ORANGE, AQUA, INK, MUTED = "#257de0", "#f47230", "#00b894", "#26354a", "#8a94a3"


def aggregate(metrics, protocol):
    seeds = sorted({int(row["seed"]) for row in metrics})
    index = {(row["composition"], row["wiring"], row["drive"], int(row["seed"])): row for row in metrics}
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    draws = rng.integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    cells = sorted({key[:3] for key in index})
    data = {cell: np.array([[float(index[(*cell, s)][k]) for k in OUTCOMES] for s in seeds]) for cell in cells}
    summary, paired = [], []
    for (composition, wiring, drive), values in data.items():
        for i, outcome in enumerate(OUTCOMES):
            low, high = np.quantile(values[draws, i].mean(axis=1), [0.025, 0.975])
            summary.append({"composition": composition, "wiring": wiring, "drive": drive, "outcome": outcome,
                            "seeds": len(seeds), "mean": values[:, i].mean(), "sd": values[:, i].std(ddof=1),
                            "ci_low": low, "ci_high": high})
    for wiring in protocol["wirings"]:
        for drive in ("raw", "drive_matched"):
            for after, before in protocol["comparisons"]:
                delta = data[after, wiring, drive] - data[before, wiring, drive]
                for i, outcome in enumerate(OUTCOMES):
                    low, high = np.quantile(delta[draws, i].mean(axis=1), [0.025, 0.975])
                    paired.append({"wiring": wiring, "drive": drive, "after": after, "before": before,
                                   "outcome": outcome, "seeds": len(seeds), "delta_mean": delta[:, i].mean(),
                                   "ci_low": low, "ci_high": high, "seeds_positive": int((delta[:, i] > 0).sum()),
                                   "seeds_negative": int((delta[:, i] < 0).sum())})
    return summary, paired


def plot(path, census, summary, mixing, protocol, wiring="F_patchseq"):
    figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]})
    fig.suptitle("Measured MERSCOPE composition | one macaque, 47 striatal sections", fontsize=15,
                 fontweight="bold")

    ax = axes[0]
    comp = census["dorsal_striatum_composition"]["0.25"]
    shares = census["regional_density"]["0.1"]["regions"]
    labels = ["Striatal IN\n(% of SPN+IN)", "GPe", "STN", "GPi"]
    measured = [100 * comp["interneuron_fraction_of_spn_plus_interneurons"]] + \
               [100 * shares[r]["share_of_GPe_STN_GPi"] for r in ("GPe", "STN", "GPi")]
    model = [100 * 28 / 500, 40, 24, 36]
    x = np.arange(len(labels))
    ax.bar(x - 0.19, model, 0.36, color=MUTED, label="Model (config.json)")
    ax.bar(x + 0.19, measured, 0.36, color=BLUE, label="MERSCOPE estimate")
    ax.plot([-0.19 - 0.18, 0.19 + 0.18], [100 * protocol["literature_interneuron_fraction"]] * 2, color=ORANGE,
            lw=2, label="Stereology, Del Rey 2022")
    for xi, value in zip(x + 0.19, measured):
        ax.text(xi, value + 1, f"{value:.0f}" if value >= 10 else f"{value:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("% of striatal neurons  |  % of GPe+STN+GPi")
    ax.set_title("A  Composition")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[1]
    stats = {(r["composition"], r["wiring"], r["drive"], r["outcome"]): r for r in summary}
    names = list(protocol["compositions"])
    for offset, drive, color, marker, label in ((-0.12, "raw", ORANGE, "o", "Counts only (fixed out-degree)"),
                                                (0.12, "drive_matched", BLUE, "s", "Counts + group-pair drive matched")):
        rows = [stats[n, wiring, drive, "tonic_rate_hz_mae"] for n in names]
        mean = np.array([float(r["mean"]) for r in rows])
        err = np.array([[mean[i] - float(r["ci_low"]), float(r["ci_high"]) - mean[i]] for i, r in enumerate(rows)]).T
        ax.errorbar(np.arange(len(names)) + offset, mean, yerr=err, color=color, marker=marker, ms=7, lw=0,
                    elinewidth=2, capsize=3, label=label)
    ax.set_xticks(np.arange(len(names)), [LABELS[n] for n in names], fontsize=8.5)
    ax.set_ylabel("Tonic rate error vs awake macaque (Hz)")
    ax.set_title("B  Network with Patch-seq τ (F wiring), 24 seeds")
    ax.set_ylim(0, max(float(r["ci_high"]) for r in summary if r["outcome"] == "tonic_rate_hz_mae") * 1.45)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncols=2)

    ax = axes[2]
    rng = np.random.default_rng(3)
    for i, (key, label) in enumerate((("d1d2", "D1 vs D2"), ("compartment", "Striosome vs matrix"))):
        for j, (scope, color, marker) in enumerate((("section", ORANGE, "o"), ("tile", BLUE, "s"))):
            values = np.array([100 * float(r[f"{key}_{scope}_excess"]) for r in mixing])
            position = i + (j - 0.5) * 0.36
            ax.scatter(position + rng.uniform(-0.08, 0.08, len(values)), values, s=16, color=color, marker=marker,
                       alpha=0.75, linewidths=0, label=None if i else
                       {"section": "Shuffle within section", "tile": "Shuffle within 0.5 mm tile"}[scope])
            ax.plot([position - 0.13, position + 0.13], [np.median(values)] * 2, color=INK, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8, linestyle="--")
    ax.set_xticks([0, 1], ["D1 vs D2", "Striosome vs matrix"])
    ax.set_ylabel("Same-label share of 8 neighbours,\nobserved − shuffled (points)")
    ax.set_title("C  Local mixing per section")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=HERE / "results")
    args = parser.parse_args()
    protocol = json.loads((HERE / "composition_protocol.json").read_text(encoding="utf-8"))
    census = json.loads((args.results / "census.json").read_text(encoding="utf-8"))
    metrics = read_csv(args.results / "composition" / "composition_metrics.csv")
    summary, paired = aggregate(metrics, protocol)
    write_csv(args.results / "composition" / "composition_summary.csv", summary)
    write_csv(args.results / "composition" / "composition_paired.csv", paired)
    plot(args.results / "composition.png", census, summary, read_csv(args.results / "spn_mixing_by_section.csv"),
         protocol)
