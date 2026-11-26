"""Score the 12 Hz positive control against the pre-registered criteria and draw index-vs-amplitude curves."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from run import write_csv


GROUPS = ("GPe", "STN", "GPi")
METRICS = ("pop_beta_index", "pop_beta_power_hz2", "rate_hz", "isi_cv", "peak_freq_hz")


def summarize(rows, protocol):
    seeds = sorted({int(r["seed"]) for r in rows})
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    keys = sorted({(r["state"], r["target"], float(r["amplitude_mv"])) for r in rows})
    index = {(r["state"], r["target"], float(r["amplitude_mv"]), int(r["seed"])): r for r in rows}
    out = []
    for state, target, amp in keys:
        row = {"state": state, "target": target, "amplitude_mv": amp, "seeds": len(seeds)}
        for g in GROUPS:
            for m in METRICS:
                k = f"{g}_{m}"
                if index[state, target, amp, seeds[0]].get(k, "") == "":
                    continue
                v = np.array([float(index[state, target, amp, s][k]) for s in seeds])
                low, high = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
                row.update({k: float(v.mean()), k + "_ci_low": float(low), k + "_ci_high": float(high)})
        out.append(row)
    return out


def curve(summary, state, target):
    base = next(r for r in summary if r["state"] == state and r["target"] == "none")
    rest = sorted((r for r in summary if r["state"] == state and r["target"] == target), key=lambda r: r["amplitude_mv"])
    return [base] + rest


def verdict(summary):
    stn_curve = curve(summary, "pd", "STN")
    eligible = [r for r in stn_curve if r["STN_pop_beta_index"] >= 1.5]
    gpi_curve = curve(summary, "pd", "GPi")
    return {
        "R1_gpi_reachable_via_stn": {
            "amplitudes_with_stn_index_at_least_1_5": [r["amplitude_mv"] for r in eligible],
            "gpi_index_at_those_amplitudes": [r["GPi_pop_beta_index"] for r in eligible],
            "pass": any(r["GPi_pop_beta_index"] >= 1.5 for r in eligible)},
        "R2_gpi_metric_can_register": {
            "gpi_index_by_amplitude": {str(r["amplitude_mv"]): r["GPi_pop_beta_index"] for r in gpi_curve},
            "pass": any(r["GPi_pop_beta_index"] >= 1.5 for r in gpi_curve)},
        "isolated_gpi_index": next(r for r in summary if r["state"] == "isolated")["GPi_pop_beta_index"],
    }


def make_figure(out, summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    fig.suptitle("Positive control: a known 12 Hz rhythm and the 8–15 Hz index", fontsize=14, fontweight="bold")
    colors = {"GPe": "#437fb4", "STN": "#a34b31", "GPi": "#207c74"}
    panels = (("pd", "STN", "Rhythm imposed on STN | dopamine-depleted"), ("normal", "STN", "Rhythm imposed on STN | normal"),
              ("pd", "GPi", "Rhythm imposed on GPi | dopamine-depleted"))
    isolated = next(r for r in summary if r["state"] == "isolated")["GPi_pop_beta_index"]
    for ax, (state, target, title) in zip(axes, panels):
        rows = curve(summary, state, target)
        x = [r["amplitude_mv"] for r in rows]
        for g in GROUPS:
            y = np.array([r[f"{g}_pop_beta_index"] for r in rows])
            lo = np.array([r[f"{g}_pop_beta_index_ci_low"] for r in rows])
            hi = np.array([r[f"{g}_pop_beta_index_ci_high"] for r in rows])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3, color=colors[g], label=g)
        ax.axhline(1.5, color="#7a2f7a", linestyle="--", lw=1, label="Stage 4B threshold 1.5")
        ax.axhline(1.0, color="#929ca9", linestyle=":", lw=1, label="Poisson level 1")
        if target == "GPi":
            ax.axhline(isolated, color="#207c74", linestyle=":", lw=1, label=f"Synapse-free GPi ({isolated:.2f})")
        ax.set(title=title, xlabel="Imposed rhythm amplitude (mV)", ylabel="8–15 Hz index", yscale="log")
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xticks([0, 0.5, 1, 2, 4, 8], ["0", "0.5", "1", "2", "4", "8"])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    save(fig, out / "positive_control.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    protocol = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["protocol"]
    summary = summarize(read_csv(out / "evaluations.csv"), protocol)
    write_csv(out / "summary.csv", summary)
    result = verdict(summary)
    (out / "verdict.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figure_style()
    make_figure(out, summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))
