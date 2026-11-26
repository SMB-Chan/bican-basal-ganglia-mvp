"""Summarize the noise-regime sweep with seed-bootstrap intervals and draw the figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from run import write_csv


METRICS = ("tonic_rate_mae_hz", "tonic_cv_mae", "GPe_isi_cv", "STN_isi_cv", "GPi_isi_cv", "GPe_rate_hz", "STN_rate_hz",
           "GPi_rate_hz", "pulse_early_hz", "pulse_late_hz", "blocked_early_hz", "blocked_late_hz", "late_attenuation_hz",
           "pulse_features", "late_inhibition", "late_attenuation", "stn_late_hz", "stn_blocked_late_hz",
           "coarse_candidate", "strict_candidate")
COLORS = {"GPe": "#437fb4", "STN": "#a34b31", "GPi": "#207c74"}


def add_candidates(rows, ident):
    for row in rows:
        coarse = float(row["tonic_rate_mae_hz"]) <= ident["external_rate_mae_hz"] and int(row["pulse_features"]) == 4
        row["coarse_candidate"] = int(coarse)
        row["strict_candidate"] = int(coarse and abs(float(row["blocked_late_hz"])) <= ident["external_blocked_late_abs_hz"])
    return rows


def summarize(rows, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    seeds = sorted({int(r["seed"]) for r in rows})
    keys = sorted({(float(r["noise_multiplier"]), r["changed_groups"], r["membrane"]) for r in rows})
    index = {(float(r["noise_multiplier"]), r["changed_groups"], r["membrane"], int(r["seed"])): r for r in rows}
    draws = rng.integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    summary = []
    for multiplier, changed, membrane in keys:
        for metric in METRICS:
            values = np.array([float(index[multiplier, changed, membrane, seed][metric]) for seed in seeds])
            low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
            summary.append({"noise_multiplier": multiplier, "changed_groups": changed, "membrane": membrane, "metric": metric,
                            "seeds": len(seeds), "mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)})
    return summary


def make_figure(out, isolated, curves, summary, everyone, selected):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), layout="constrained")
    fig.suptitle("Regular (pacemaker-like) firing creates GPi inhibition without any inhibitory pathway",
                 fontsize=14, fontweight="bold")
    ax = axes[0, 0]
    for group, color in COLORS.items():
        rows = sorted((r for r in isolated if r["group"] == group), key=lambda r: float(r["noise_multiplier"]))
        ax.plot([float(r["isi_cv"]) for r in rows], [float(r["trough_5ms_hz"]) for r in rows], "o-", color=color, label=group)
    ax.axhline(0, color="#929ca9", lw=0.8, linestyle="--")
    ax.set(title="Isolated cells, no synapses: deepest 5-ms dip after a 4 mV kick",
           xlabel="ISI CV at matched firing rate", ylabel="Response vs sham (Hz)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    for multiplier, style in ((1.0, "-"), (3.0, "--"), (10.0, ":")):
        sample = [r for r in curves if r["group"] == "GPi" and float(r["noise_multiplier"]) == multiplier]
        time = np.array([float(r["time_ms"]) for r in sample])
        smooth = np.convolve([float(r["response_hz"]) for r in sample], np.ones(3) / 3, mode="same")
        ax.plot(time[1:-1], smooth[1:-1], style, color="#207c74", label=f"noise ×{multiplier:g}")
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.set(title="Isolated GPi response to one kick (3-ms average)", xlabel="Time after kick (ms)",
           ylabel="Response vs sham (Hz)", ylim=(-16, 40), xlim=(0, 60))
    ax.legend(frameon=False, fontsize=8)
    index = {(r["noise_multiplier"], r["changed_groups"], r["membrane"], r["metric"]): r for r in summary}
    multipliers = sorted({r["noise_multiplier"] for r in summary if r["changed_groups"] == everyone})
    ax = axes[1, 0]
    for metric, color, name in (("tonic_cv_mae", "#663f82", "CV error"), ("tonic_rate_mae_hz", "#937638", "Rate error (Hz)")):
        rows = [index[m, everyone, "E", metric] for m in multipliers]
        ax.errorbar(multipliers, [r["mean"] for r in rows], yerr=[[r["mean"] - r["ci_low"] for r in rows],
                    [r["ci_high"] - r["mean"] for r in rows]], marker="o", color=color, capsize=3, label=name)
    ax.axvline(selected, color="#a34b31", lw=0.8, linestyle=":", label=f"selected ×{selected:g} (grid edge)")
    ax.set(title="Network E: tonic error vs Gatev & Wichmann (selection input)", xlabel="Noise multiplier (GPe, STN, GPi)",
           ylabel="Mean absolute error")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for metric, color, name in (("pulse_early_hz", "#a34b31", "Early (3–13 ms)"), ("pulse_late_hz", "#437fb4", "Late, intact"),
                                ("blocked_late_hz", "#207c74", "Late, STN→GPe blocked")):
        rows = [index[m, everyone, "E", metric] for m in multipliers]
        ax.errorbar(multipliers, [r["mean"] for r in rows], yerr=[[r["mean"] - r["ci_low"] for r in rows],
                    [r["ci_high"] - r["mean"] for r in rows]], marker="o", color=color, capsize=3, label=name)
    ax.axhline(0, color="#929ca9", lw=0.8, linestyle="--")
    ax.set(title="Network E: GPi response to the STN pulse (not used for selection)", xlabel="Noise multiplier (GPe, STN, GPi)",
           ylabel="Response vs sham (Hz)")
    ax.set_yscale("symlog", linthresh=0.5, linscale=1.5)
    ticks = [-2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "pacemaker_artifact.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    rows = add_candidates(read_csv(out / "network.csv"), manifest["identifiability_protocol"])
    summary = summarize(rows, protocol)
    write_csv(out / "network_summary.csv", summary)
    figure_style()
    make_figure(out, read_csv(out / "isolated.csv"), read_csv(out / "isolated_curves.csv"), summary,
                "+".join(protocol["groups"]), selection["noise_multiplier"])
