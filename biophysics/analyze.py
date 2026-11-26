"""Summarize the 3x2x2 biophysical factorial and recruitment-matched decomposition."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from run import write_csv


METRICS = (
    "tonic_rate_mae_hz", "tonic_cv_mae", "GPe_rate_hz", "STN_rate_hz", "GPi_rate_hz",
    "GPe_isi_cv", "STN_isi_cv", "GPi_isi_cv", "pulse_early_hz", "pulse_late_hz",
    "blocked_early_hz", "blocked_late_hz", "late_attenuation_hz", "pulse_features",
    "stn_immediate_hz", "stn_blocked_late_hz", "coarse_candidate", "strict_candidate",
)


def summarize(rows, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    seeds = sorted({int(r["seed"]) for r in rows})
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    index = {(r["variant"], int(r["seed"])): r for r in rows}
    draws = rng.integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    summary = []
    for variant in variants:
        meta = index[variant, seeds[0]]
        for metric in METRICS:
            vals = np.array([float(index[variant, s][metric]) for s in seeds])
            lo, hi = np.quantile(vals[draws].mean(axis=1), [0.025, 0.975])
            summary.append({
                "variant": variant, "kind": meta["kind"], "pathway": meta["pathway"],
                "synapse": meta["synapse"], "fluctuation": meta["fluctuation"],
                "metric": metric, "seeds": len(seeds),
                "mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi),
            })
    return summary


def smooth3(values):
    return np.convolve(values, np.ones(3) / 3, mode="same")


def make_figure(out, summary, curves, selection):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), layout="constrained")
    fig.suptitle("Separating pacemaker reset from pathway-mediated inhibition", fontsize=14, fontweight="bold")
    idx = {(r["variant"], r["metric"]): r for r in summary}

    ax = axes[0, 0]
    fl_order = ("pacemaker", "moderate_ou", "high_cv_ou")
    fl_labels = ["Pacemaker\n(×1 white)", "Moderate OU\n(×2.2 + 9 mV slow)", "High-CV OU\n(×3 + GPe 32 mV slow)"]
    x = np.arange(len(fl_order))
    for metric, color, name in (("tonic_cv_mae", "#663f82", "Tonic CV MAE"),
                                ("strict_candidate", "#207c74", "Strict candidate fraction")):
        rows = [idx[f"indirect_balanced|exponential|{fl}", metric] for fl in fl_order]
        means = [r["mean"] for r in rows]
        yerr = [[r["mean"] - r["ci_low"] for r in rows], [r["ci_high"] - r["mean"] for r in rows]]
        ax.errorbar(x, means, yerr=yerr, marker="o", capsize=3, lw=1.6, color=color, label=name)
    ax.set_xticks(x, fl_labels)
    ax.set(title="Indirect + exponential: CV error vs strict pass rate", ylabel="Score / MAE", ylim=(-0.05, 1.15))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    combos = [
        ("baseline|delta", "Base + delta"),
        ("baseline|exponential", "Base + exp"),
        ("indirect_balanced|delta", "Indirect + delta"),
        ("indirect_balanced|exponential", "Indirect + exp"),
    ]
    cx = np.arange(len(combos))
    for i, (fl, color, label) in enumerate((("pacemaker", "#647b91", "Pacemaker (intact / blocked)"),
                                            ("moderate_ou", "#207c74", "Moderate OU (intact / blocked)"))):
        intact = [idx[f"{c}|{fl}", "pulse_late_hz"]["mean"] for c, _ in combos]
        blocked = [idx[f"{c}|{fl}", "blocked_late_hz"]["mean"] for c, _ in combos]
        offset = (i - 0.5) * 0.34
        ax.bar(cx + offset - 0.08, intact, width=0.15, color=color, alpha=0.95, label=f"{label.split(' (')[0]}: intact")
        ax.bar(cx + offset + 0.08, blocked, width=0.15, color=color, alpha=0.40, hatch="//", label=f"{label.split(' (')[0]}: blocked")
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.axhspan(-0.5, 0.5, color="#eef3f7", alpha=0.7)
    ax.set_xticks(cx, [name for _, name in combos])
    ax.set(title="Late GPi response (14–54 ms): intact vs STN→GPe blocked", ylabel="Response vs sham (Hz)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    ax = axes[1, 0]
    chosen = selection["variant"]
    for variant, color, title in (("baseline|delta|pacemaker", "#a34b31", "Baseline pacemaker (F)"),
                                  (chosen, "#207c74", "Selected biophysical")):
        sub = sorted((r for r in curves if r["variant"] == variant and r["group"] == "GPi"), key=lambda r: float(r["time_ms"]))
        t = np.array([float(r["time_ms"]) for r in sub])
        intact = smooth3([float(r["intact_hz"]) for r in sub])
        blocked = smooth3([float(r["blocked_hz"]) for r in sub])
        ax.plot(t[1:-1], intact[1:-1], "-", color=color, lw=1.7, label=f"{title}, intact")
        ax.plot(t[1:-1], blocked[1:-1], "--", color=color, lw=1.3, alpha=0.8, label=f"{title}, STN→GPe blocked")
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.axvspan(14, 54, color="#eef3f7", alpha=0.6)
    ax.set(title="GPi response to STN pulse (3-ms average)", xlabel="Time after STN pulse (ms)",
           ylabel="Response vs sham (Hz)", xlim=(0, 60))
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1, 1]
    decomp = [
        ("baseline|delta|pacemaker", "All pacemaker"),
        ("stn_only_unmatched", "STN mod-OU\n(12 mV pulse)"),
        ("stn_only_matched", "STN mod-OU\n(evoked-matched)"),
        ("gpi_only", "GPi mod-OU\n(12 mV pulse)"),
    ]
    dx = np.arange(len(decomp))
    rows = [idx[v, "blocked_late_hz"] for v, _ in decomp]
    means = [r["mean"] for r in rows]
    yerr = [[r["mean"] - r["ci_low"] for r in rows], [r["ci_high"] - r["mean"] for r in rows]]
    ax.bar(dx, means, width=0.5, yerr=yerr, capsize=3, color=["#647b91", "#937638", "#a34b31", "#437fb4"])
    ax.axhline(0, color="#929ca9", lw=0.8, linestyle="--")
    ax.set_xticks(dx, [label for _, label in decomp])
    ax.set(title="Blocked late GPi response: recruitment-matched decomposition", ylabel="Blocked late response (Hz)")

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "biophysics_factorial.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    summary = summarize(read_csv(out / "network.csv"), manifest["protocol"])
    write_csv(out / "summary.csv", summary)
    figure_style()
    make_figure(out, summary, read_csv(out / "mean_pulse_curves.csv"), selection)
