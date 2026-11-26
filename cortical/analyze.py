"""Summarize the held-out Stage 3 cortical benchmark with bootstrap intervals and plot results."""

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
    "intact_early_hz", "intact_inhibition_hz", "intact_late_hz",
    "intact_triphasic_signs", "non_synaptic_inh_hz", "indirect_feedforward_inh_hz",
    "direct_d1_inh_hz", "pathway_mediated_inhibition", "genuine_triphasic",
    "direct_only_no_early", "stn_block_early_drop_hz", "stn_block_late_drop_hz",
    "nambu_stn_block_pass", "gaba_block_inh_hz", "tachibana_gaba_block_pass", "all_causal_checks",
)
LATENCY_METRICS = ("intact_early_onset_ms", "intact_inhibition_onset_ms", "intact_late_onset_ms", "triphasic_latency_mae_ms")


def summarize(rows, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    seeds = sorted({int(r["seed"]) for r in rows})
    models = ("baseline_lif", "selected_biophysics")
    index = {(r["model"], int(r["seed"])): r for r in rows}
    draws = rng.integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    summary, paired = [], []
    for model in models:
        for metric in METRICS:
            vals = np.array([float(index[model, s][metric]) for s in seeds])
            lo, hi = np.quantile(vals[draws].mean(axis=1), [0.025, 0.975])
            summary.append({"model": model, "metric": metric, "n_valid": len(seeds),
                            "mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi)})
        for metric in LATENCY_METRICS:
            raw = [index[model, s][metric] for s in seeds if index[model, s][metric] != ""]
            vals = np.array([float(x) for x in raw])
            if len(vals) >= 2:
                sub_draws = rng.integers(0, len(vals), size=(protocol["bootstrap_replicates"], len(vals)))
                lo, hi = np.quantile(vals[sub_draws].mean(axis=1), [0.025, 0.975])
            elif len(vals) == 1:
                lo = hi = float(vals[0])
            else:
                lo = hi = None
            summary.append({"model": model, "metric": metric, "n_valid": len(vals),
                            "mean": float(vals.mean()) if len(vals) else None, "ci_low": lo, "ci_high": hi})
    for metric in METRICS:
        diff = np.array([float(index["selected_biophysics", s][metric]) - float(index["baseline_lif", s][metric]) for s in seeds])
        lo, hi = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
        paired.append({"metric": metric, "seeds": len(seeds), "diff_mean": float(diff.mean()),
                       "ci_low": float(lo), "ci_high": float(hi)})
    return summary, paired


def smooth3(values):
    return np.convolve(values, np.ones(3) / 3, mode="same")


def curve_for(curves, model, intervention, group="GPi"):
    sub = sorted((r for r in curves if r["model"] == model and r["intervention"] == intervention and r["group"] == group),
                 key=lambda r: float(r["time_ms"]))
    return np.array([float(r["time_ms"]) for r in sub]), np.array([float(r["response_hz"]) for r in sub])


def make_figure(out, summary, curves, protocol):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), layout="constrained")
    fig.suptitle("Held-out Stage 3 evaluation: cortical (M1) stimulation and pathway blockades",
                 fontsize=14, fontweight="bold")
    ref = protocol["iwamuro_m1_forelimb_gpi_ms"]
    ax = axes[0, 0]
    for model, color, label in (("baseline_lif", "#647b91", "Baseline LIF (F)"),
                                ("selected_biophysics", "#207c74", "Selected biophysical model")):
        t, y = curve_for(curves, model, "intact")
        ax.plot(t[1:-1], smooth3(y)[1:-1], color=color, lw=1.8, label=label)
    for phase, style, name in (("early_excitation", "--", "Iwamuro early (10.2 ms)"),
                               ("inhibition", ":", "Iwamuro inh (20.1 ms)"),
                               ("late_excitation", "-.", "Iwamuro late (27.8 ms)")):
        ax.axvline(ref[phase]["mean"], color="#a34b31", lw=0.9, linestyle=style, alpha=0.75, label=name)
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.set(title="Intact GPi response to M1 stimulation (3-ms average)", xlabel="Time after M1 pulse (ms)",
           ylabel="Response vs sham (Hz)", xlim=(0, 65))
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[0, 1]
    for cond, color, style, label in (
            ("intact", "#207c74", "-", "Intact (M1→STN + STR)"),
            ("hyperdirect_only", "#437fb4", "--", "M1→STN only (intact)"),
            ("hyperdirect_gpe_blocked", "#a34b31", ":", "M1→STN + STN→GPe blocked (artifact check)"),
            ("direct_only", "#663f82", "-.", "M1→STR only")):
        t, y = curve_for(curves, "selected_biophysics", cond)
        ax.plot(t[1:-1], smooth3(y)[1:-1], color=color, linestyle=style, lw=1.6, label=label)
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.axvspan(*protocol["windows"]["inhibition_ms"], color="#eef3f7", alpha=0.7)
    ax.set(title="Selected model: pathway & pacemaker controls", xlabel="Time after M1 pulse (ms)",
           ylabel="GPi response vs sham (Hz)", xlim=(0, 65))
    ax.legend(frameon=False, fontsize=7.5)

    idx = {(r["model"], r["metric"]): r for r in summary}
    ax = axes[1, 0]
    parts = (
        ("intact_inhibition_hz", "Intact\n(16–28 ms)"),
        ("non_synaptic_inh_hz", "Non-synaptic\n(M1→STN, GPe blk)"),
        ("indirect_feedforward_inh_hz", "Indirect\n(STN→GPe→GPi)"),
        ("direct_d1_inh_hz", "Direct\n(D1→GPi)"),
    )
    x = np.arange(len(parts))
    for i, (model, color, label) in enumerate((("baseline_lif", "#647b91", "Baseline LIF (F)"),
                                               ("selected_biophysics", "#207c74", "Selected biophysical model"))):
        means = [idx[model, m]["mean"] for m, _ in parts]
        lows = [idx[model, m]["mean"] - idx[model, m]["ci_low"] for m, _ in parts]
        highs = [idx[model, m]["ci_high"] - idx[model, m]["mean"] for m, _ in parts]
        ax.bar(x + (i - 0.5) * 0.34, means, width=0.32, yerr=[lows, highs], capsize=3, color=color, label=label)
    ax.axhline(0, color="#929ca9", lw=0.8, linestyle="--")
    ax.set_xticks(x, [label for _, label in parts])
    ax.set(title="Inhibition-window decomposition (16–28 ms)", ylabel="GPi rate change (Hz)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for cond, color, style, label in (
            ("intact", "#207c74", "-", "Intact"),
            ("stn_blocked", "#a34b31", "--", "STN blocked (Nambu et al.)"),
            ("gpi_gaba_blocked", "#937638", ":", "GPi GABA blocked (Tachibana et al.)")):
        t, y = curve_for(curves, "selected_biophysics", cond)
        ax.plot(t[1:-1], smooth3(y)[1:-1], color=color, linestyle=style, lw=1.7, label=label)
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.set(title="Selected model: causal blockade tests", xlabel="Time after M1 pulse (ms)",
           ylabel="GPi response vs sham (Hz)", xlim=(0, 65))
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "cortical_benchmark.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary, paired = summarize(read_csv(out / "evaluations.csv"), manifest["protocol"])
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_differences.csv", paired)
    figure_style()
    make_figure(out, summary, read_csv(out / "mean_curves.csv"), manifest["protocol"])
