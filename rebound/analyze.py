"""Summarize Stage 4A normal-state STN rebound factorial and draw summary figures."""

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
    "isolated_stn_rebound_hz", "isolated_rebound_pass",
    "tonic_rate_mae_hz", "tonic_cv_mae",
    "pulse_early_hz", "pulse_late_hz", "blocked_late_hz", "pulse_features", "kita_strict_pass",
    "intact_early_hz", "intact_inhibition_hz", "intact_late_hz",
    "stn_m1_late_hz", "gpe_m1_late_hz", "non_synaptic_inh_hz",
    "stn_block_early_drop_hz", "stn_block_late_drop_hz", "nambu_stn_block_pass",
    "gaba_block_inh_hz", "tachibana_gaba_block_pass",
    "m1_all_causal_pass", "joint_normal_pass",
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
                "variant": variant, "pathway": meta["pathway"], "synapse": meta["synapse"],
                "rebound": meta["rebound"], "metric": metric, "seeds": len(seeds),
                "mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi),
            })
        lat_raw = [index[variant, s]["triphasic_latency_mae_ms"] for s in seeds
                   if index[variant, s]["triphasic_latency_mae_ms"] != ""]
        lat_vals = np.array([float(x) for x in lat_raw])
        if len(lat_vals) >= 2:
            sdraws = rng.integers(0, len(lat_vals), size=(protocol["bootstrap_replicates"], len(lat_vals)))
            lo, hi = np.quantile(lat_vals[sdraws].mean(axis=1), [0.025, 0.975])
        elif len(lat_vals) == 1:
            lo = hi = float(lat_vals[0])
        else:
            lo = hi = None
        summary.append({
            "variant": variant, "pathway": meta["pathway"], "synapse": meta["synapse"],
            "rebound": meta["rebound"], "metric": "triphasic_latency_mae_ms", "seeds": len(lat_vals),
            "mean": float(lat_vals.mean()) if len(lat_vals) else None, "ci_low": lo, "ci_high": hi,
        })
    return summary


def smooth3(values):
    return np.convolve(values, np.ones(3) / 3, mode="same")


def m1_curve(curves, variant, intervention, group="GPi"):
    sub = sorted((r for r in curves if r["variant"] == variant and r["intervention"] == intervention and r["group"] == group),
                 key=lambda r: float(r["time_ms"]))
    return np.array([float(r["time_ms"]) for r in sub]), np.array([float(r["response_hz"]) for r in sub])


def make_figure(out, summary, iso_curves, m1_curves, selection, protocol):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), layout="constrained")
    fig.suptitle("Stage 4A normal-state resolution: STN Cav3 rebound and GPe–STN loop routing",
                 fontsize=13.5, fontweight="bold")
    idx = {(r["variant"], r["metric"]): r for r in summary}

    ax = axes[0, 0]
    for reb, color, style, label in (
        ("passive", "#647b91", "--", "Passive LIF STN (0 mV)"),
        ("moderate_ifb", "#207c74", "-", "Moderate Cav3 IFB (75 mV)"),
        ("strong_ifb", "#a34b31", "-.", "Strong Cav3 IFB (120 mV)"),
    ):
        sub = sorted((r for r in iso_curves if r["rebound"] == reb), key=lambda r: float(r["time_ms"]))
        t = np.array([float(r["time_ms"]) for r in sub])
        y = smooth3([float(r["response_hz"]) for r in sub])
        ax.plot(t[1:-1], y[1:-1], color=color, linestyle=style, lw=1.7, label=label)
    h0, h1 = protocol["isolated_stn_rebound"]["hyperpol_window_ms"]
    ax.axvspan(h0, h1, color="#eef3f7", alpha=0.8, label="Hyperpolarizing barrage (20 ms)")
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.set(title="Isolated STN post-inhibitory rebound (Bevan & Wilson test)",
           xlabel="Time (ms)", ylabel="STN rate vs sham (Hz)", xlim=(180, 285))
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[0, 1]
    reb_order = ("passive", "moderate_ifb", "strong_ifb")
    reb_labels = ["Passive\n(0 mV)", "Moderate IFB\n(75 mV)", "Strong IFB\n(120 mV)"]
    rx = np.arange(len(reb_order))
    combos = (
        ("gpe_gpi_heavy|delta", "#647b91", "GPe→GPi heavy + delta"),
        ("gpe_gpi_heavy|physiological_37c", "#937638", "GPe→GPi heavy + 37°C syn"),
        ("stn_loop_dominant|delta", "#437fb4", "STN-loop + delta"),
        ("stn_loop_dominant|physiological_37c", "#207c74", "STN-loop + 37°C syn"),
    )
    for i, (prefix, color, label) in enumerate(combos):
        rows = [idx[f"{prefix}|{r}", "stn_block_late_drop_hz"] for r in reb_order]
        means = [r["mean"] for r in rows]
        lows = [r["mean"] - r["ci_low"] for r in rows]
        highs = [r["ci_high"] - r["mean"] for r in rows]
        ax.bar(rx + (i - 1.5) * 0.19, means, width=0.17, yerr=[lows, highs], capsize=2.5, color=color, label=label)
    ax.axhline(0.25, color="#a34b31", lw=0.9, linestyle="--", label="Nambu threshold (+0.25 Hz)")
    ax.axhline(0.0, color="#929ca9", lw=0.7)
    ax.set_xticks(rx, reb_labels)
    ax.set(title="Nambu et al. (2000): GPi late-excitation drop on STN block", ylabel="Intact − STN-blocked late GPi (Hz)")
    ax.legend(frameon=False, fontsize=7.2)

    ax = axes[1, 0]
    for i, (prefix, color, label) in enumerate(combos):
        kita = [idx[f"{prefix}|{r}", "kita_strict_pass"]["mean"] for r in reb_order]
        joint = [idx[f"{prefix}|{r}", "joint_normal_pass"]["mean"] for r in reb_order]
        ax.plot(rx, kita, linestyle=":", marker="s", ms=5, lw=1.3, color=color, alpha=0.75)
        ax.plot(rx, joint, linestyle="-", marker="o", ms=6, lw=1.8, color=color, label=f"{label} (joint)")
    ax.set_xticks(rx, reb_labels)
    ax.set(title="Normal-state joint pass rate (solid) vs Kita strict (dotted)",
           ylabel="Fraction of 12 seeds passing", ylim=(-0.05, 1.12))
    ax.legend(frameon=False, fontsize=7.3)

    ax = axes[1, 1]
    chosen = selection["variant"]
    for cond, color, style, label in (
        ("intact", "#207c74", "-", "Selected model: intact"),
        ("stn_blocked", "#a34b31", "--", "Selected model: STN blocked (Nambu)"),
        ("gpi_gaba_blocked", "#937638", ":", "Selected model: GPi GABA blocked (Tachibana)"),
    ):
        t, y = m1_curve(m1_curves, chosen, cond, "GPi")
        ax.plot(t[1:-1], smooth3(y)[1:-1], color=color, linestyle=style, lw=1.7, label=label)
    t_old, y_old = m1_curve(m1_curves, "gpe_gpi_heavy|delta|passive", "stn_blocked", "GPi")
    ax.plot(t_old[1:-1], smooth3(y_old)[1:-1], color="#647b91", linestyle="-.", lw=1.4,
            label="Stage 2 model: STN blocked (residual late exc.)")
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.axvspan(28, 50, color="#f6e7df", alpha=0.55)
    ax.set(title="M1→GPi causal blockade comparison (3-ms average)",
           xlabel="Time after M1 pulse (ms)", ylabel="GPi response vs sham (Hz)", xlim=(0, 65))
    ax.legend(frameon=False, fontsize=7.3)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "rebound_factorial.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    summary = summarize(read_csv(out / "evaluations.csv"), manifest["protocol"])
    write_csv(out / "summary.csv", summary)
    figure_style()
    make_figure(out, summary, read_csv(out / "isolated_rebound_curves.csv"),
                read_csv(out / "mean_m1_curves.csv"), selection, manifest["protocol"])
