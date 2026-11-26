"""Summarize the held-out Stage 4B parkinsonian benchmark and draw summary figures."""

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
    "normal_gpe_rate_hz", "pd_gpe_rate_hz", "normal_stn_rate_hz", "pd_stn_rate_hz",
    "normal_gpi_rate_hz", "pd_gpi_rate_hz", "pd_minus_normal_gpi_rate_hz",
    "normal_stn_beta_index", "pd_stn_beta_index", "pd_striatal_only_stn_beta_index",
    "normal_gpi_beta_index", "pd_gpi_beta_index", "pd_striatal_only_gpi_beta_index",
    "normal_gpe_beta_index", "pd_gpe_beta_index",
    "pd_stn_unit_beta_index", "pd_gpi_unit_beta_index",
    "pd_stn_peak_freq_hz", "pd_gpi_peak_freq_hz",
    "normal_gpi_beta_power_hz2", "pd_gpi_beta_power_hz2",
    "pd_stn_blk_gpi_beta_power_hz2", "pd_stn_blk_gpi_beta_index", "pd_stn_blk_gpi_rate_hz",
    "pd_stn_beta_power_hz2", "pd_gpe_blk_stn_beta_power_hz2",
    "pd_gpe_blk_stn_beta_index", "pd_gpe_blk_stn_rate_hz",
    "normal_cort_gpi_inh_hz", "pd_cort_gpi_inh_hz", "pd_stn_blk_cort_gpi_inh_hz",
    "normal_cort_gpi_late_hz", "pd_cort_gpi_late_hz", "pd_stn_blk_cort_gpi_late_hz",
    "normal_cort_gpe_late_hz", "pd_cort_gpe_late_hz",
    "normal_cort_stn_late_hz", "pd_cort_stn_late_hz",
    "pd_rate_shift_pass", "pd_beta_emergence_pass", "tachibana_stn_inactivation_pass",
    "tachibana_gpe_stn_block_pass", "chiken_pd_cortical_shift_pass", "chiken_stn_block_rescue_pass",
    "pd_checks_passed", "all_six_pd_checks",
)
MODELS = ("baseline_lif", "selected_biophysics", "selected_rebound")


def summarize(rows, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    seeds = sorted({int(r["seed"]) for r in rows})
    index = {(r["model"], int(r["seed"])): r for r in rows}
    draws = rng.integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    summary, paired = [], []
    for model in MODELS:
        for metric in METRICS:
            vals = np.array([float(index[model, s][metric]) for s in seeds])
            lo, hi = np.quantile(vals[draws].mean(axis=1), [0.025, 0.975])
            summary.append({
                "model": model, "metric": metric, "seeds": len(seeds),
                "mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi),
            })
    for lhs, rhs in (("selected_rebound", "selected_biophysics"), ("selected_rebound", "baseline_lif")):
        for metric in METRICS:
            diff = np.array([float(index[lhs, s][metric]) - float(index[rhs, s][metric]) for s in seeds])
            lo, hi = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
            paired.append({
                "comparison": f"{lhs}_minus_{rhs}", "metric": metric, "seeds": len(seeds),
                "diff_mean": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            })
    return summary, paired


def smooth3(values):
    return np.convolve(values, np.ones(3) / 3, mode="same")


def spectrum_for(spectra, model, condition, group="GPi"):
    sub = sorted((r for r in spectra if r["model"] == model and r["condition"] == condition and r["group"] == group),
                 key=lambda r: float(r["freq_hz"]))
    return np.array([float(r["freq_hz"]) for r in sub]), np.array([float(r["pop_psd_hz2_per_hz"]) for r in sub])


def curve_for(curves, model, condition, group="GPi"):
    sub = sorted((r for r in curves if r["model"] == model and r["condition"] == condition and r["group"] == group),
                 key=lambda r: float(r["time_ms"]))
    return np.array([float(r["time_ms"]) for r in sub]), np.array([float(r["response_hz"]) for r in sub])


def make_figure(out, summary, spectra, curves, protocol, cortical_protocol):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.9), layout="constrained")
    fig.suptitle("Held-out Stage 4B evaluation: parkinsonian 8–15 Hz oscillations and cortical response rescue",
                 fontsize=13.5, fontweight="bold")
    idx = {(r["model"], r["metric"]): r for r in summary}
    b_lo, b_hi = protocol["spontaneous"]["low_beta_band_hz"]

    ax = axes[0, 0]
    for model, cond, color, style, label in (
        ("selected_rebound", "normal_intact", "#437fb4", "--", "Rebound model: Normal"),
        ("selected_rebound", "pd_intact", "#a34b31", "-", "Rebound model: PD intact"),
        ("selected_rebound", "pd_stn_blocked", "#207c74", "-.", "Rebound model: PD + STN blocked"),
        ("selected_biophysics", "pd_intact", "#647b91", ":", "Passive biophysics: PD intact"),
    ):
        f, p = spectrum_for(spectra, model, cond, "GPi")
        use = (f >= 3.0) & (f <= 45.0)
        ax.plot(f[use], p[use], color=color, linestyle=style, lw=1.7, label=label)
    ax.axvspan(b_lo, b_hi, color="#f6e7df", alpha=0.65, label="Primate PD band (8–15 Hz)")
    ax.set(title="Spontaneous GPi population power spectrum", xlabel="Frequency (Hz)",
           ylabel="PSD (Hz² / Hz)", xlim=(3, 45))
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[0, 1]
    bars = (
        ("normal_gpi_beta_index", "GPi\nNormal"),
        ("pd_gpi_beta_index", "GPi\nPD intact"),
        ("pd_stn_blk_gpi_beta_index", "GPi\nPD+STN blk"),
        ("pd_stn_beta_index", "STN\nPD intact"),
        ("pd_gpe_blk_stn_beta_index", "STN\nPD+GPe→STN blk"),
    )
    x = np.arange(len(bars))
    for i, (model, color, label) in enumerate((
        ("baseline_lif", "#647b91", "Baseline LIF (F)"),
        ("selected_biophysics", "#437fb4", "Stage 2 passive biophysics"),
        ("selected_rebound", "#207c74", "Stage 4A STN-rebound"),
    )):
        means = [idx[model, m]["mean"] for m, _ in bars]
        lows = [idx[model, m]["mean"] - idx[model, m]["ci_low"] for m, _ in bars]
        highs = [idx[model, m]["ci_high"] - idx[model, m]["mean"] for m, _ in bars]
        ax.bar(x + (i - 1) * 0.25, means, width=0.23, yerr=[lows, highs], capsize=2.5, color=color, label=label)
    ax.axhline(1.0, color="#929ca9", lw=0.9, linestyle="--", label="Poisson baseline (1.0)")
    ax.set_xticks(x, [lbl for _, lbl in bars])
    ax.set(title="8–15 Hz oscillation index (Tachibana et al. 2011 tests)", ylabel="8–15 Hz / 55–150 Hz PSD ratio")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    ax = axes[1, 0]
    for cond, color, style, label in (
        ("normal_intact", "#437fb4", "--", "Normal intact"),
        ("pd_intact", "#a34b31", "-", "PD intact (diminished inhibition)"),
        ("pd_stn_blocked", "#207c74", "-.", "PD + STN blocked (inhibition unmasked)"),
    ):
        t, y = curve_for(curves, "selected_rebound", cond, "GPi")
        ax.plot(t[1:-1], smooth3(y)[1:-1], color=color, linestyle=style, lw=1.8, label=label)
    ax.axhline(0, color="#929ca9", lw=0.8)
    ax.axvspan(*cortical_protocol["windows"]["inhibition_ms"], color="#eef3f7", alpha=0.7, label="Direct-pathway window (16–28 ms)")
    ax.set(title="Rebound model: M1→GPi response in PD & STN blockade (Chiken et al. 2021)",
           xlabel="Time after M1 pulse (ms)", ylabel="GPi response vs sham (Hz)", xlim=(0, 65))
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1, 1]
    checks = (
        ("pd_rate_shift_pass", "1. GPe↓\nSTN↑"),
        ("pd_beta_emergence_pass", "2. 8–15 Hz\nemerges"),
        ("tachibana_stn_inactivation_pass", "3. STN blk\n↓GPi β & rate"),
        ("tachibana_gpe_stn_block_pass", "4. GPe blk\n↓STN β, ↑rate"),
        ("chiken_pd_cortical_shift_pass", "5. PD M1\nGPi inh↓, GPe↑"),
        ("chiken_stn_block_rescue_pass", "6. PD+STN blk\nunmasks inh"),
    )
    cx = np.arange(len(checks))
    for i, (model, color, label) in enumerate((
        ("baseline_lif", "#647b91", "Baseline LIF (F)"),
        ("selected_biophysics", "#437fb4", "Stage 2 passive biophysics"),
        ("selected_rebound", "#207c74", "Stage 4A STN-rebound"),
    )):
        means = [idx[model, m]["mean"] for m, _ in checks]
        lows = [idx[model, m]["mean"] - idx[model, m]["ci_low"] for m, _ in checks]
        highs = [idx[model, m]["ci_high"] - idx[model, m]["mean"] for m, _ in checks]
        ax.bar(cx + (i - 1) * 0.25, means, width=0.23, yerr=[lows, highs], capsize=2.5, color=color, label=label)
    ax.set_xticks(cx, [lbl for _, lbl in checks], fontsize=8.5)
    ax.set(title="Held-out pass fraction across 12 seeds (6 macaque PD tests)", ylabel="Fraction of seeds passing",
           ylim=(0, 1.12))
    ax.legend(frameon=False, fontsize=7.5)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "parkinsonian_benchmark.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary, paired = summarize(read_csv(out / "evaluations.csv"), manifest["protocol"])
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_differences.csv", paired)
    figure_style()
    make_figure(out, summary, read_csv(out / "mean_spectra.csv"),
                read_csv(out / "mean_cortical_curves.csv"),
                manifest["protocol"], manifest["cortical_protocol"])
