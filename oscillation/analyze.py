"""Summarize the loop-gain sweep: positive variants, indices with seed-bootstrap intervals, and a heat map."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from run import write_csv


METRICS = ("normal_STN_pop_beta_index", "pd_STN_pop_beta_index", "normal_GPi_pop_beta_index", "pd_GPi_pop_beta_index",
           "pd_STN_peak_freq_hz", "normal_STN_rate_hz", "pd_STN_rate_hz", "normal_GPe_rate_hz", "pd_GPe_rate_hz",
           "normal_GPi_rate_hz", "pd_GPi_rate_hz", "normal_STN_isi_cv", "pd_STN_unit_beta_index", "pd_specific_oscillation")


def summarize(rows, protocol, variants):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    seeds = sorted({int(r["seed"]) for r in rows})
    draws = rng.integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    index = {(r["variant"], int(r["seed"])): r for r in rows}
    out = []
    for key, entry in variants.items():
        row = {"variant": key, "gpe_to_stn_scale": entry["gpe_to_stn_scale"], "stn_to_gpe_scale": entry["stn_to_gpe_scale"],
               "stn_rebound_mv": entry["stn_rebound_mv"], "max_calibration_error_hz": entry["max_calibration_error_hz"],
               "seeds": len(seeds)}
        for metric in METRICS:
            values = np.array([float(index[key, s][metric]) for s in seeds])
            low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
            row.update({metric: float(values.mean()), metric + "_ci_low": float(low), metric + "_ci_high": float(high)})
        row["positive_seeds"] = int(sum(int(index[key, s]["pd_specific_oscillation"]) for s in seeds))
        row["variant_positive"] = int(row["positive_seeds"] >= 6)
        row["calibration_failure"] = int(entry["max_calibration_error_hz"] > 2.0)
        out.append(row)
    return out


def make_figure(out, summary, grid):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), layout="constrained")
    fig.suptitle("Open-loop GPe–STN loop gain and 8–15 Hz oscillation (frozen Stage 4A model)", fontsize=14, fontweight="bold")
    a_vals, b_vals = grid["gpe_to_stn_scale"], grid["stn_to_gpe_scale"]
    specs = (("normal_STN_pop_beta_index", "STN index, normal"), ("pd_STN_pop_beta_index", "STN index, dopamine-depleted"),
             ("pd_GPi_pop_beta_index", "GPi index, dopamine-depleted"))
    for row_i, rebound in enumerate(grid["stn_rebound_mv"]):
        for col, (metric, title) in enumerate(specs):
            ax = axes[row_i, col]
            table = np.full((len(b_vals), len(a_vals)), np.nan)
            for r in summary:
                if r["stn_rebound_mv"] == rebound:
                    table[b_vals.index(r["stn_to_gpe_scale"]), a_vals.index(r["gpe_to_stn_scale"])] = r[metric]
            image = ax.imshow(table, origin="lower", cmap="viridis", vmin=0, vmax=max(2.0, np.nanmax(table)))
            for i in range(len(b_vals)):
                for j in range(len(a_vals)):
                    ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if table[i, j] < 1.2 else "black")
            ax.set_xticks(range(len(a_vals)), [f"×{v:g}" for v in a_vals])
            ax.set_yticks(range(len(b_vals)), [f"×{v:g}" for v in b_vals])
            ax.set(title=f"{title} | rebound {rebound:g} mV", xlabel="GPe→STN weight", ylabel="STN→GPe weight")
            fig.colorbar(image, ax=ax, shrink=0.8, label="8–15 Hz index (1 = Poisson)")
    save(fig, out / "loop_gain_oscillation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    variants = json.loads((out / "variants.json").read_text(encoding="utf-8"))
    summary = summarize(read_csv(out / "evaluations.csv"), manifest["protocol"], variants)
    write_csv(out / "summary.csv", summary)
    figure_style()
    make_figure(out, summary, manifest["protocol"]["grid"])
