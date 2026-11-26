"""Seed-bootstrap summary of each condition's change from the normal state, and a figure."""

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
METRICS = ("GPi_rate_hz", "GPi_isi_cv", "GPi_beta_index", "GPe_rate_hz", "STN_rate_hz", "D1_rate_hz", "D2_rate_hz")


def summarize(rows, protocol):
    stats = protocol["statistics"]
    seeds = sorted({int(r["seed"]) for r in rows})
    draws = np.random.default_rng(stats["bootstrap_seed"]).integers(0, len(seeds), size=(stats["replicates"], len(seeds)))
    index = {(r["model"], r["condition"], int(r["seed"])): r for r in rows}
    out = []
    for model in protocol["models"]:
        for cond in protocol["conditions"]:
            row = {"model": model, "condition": cond}
            for m in METRICS:
                base = np.array([float(index[model, "normal", s][m]) for s in seeds])
                value = np.array([float(index[model, cond, s][m]) for s in seeds])
                d = value - base
                low, high = np.quantile(d[draws].mean(axis=1), [0.025, 0.975])
                row.update({f"{m}_level": float(value.mean()), f"{m}_change": float(d.mean()),
                            f"{m}_ci_low": float(low), f"{m}_ci_high": float(high)})
            margin = stats["reliable_change_hz"]
            row["gpi_rate_direction"] = ("decrease" if row["GPi_rate_hz_ci_high"] < 0 and row["GPi_rate_hz_change"] <= -margin else
                                         "increase" if row["GPi_rate_hz_ci_low"] > 0 and row["GPi_rate_hz_change"] >= margin else
                                         "no reliable change")
            out.append(row)
    return out


def make_figure(out, summary, protocol):
    conds = [c for c in protocol["conditions"] if c != "normal"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2), layout="constrained", sharey=True)
    fig.suptitle("Change in GPi, GPe and STN firing from the normal state, by depletion component (exploratory)",
                 fontsize=13, fontweight="bold")
    colors = {"baseline_lif": "#66788a", "selected_biophysics": "#937638", "selected_rebound": "#207c74"}
    for ax, metric in zip(axes, ("GPi_rate_hz", "GPe_rate_hz", "STN_rate_hz")):
        for j, model in enumerate(protocol["models"]):
            rows = [next(r for r in summary if r["model"] == model and r["condition"] == c) for c in conds]
            y = np.arange(len(conds)) + (j - 1) * 0.25
            x = np.array([r[f"{metric}_change"] for r in rows])
            lo = np.array([r[f"{metric}_ci_low"] for r in rows])
            hi = np.array([r[f"{metric}_ci_high"] for r in rows])
            ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", ms=4, capsize=2, color=colors[model],
                        label=model.replace("selected_", ""))
        ax.axvline(0, color="#929ca9", lw=0.8, linestyle="--")
        ax.set(title=metric.replace("_rate_hz", " rate change (Hz)"), xlabel="Condition minus normal (Hz)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(np.arange(len(conds)), conds)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, out / "depletion_components.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    summary = summarize(read_csv(out / "conditions.csv"), protocol)
    write_csv(out / "summary.csv", summary)
    figure_style()
    make_figure(out, summary, protocol)
