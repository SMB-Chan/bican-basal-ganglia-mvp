"""Summarize results using seed as the unit of replication and draw figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import write_csv


COLORS = ["#66788a", "#437fb4", "#207d73", "#a94d27", "#9b6995"]
SHORT = {"D_synthetic": "D (synthetic)"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def distribution(row, label, values):
    values = np.asarray(values, dtype=float)
    row[label + "_mean"] = float(values.mean())
    row[label + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None


def summary(out, manifest):
    graphs = read_csv(out / "graphs.csv")
    trials = read_csv(out / "trials.csv")
    rates = read_csv(out / "rates.csv")
    conditions = manifest["conditions"]
    seeds = manifest["seeds"]
    groups = list(manifest["config"]["groups"])
    deltas = manifest["config"]["simulation"]["cue_deltas_mv"]
    graph_summary, task_seed, task_summary, rate_summary = [], [], [], []
    for condition in conditions:
        graph_rows = [row for row in graphs if row["condition"] == condition]
        stats = {"condition": condition, "seeds": len(seeds)}
        for key in list(graph_rows[0])[3:]:
            distribution(stats, key, [row[key] for row in graph_rows])
        graph_summary.append(stats)
        for delta in [0.0, *deltas]:
            rows_by_seed = []
            for seed in seeds:
                samples = [row for row in trials if row["condition"] == condition and int(row["seed"]) == seed
                           and float(row["delta_mv"]) == delta]
                row = {"condition": condition, "seed": seed, "delta_mv": delta, "trials": len(samples),
                       "accuracy": np.mean([float(t["score"]) for t in samples]) if delta else None,
                       "correct_margin_hz": np.mean([float(t["correct_margin_hz"]) for t in samples]) if delta else None,
                       "choice_0_fraction": np.mean([0.5 if t["tie"] == "1" else float(t["chosen_channel"] == "0") for t in samples]),
                       "tie_fraction": np.mean([float(t["tie"]) for t in samples]),
                       "gpi_ch1_minus_ch0_hz": np.mean([float(t["gpi_cue_ch1_hz"]) - float(t["gpi_cue_ch0_hz"]) for t in samples])}
                task_seed.append(row)
                rows_by_seed.append(row)
            aggregate = {"condition": condition, "delta_mv": delta, "seeds": len(seeds),
                         "total_trials": sum(row["trials"] for row in rows_by_seed)}
            for key in ("accuracy", "correct_margin_hz", "choice_0_fraction", "tie_fraction", "gpi_ch1_minus_ch0_hz"):
                if rows_by_seed[0][key] is None:
                    aggregate[key + "_mean"] = aggregate[key + "_sd"] = None
                else:
                    distribution(aggregate, key, [row[key] for row in rows_by_seed])
            task_summary.append(aggregate)

        for group in groups:
            stats = {"condition": condition, "group": group, "seeds": len(seeds)}
            # Both channels contribute equally because group sizes are even and balanced.
            for key in ("baseline_hz", "cue_hz", "baseline_silent_fraction", "cue_silent_fraction"):
                by_seed = [np.mean([float(row[key]) for row in rates if row["condition"] == condition
                                    and row["group"] == group and int(row["seed"]) == seed]) for seed in seeds]
                distribution(stats, key, by_seed)
            rate_summary.append(stats)

    paired = []
    task_index = {(row["condition"], row["seed"], row["delta_mv"]): row for row in task_seed}
    for reference in [name for name in ("C", "D_synthetic") if name in conditions]:
        for seed in seeds:
            for delta in deltas:
                d = task_index["D", seed, delta]
                control = task_index[reference, seed, delta]
                paired.append({"reference": reference, "seed": seed, "delta_mv": delta,
                               "accuracy_D_minus_reference": d["accuracy"] - control["accuracy"],
                               "margin_hz_D_minus_reference": d["correct_margin_hz"] - control["correct_margin_hz"]})

    for name, rows in (("graph_summary", graph_summary), ("task_by_seed", task_seed), ("task_summary", task_summary),
                       ("rate_summary", rate_summary), ("paired_task_differences", paired)):
        write_csv(out / f"{name}.csv", rows)
    return graph_summary, task_seed, task_summary, rate_summary


def figure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.labelcolor": "#26354a",
                         "figure.facecolor": "white", "text.color": "#17263a", "savefig.dpi": 170})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def plot_wiring(out, stats, conditions):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.5), layout="constrained")
    fig.suptitle("Wiring statistics | equal per-cell out-degree", fontsize=15, fontweight="bold")
    specs = (("mean_length_mm", "Mean edge length (mm)"), ("mean_delay_ms", "Mean synaptic delay (ms)"),
             ("permitted_pathway_fraction", "Fraction on pathways used by D"), ("in_degree_sd", "In-degree SD across neurons"))
    for ax, (key, label) in zip(axes.flat, specs):
        ax.bar(range(len(conditions)), [row[key + "_mean"] for row in stats],
               yerr=[row[key + "_sd"] or 0 for row in stats], color=COLORS[:len(conditions)], capsize=3, width=0.63)
        ax.set_xticks(range(len(conditions)), [SHORT.get(c, c) for c in conditions])
        ax.set_title(label, loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "wiring.png")


def plot_matrices(out, manifest):
    groups = list(manifest["config"]["groups"])
    rows = read_csv(out / "edge_matrices.csv")
    index = {name: i for i, name in enumerate(groups)}
    matrices = np.zeros((4, len(groups), len(groups)))
    for row in rows:
        if row["condition"] in tuple("ABCD"):
            matrices["ABCD".index(row["condition"]), index[row["source"]], index[row["target"]]] += float(row["density"])
    matrices *= 100 / len(manifest["seeds"])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), layout="constrained", sharey=True)
    fig.suptitle("Directed connection density by source and target type", fontsize=15, fontweight="bold")
    for condition, matrix, ax in zip("ABCD", matrices, axes):
        image = ax.imshow(matrix, vmin=0, vmax=matrices.max(), cmap="Blues", interpolation="nearest")
        ax.set_title(condition)
        ax.set_xticks(range(len(groups)), groups, rotation=40, ha="right")
        ax.set_yticks(range(len(groups)), groups)
        ax.set_xlabel("Target")
    axes[0].set_ylabel("Source")
    fig.colorbar(image, ax=axes, shrink=0.70, label="% of possible directed pairs")
    save(fig, out / "projection_matrices.png")


def plot_rates(out, manifest, rows):
    conditions = manifest["conditions"]
    groups = list(manifest["config"]["groups"])
    index = {(row["condition"], row["group"]): row for row in rows}
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1), layout="constrained", sharey=True)
    fig.suptitle("Simulated firing rates by cell population", fontsize=15, fontweight="bold")
    for ax, phase in zip(axes, ("baseline", "cue")):
        width = 0.14
        for i, condition in enumerate(conditions):
            rows_for_group = [index[condition, group] for group in groups]
            key = phase + "_hz"
            ax.bar(x + (i - (len(conditions) - 1) / 2) * width,
                   [row[key + "_mean"] for row in rows_for_group], width=width,
                   yerr=[row[key + "_sd"] or 0 for row in rows_for_group], capsize=2,
                   color=COLORS[i], label=SHORT.get(condition, condition))
        ax.set_title(phase.capitalize(), loc="left")
        ax.set_xticks(x, groups)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Spikes per neuron per second (Hz)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out / "firing_rates.png")


def plot_task(out, manifest, task_seed, task):
    conditions = manifest["conditions"]
    deltas = manifest["config"]["simulation"]["cue_deltas_mv"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), layout="constrained")
    fig.suptitle("Two-channel selection | GPi firing-rate readout", fontsize=15, fontweight="bold")
    for ax, key, title, baseline in zip(axes, ("accuracy", "correct_margin_hz"),
                                            ("Balanced two-choice accuracy", "GPi competing minus cued (Hz)"), (0.5, 0)):
        ax.axhline(baseline, color="#7b8797", linestyle="--", linewidth=1)
        for i, condition in enumerate(conditions):
            dx = (i - (len(conditions) - 1) / 2) * 0.12
            for j, delta in enumerate(deltas):
                aggregate = next(row for row in task if row["condition"] == condition and row["delta_mv"] == delta)
                dots = [row[key] for row in task_seed if row["condition"] == condition and row["delta_mv"] == delta]
                wiggle = np.linspace(-0.03, 0.03, len(dots))
                ax.scatter(j + dx + wiggle, dots, s=16, color=COLORS[i], alpha=0.45, linewidths=0, zorder=2)
                ax.errorbar(j + dx, aggregate[key + "_mean"], yerr=aggregate[key + "_sd"] or 0,
                            color=COLORS[i], marker="D", ms=4, capsize=3, zorder=3,
                            label=SHORT.get(condition, condition) if j == 0 else "_nolegend_")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(range(len(deltas)), [f"{delta:g}" for delta in deltas])
        ax.set_xlim(-0.42, len(deltas) - 0.58)
        ax.set_xlabel("Cued channel bias advantage (mV)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(-0.10, 1.10)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out / "selection.png")


def plot_example(out, manifest):
    seed = manifest["seeds"][0]
    sim = manifest["config"]["simulation"]
    with np.load(out / f"network_D_seed{seed}.npz") as network:
        xyz, cell_type, channels = (network[key] for key in ("xyz_mm", "cell_type", "channel"))
        groups = network["groups"]
        times, cells = network["spike_time_ms"], network["spike_neuron"]
        cue_channel = int(network["cue_channel"])
        delta = float(network["delta_mv"])
    group_colors = ["#507f9b", "#5aa49c", "#939cb4", "#db8c44", "#8064a5", "#a7434a"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})
    title = "Simulated cells inside MRI masks" if manifest["mode"] == "atlas" else "Synthetic regional geometry"
    axes[0].set_title(title, loc="left", fontsize=11)
    axes[1].set_title(f"D spikes | seed {seed}, cue {cue_channel}, +{delta:g} mV", loc="left", fontsize=11)
    group_centers = []
    neuron_order = np.lexsort((np.arange(len(channels)), channels, cell_type))
    display_index = np.argsort(neuron_order)
    for i, group in enumerate(groups):
        mask = cell_type == i
        axes[0].scatter(xyz[mask, 1], xyz[mask, 2], s=7, color=group_colors[i], alpha=0.65, linewidths=0, label=group)
        for channel in (0, 1):
            use = (cell_type[cells] == i) & (channels[cells] == channel)
            axes[1].scatter(times[use], display_index[cells[use]], s=2, marker=".", linewidths=0, color=group_colors[i],
                            alpha=0.78 if channel == cue_channel else 0.35)
        end = display_index[mask].max()
        axes[1].axhline(end + 0.5, color="#bbc3cc", lw=0.55)
        group_centers.append(display_index[mask].mean())
    axes[0].set(xlabel="Anterior–posterior coordinate (mm)", ylabel="Superior–inferior coordinate (mm)", aspect="equal")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    axes[1].set(xlabel="Time (ms)", xlim=(0, sim["baseline_ms"] + sim["cue_ms"]), ylim=(len(cell_type), 0))
    axes[1].set_yticks(group_centers, groups)
    axes[1].axvline(sim["baseline_ms"], color="#17263a", linestyle="--", linewidth=1)
    axes[1].axvspan(sim["baseline_ms"] + sim["cue_ms"] - sim["cue_readout_ms"], sim["baseline_ms"] + sim["cue_ms"],
                   color="#c6d7e2", alpha=0.21, zorder=0)
    save(fig, out / "example_raster.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    path = args.results
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    graph, task_seed, task, rate = summary(path, manifest)
    figure_style()
    plot_wiring(path, graph, manifest["conditions"])
    plot_matrices(path, manifest)
    plot_rates(path, manifest, rate)
    plot_task(path, manifest, task_seed, task)
    plot_example(path, manifest)
