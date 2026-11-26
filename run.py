"""Run the paired atlas or synthetic basal ganglia experiment."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from model import connect, edge_matrix, graph_statistics, population, simulate, ticks


def trial_plan(sim, seed):
    plan = [(delta, side) for delta in sim["cue_deltas_mv"] for side in (0, 1) for _ in range(sim["trials_per_side"])]
    plan.extend((0.0, None) for _ in range(sim["equal_trials"]))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    return [plan[i] for i in rng.permutation(len(plan))]


def cue_input(pop, sim, delta, side):
    spn = np.isin(pop.cell_type, [pop.names.index("D1"), pop.names.index("D2")])
    stn = pop.cell_type == pop.names.index("STN")
    return sim["cue_common_mv"] * spn + sim["cue_stn_common_mv"] * stn + delta * (spn & (pop.channel == side))


def trial_stimuli(pop, sim, seed, trial):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3, trial]))
    n = len(pop.cell_type)
    steps = ticks(sim["baseline_ms"], sim["dt_ms"]) + ticks(sim["cue_ms"], sim["dt_ms"])
    initial = rng.uniform(sim["reset_mv"], sim["threshold_mv"], n)
    noise = rng.standard_normal((steps, n)) * sim["noise_mv_sqrt_ms"] * np.sqrt(sim["dt_ms"])
    if pop.noise_scale is not None:
        # Per-cell multipliers rescale the same draws, keeping sham and evoked trials paired.
        noise *= pop.noise_scale
    return noise, initial


def rates(spikes, sim):
    dt = sim["dt_ms"]
    cue_start = ticks(sim["baseline_ms"], dt)
    base_span = ticks(sim["baseline_readout_ms"], dt)
    cue_span = ticks(sim["cue_readout_ms"], dt)
    base = spikes[cue_start - base_span:cue_start].sum(axis=0) * (1000 / sim["baseline_readout_ms"])
    cue = spikes[-cue_span:].sum(axis=0) * (1000 / sim["cue_readout_ms"])
    return base, cue


def score_trial(gpi_rates, side):
    # Lower absolute GPi firing denotes greater disinhibition of that channel.
    a, b = gpi_rates
    chosen = (0 if a < b else 1) if not np.isclose(a, b, atol=1e-10, rtol=0) else None
    if side is None:
        return chosen, None, None
    margin = gpi_rates[1 - side] - gpi_rates[side]
    score = 0.5 if chosen is None else float(chosen == side)
    return chosen, score, margin


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record_network(path, pop, graph, spikes, sim, trial, delta, side):
    source = np.repeat(graph.sources, graph.targets.shape[1])
    target = graph.targets[graph.sources].ravel()
    times, cells = np.nonzero(spikes)
    np.savez_compressed(
        path, groups=np.array(pop.names), cell_type=pop.cell_type, channel=pop.channel,
        xyz_mm=pop.xyz, source=source, target=target, weight_mv=graph.weights[source],
        delay_ms=graph.delay[graph.sources].ravel() * sim["dt_ms"], length_mm=graph.lengths.ravel(),
        spike_neuron=cells, spike_time_ms=(times + 1) * sim["dt_ms"],
        cue_start_ms=sim["baseline_ms"], trial=trial, delta_mv=delta, cue_channel=side,
    )


def execute(config, config_path, out, data, mode, seeds, synthetic_control):
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    sim = config["simulation"]
    if not (0 < sim["baseline_readout_ms"] <= sim["baseline_ms"] and 0 < sim["cue_readout_ms"] <= sim["cue_ms"]):
        raise ValueError("Readout windows must fit inside their phases")
    if not sim["cue_deltas_mv"] or any(delta <= 0 for delta in sim["cue_deltas_mv"]):
        raise ValueError("At least one positive cue difference is required")
    if sim["trials_per_side"] < 1 or sim["equal_trials"] < 1:
        raise ValueError("At least one trial is required in each task cell")
    atlas, atlas_info = load_atlas(data) if mode == "atlas" else (None, None)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    graphs, matrices, group_rates, trials = [], [], [], []
    conditions = ["A", "B", "C", "D"]
    if mode == "atlas" and synthetic_control:
        conditions.append("D_synthetic")

    for seed in seeds:
        paired_pop = population(config, atlas, seed)
        plan = trial_plan(sim, seed)
        example = plan.index((max(sim["cue_deltas_mv"]), 0))
        for condition in conditions:
            pop = population(config, None, seed) if condition == "D_synthetic" else paired_pop
            graph = connect(condition[0], pop, config, seed)
            geometry = "synthetic" if atlas is None or condition == "D_synthetic" else "atlas"
            keys = {"seed": seed, "condition": condition, "geometry": geometry}
            graphs.append({**keys, **graph_statistics(pop, graph, config)})
            matrix = edge_matrix(pop, graph)
            n_types = len(pop.names)
            counts = np.bincount(pop.cell_type, minlength=n_types)
            for i, pre in enumerate(pop.names):
                for j, post in enumerate(pop.names):
                    opportunities = int(counts[i] * (counts[j] - (i == j)))
                    matrices.append({**keys, "source": pre, "target": post, "edges": int(matrix[i, j]),
                                     "possible_pairs": opportunities, "density": matrix[i, j] / opportunities})
            gpi = pop.cell_type == pop.names.index(config["output_group"])
            masks = [[(pop.cell_type == i) & (pop.channel == ch) for ch in (0, 1)] for i in range(n_types)]
            for trial, (delta, side) in enumerate(plan):
                noise, initial = trial_stimuli(pop, sim, seed, trial)
                spikes = simulate(pop, graph, sim, noise, initial, cue_input(pop, sim, delta, side))
                base, cue = rates(spikes, sim)
                base_gpi = [float(base[gpi & (pop.channel == ch)].mean()) for ch in (0, 1)]
                cue_gpi = [float(cue[gpi & (pop.channel == ch)].mean()) for ch in (0, 1)]
                choice, score, margin = score_trial(cue_gpi, side)
                trials.append({**keys, "trial": trial, "delta_mv": delta, "cue_channel": side,
                               "chosen_channel": choice, "tie": int(choice is None), "score": score,
                               "correct_margin_hz": margin, "gpi_base_ch0_hz": base_gpi[0],
                               "gpi_base_ch1_hz": base_gpi[1], "gpi_cue_ch0_hz": cue_gpi[0],
                               "gpi_cue_ch1_hz": cue_gpi[1], "global_base_hz": float(base.mean()),
                               "global_cue_hz": float(cue.mean())})
                for i, name in enumerate(pop.names):
                    for channel, mask in enumerate(masks[i]):
                        group_rates.append({**keys, "trial": trial, "delta_mv": delta, "cue_channel": side,
                                            "group": name, "channel": channel, "n": int(mask.sum()),
                                            "baseline_hz": float(base[mask].mean()), "cue_hz": float(cue[mask].mean()),
                                            "baseline_silent_fraction": float((base[mask] == 0).mean()),
                                            "cue_silent_fraction": float((cue[mask] == 0).mean())})
                if seed == seeds[0] and trial == example:
                    record_network(out / f"network_{condition}_seed{seed}.npz", pop, graph, spikes, sim, trial, delta, side)
            print(f"seed {seed:>2}  {condition:>11}  {len(plan)} trials", flush=True)

    for name, rows in (("graphs", graphs), ("edge_matrices", matrices), ("trials", trials), ("rates", group_rates)):
        write_csv(out / f"{name}.csv", rows)
    manifest = {
        "mode": mode, "atlas": atlas_info, "conditions": conditions, "seeds": seeds, "trials_per_graph": len(plan),
        "config": config, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "software": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                     "nibabel": version("nibabel") if mode == "atlas" else None},
        "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data", type=Path, default=ROOT / "atlas_data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("atlas", "synthetic"), default="atlas")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--no-synthetic-control", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    execute(config, args.config, args.out, args.data, args.mode, args.seeds or config["seeds"], not args.no_synthetic_control)


if __name__ == "__main__":
    sys.exit(main())
