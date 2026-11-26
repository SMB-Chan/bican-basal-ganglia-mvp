"""Compare measured membrane-property distributions on fixed basal ganglia graphs."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from heterogeneity.patchseq import PATCHSEQ_ASSETS, RHESUS, TYPES, load_cells, summarize
from model import Graph, connect, population, simulate, ticks
from run import trial_stimuli, write_csv
from validation.prepare_patchseq import BASE
from validation.run_validation import measure_choices, measure_pulse, measure_tonic, pulse_metrics


HERE = Path(__file__).resolve().parent


def measured_parameters(pop, cells, sim, mode, seed):
    if mode == "original":
        return pop
    tau = pop.tau.astype(float, copy=True)
    threshold = np.full(len(tau), sim["threshold_mv"], dtype=float)
    gain = np.ones(len(tau))
    for j, kind in enumerate(TYPES):
        use = pop.cell_type == pop.names.index(kind)
        group = np.array([cell.predictors() for cell in cells if cell.kind == kind
                          and (mode != "rhesus_joint" or cell.species == RHESUS)])
        median = np.median(group, axis=0)
        if mode == "median":
            tau[use] = median[0]
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, 6, j]))
        sampled = group[rng.integers(len(group), size=int(use.sum()))].copy()
        tau[use] = sampled[:, 0]
        if mode == "tau":
            continue
        if mode == "shuffled":
            rng.shuffle(sampled[:, 1])
            rng.shuffle(sampled[:, 2])
        gap_ratio = sampled[:, 1] / median[1]
        threshold[use] = sim["rest_mv"] + (sim["threshold_mv"] - sim["rest_mv"]) * gap_ratio
        gain[use] = sampled[:, 2] / median[2]
    if mode in ("median", "tau"):
        return replace(pop, tau=tau)
    return replace(pop, tau=tau, threshold_mv=threshold, bias=pop.bias * gain, cue_gain=gain)


def isolated_mean_rate(pop, use, sim, spec, seed, offset_mv=0.0):
    """Mean firing rate of the selected cells with no synapses, on a noise realization fixed by `seed`."""
    cells = np.flatnonzero(use)
    subset = replace(pop, cell_type=pop.cell_type[cells], channel=pop.channel[cells], xyz=pop.xyz[cells],
                     sign=pop.sign[cells], jump=pop.jump[cells], bias=pop.bias[cells], tau=pop.tau[cells],
                     threshold_mv=(np.full(len(cells), float(sim["threshold_mv"])) if pop.threshold_mv is None
                                   else pop.threshold_mv[cells]) + offset_mv,
                     cue_gain=None if pop.cue_gain is None else pop.cue_gain[cells],
                     noise_scale=None if pop.noise_scale is None else pop.noise_scale[cells])
    n = len(cells)
    params = {**sim, "baseline_ms": spec["duration_ms"], "cue_ms": 0}
    graph = Graph(np.array([], dtype=int), np.full((n, 1), -1), np.zeros(n), np.zeros((n, 1), dtype=int), np.empty((0, 1)))
    noise, initial = trial_stimuli(subset, params, seed, spec["trial"])
    spikes = simulate(subset, graph, params, noise, initial, np.zeros(n))[ticks(spec["burn_in_ms"], sim["dt_ms"]):]
    return float(spikes.mean() * 1000 / sim["dt_ms"])


def match_mean_rate(pop, reference, sim, spec, seed):
    """Shift each SPN group's thresholds by one constant so its synapse-free mean rate equals the reference.

    The shift keeps the measured spread and pairing of traits; only the group's average excitability moves.
    """
    threshold = (np.full(len(pop.tau), float(sim["threshold_mv"])) if pop.threshold_mv is None
                 else pop.threshold_mv.astype(float, copy=True))
    offsets = {}
    for j, kind in enumerate(TYPES):
        use = pop.cell_type == pop.names.index(kind)
        group_seed = seed * 10 + j
        target = isolated_mean_rate(reference, use, sim, spec, group_seed)
        low, high = spec["offset_search_mv"]
        rate = lambda offset: isolated_mean_rate(pop, use, sim, spec, group_seed, offset)
        if not rate(high) <= target <= rate(low):
            raise ValueError(f"{kind}: reference rate {target} Hz lies outside the offset search range")
        for _ in range(spec["iterations"]):
            middle = (low + high) / 2
            if rate(middle) > target:
                low = middle
            else:
                high = middle
        offsets[kind] = (low + high) / 2
        threshold[use] += offsets[kind]
    return replace(pop, threshold_mv=threshold), offsets


def passive_rheobase(gap_mv, resistance_mohm):
    return 1000 * gap_mv / resistance_mohm


def donor_validation(cells):
    observed = [cell for cell in cells if cell.rheobase_pa is not None]
    rows = []
    for fold, donor in enumerate(sorted({cell.donor for cell in observed}), 1):
        for kind in TYPES:
            held = [cell for cell in observed if cell.donor == donor and cell.kind == kind]
            if not held:
                continue
            train = [cell for cell in observed if cell.donor != donor and cell.kind == kind]
            training_response = np.array([cell.rheobase_pa for cell in train])
            passive_train = np.array([passive_rheobase(cell.gap_mv, cell.resistance_mohm) for cell in train])
            scale = np.median(training_response / passive_train)
            target = np.array([cell.rheobase_pa for cell in held])
            passive = np.array([passive_rheobase(cell.gap_mv, cell.resistance_mohm) for cell in held])
            predictions = {
                "train_median": np.full(len(held), np.median(training_response)),
                "cell_passive": passive,
                "calibrated_cell": scale * passive,
            }
            for model, prediction in predictions.items():
                error = prediction - target
                rows.append({"fold": fold, "species": held[0].species, "kind": kind, "model": model,
                             "held_cells": len(held), "mae_pa": float(np.abs(error).mean()),
                             "signed_error_pa": float(error.mean())})
    return rows


def experiment(config, protocol, validation, cells, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    atlas, atlas_info = load_atlas(ROOT / "atlas_data")
    out.mkdir(parents=True, exist_ok=True)
    sim = config["simulation"]
    tonic = validation["tonic"]
    pulse = validation["stn_pulse"]
    references = {key: {name: np.mean(values) for name, values in tonic[key].items()} for key in ("rate_hz", "isi_cv")}
    # Record every group, score only the externally referenced ones.
    recorded = {**tonic, "rate_hz": dict.fromkeys(config["groups"])}
    network, firing, offsets = [], [], []
    for seed in seeds:
        baseline = population(config, atlas, seed)
        populations = {}
        for mode in protocol["conditions"]:
            base_mode = mode.removesuffix("_matched")
            pop = measured_parameters(baseline, cells, sim, base_mode, seed)
            if mode != base_mode:
                pop, shift = match_mean_rate(pop, populations["median"], sim, protocol["rate_matching"], seed)
                offsets.extend({"seed": seed, "mode": mode, "group": kind, "threshold_offset_mv": value}
                               for kind, value in shift.items())
            populations[mode] = pop
        graphs = {name: connect(code, baseline, {**config, "matched_channel_gain": 1.0}, seed)
                  for name, code in zip(protocol["wiring"], ("C", "D"))}
        for wiring, graph in graphs.items():
            for mode, pop in populations.items():
                keys = {"seed": seed, "wiring": wiring, "mode": mode}
                everything = measure_tonic(pop, graph, sim, recorded, seed)
                firing.extend({**keys, "group": group, **values} for group, values in everything.items())
                measured = {group: everything[group] for group in tonic["rate_hz"]}
                errors = {key: float(np.mean([abs(values[key] - references[key][group]) for group, values in measured.items()]))
                          for key in references}
                response = measure_pulse(pop, graph, sim, pulse, seed)
                network.append({**keys, "neurons": len(pop.cell_type), "edges": len(graph.sources) * graph.targets.shape[1],
                                "graph_sha256": hashlib.sha256(graph.targets.tobytes() + graph.delay.tobytes()).hexdigest(),
                                "tonic_rate_mae_hz": errors["rate_hz"], "tonic_cv_mae": errors["isi_cv"],
                                **{f"{kind}_rate_hz": everything[kind]["rate_hz"] for kind in TYPES},
                                **pulse_metrics(response, sim["dt_ms"], pulse), **measure_choices(pop, graph, sim, seed)})
        print(f"seed {seed}: {len(graphs) * len(populations)} conditions", flush=True)
    for file, rows in (("network", network), ("tonic", firing), ("rate_matching", offsets), ("cell_cohort", summarize(cells)),
                       ("donor_validation", donor_validation(cells))):
        write_csv(out / f"{file}.csv", rows)
    manifest = {"seeds": seeds, "protocol": protocol, "external_validation": validation,
                "patchseq": {"release": "HMBA-Macaque-PatchSeq 20260228", "license": "CC BY-NC 4.0",
                             "source_base_url": BASE, "source_sha256": PATCHSEQ_ASSETS},
                "model_config": config, "atlas": atlas_info, "python": platform.python_version(), "numpy": np.__version__,
                "raw_csv_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.glob("*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "validation/protocol.json").read_text(encoding="utf-8"))
    cells = load_cells(args.cache, protocol["patchseq_qc"], fetch=args.fetch)
    experiment(config, protocol, validation, cells, args.out, args.seeds or protocol["seeds"])
