"""Positive control for the 8-15 Hz index: impose a known 12 Hz rhythm on STN or GPi of the frozen Stage 4A model.

For each seed and state (normal, dopamine-depleted) the 11 conditions (no rhythm, 5 STN amplitudes, 5 GPi
amplitudes) are simulated side by side on identical noise, plus synapse-free GPi cells as an intrinsic baseline.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import apply_model_spec
from model import Graph, connect, population, simulate, ticks
from parkinsonian.run import _stack_populations, configure_pd_condition, oscillation_metrics
from run import trial_stimuli, write_csv
from validation.run_validation import patchseq_population


HERE = Path(__file__).resolve().parent
GROUPS = ("GPe", "STN", "GPi")
METRICS = ("rate_hz", "isi_cv", "pop_beta_index", "unit_beta_index", "peak_freq_hz", "pop_beta_power_hz2")
_CTX = {}


def rhythm(pop, target, amplitude, frequency_hz, steps, dt):
    """Per-step input that makes the target cells' equilibrium potential oscillate by `amplitude` mV."""
    drive = np.zeros((steps, len(pop.cell_type)))
    if amplitude == 0:
        return drive
    use = pop.cell_type == pop.names.index(target)
    wave = amplitude * np.sin(2 * np.pi * frequency_hz * np.arange(steps) * dt / 1000.0)
    drive[:, use] = wave[:, None] * (1.0 - np.exp(-dt / pop.tau[use]))[None, :]
    return drive


def conditions(spec):
    rows = [("none", 0.0)]
    rows += [(target, a) for target in spec["targets"] for a in spec["amplitudes_mv"] if a > 0]
    return rows


def evaluate_seed(seed):
    ctx = _CTX
    sp = ctx["pd_protocol"]["spontaneous"]
    spec = ctx["protocol"]["imposed_rhythm"]
    base = patchseq_population(population(ctx["config"], ctx["atlas"], seed), ctx["patchseq"])
    pop, cfg, sim = apply_model_spec(base, ctx["config"], ctx["selection"]["selected_spec"])
    graph = connect("D", pop, cfg, seed)
    dt = sim["dt_ms"]
    params = {**sim, "baseline_ms": sp["duration_ms"], "cue_ms": 0}
    steps = ticks(sp["duration_ms"], dt)
    burn = ticks(sp["burn_in_ms"], dt)
    noise, initial = trial_stimuli(pop, params, seed, 4000)
    n = len(pop.cell_type)
    rows = []
    for state in spec["states"]:
        state_pop, weights, _ = configure_pd_condition(pop, graph, ctx["pd_protocol"]["dopamine_depletion"],
                                                       "normal_intact" if state == "normal" else "pd_intact")
        conds = conditions(spec)
        bpop, bgraph, bew = _stack_populations([state_pop] * len(conds), graph, [weights] * len(conds))
        inputs = np.hstack([noise + rhythm(state_pop, target, a, spec["frequency_hz"], steps, dt) for target, a in conds])
        spikes = simulate(bpop, bgraph, params, inputs, np.tile(initial, len(conds)), np.zeros(len(conds) * n),
                          edge_weights=bew)[burn:].reshape(-1, len(conds), n)
        for ci, (target, a) in enumerate(conds):
            row = {"seed": seed, "state": state, "target": target, "amplitude_mv": a}
            for g in GROUPS:
                m, _, _ = oscillation_metrics(spikes[:, ci, :], pop.cell_type == pop.names.index(g), dt, sp)
                row.update({f"{g}_{k}": m[k] for k in METRICS})
            rows.append(row)
    # Synapse-free GPi on the same noise: intrinsic baseline of the index.
    gpi = np.flatnonzero(pop.cell_type == pop.names.index("GPi"))
    iso_graph = Graph(np.array([], dtype=int), np.full((n, 1), -1), np.zeros(n), np.zeros((n, 1), dtype=int), np.empty((0, 1)))
    iso = simulate(pop, iso_graph, params, noise, initial, np.zeros(n))[burn:]
    m, _, _ = oscillation_metrics(iso, np.isin(np.arange(n), gpi), dt, sp)
    rows.append({"seed": seed, "state": "isolated", "target": "none", "amplitude_mv": 0.0,
                 **{f"GPi_{k}": m[k] for k in METRICS}})
    print(f"readout seed {seed}", flush=True)
    return rows


def execute(config, protocol, pd_protocol, patchseq, selection, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True)
    atlas, atlas_info = load_atlas(ROOT / "atlas_data")
    _CTX.update({"config": config, "protocol": protocol, "pd_protocol": pd_protocol, "patchseq": patchseq,
                 "selection": selection, "atlas": atlas})
    workers = min(len(seeds), os.cpu_count() or 1)
    if "fork" in mp.get_all_start_methods() and workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            chunks = pool.map(evaluate_seed, seeds)
    else:
        chunks = [evaluate_seed(s) for s in seeds]
    write_csv(out / "evaluations.csv", [row for chunk in chunks for row in chunk])
    manifest = {"seeds": seeds, "protocol": protocol, "pd_protocol": pd_protocol, "base_selection": selection,
                "model_config": config, "atlas": atlas_info, "python": platform.python_version(), "numpy": np.__version__,
                "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    execute(load(ROOT / "config.json"), protocol, load(ROOT / "parkinsonian/protocol.json"),
            load(ROOT / "validation/patchseq_tau.json"), load(ROOT / "rebound/results/selection.json"),
            args.out, args.seeds or protocol["seeds"])
