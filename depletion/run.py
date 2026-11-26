"""Exploratory decomposition of the Stage 4B dopamine-depletion setting and of changes that would lower GPi firing.

Each condition switches the depletion components (D1 bias, D2 bias, GPe->STN weight) on or off and may add a
GPi bias shift or a gain on one GPi input pathway. All conditions of a seed share noise and graph.
"""

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import apply_model_spec
from heldout.metrics import unit_metrics
from model import connect, population, simulate, ticks
from parkinsonian.run import _stack_populations, configure_pd_condition
from run import trial_stimuli, write_csv
from validation.run_validation import patchseq_population


HERE = Path(__file__).resolve().parent
GROUPS = ("GPi", "GPe", "STN", "D1", "D2")
_CTX = {}


def build_condition(pop, graph, depletion, condition):
    """Population and edge weights for one condition."""
    spec = {**depletion,
            "d1_bias_shift_mv": depletion["d1_bias_shift_mv"] if condition["d1"] else 0.0,
            "d2_bias_shift_mv": depletion["d2_bias_shift_mv"] if condition["d2"] else 0.0,
            "gpe_to_stn_weight_scale": depletion["gpe_to_stn_weight_scale"] if condition["gpe_stn"] else 1.0}
    cpop, weights, _ = configure_pd_condition(pop, graph, spec, "pd_intact")
    gpi = cpop.cell_type == cpop.names.index("GPi")
    if condition.get("gpi_bias_mv"):
        bias = cpop.bias.astype(float, copy=True)
        bias[gpi] += float(condition["gpi_bias_mv"])
        cpop = replace(cpop, bias=bias)
    valid = graph.targets >= 0
    post = np.where(valid, cpop.cell_type[np.maximum(graph.targets, 0)], -1)
    for source, gain in condition.get("gpi_input_gain", {}).items():
        edges = (cpop.cell_type[:, None] == cpop.names.index(source)) & (post == cpop.names.index("GPi"))
        weights[edges] *= float(gain)
    return cpop, weights


def seed_rows(seed):
    ctx = _CTX
    protocol, spec_m = ctx["protocol"], ctx["metrics"]
    base = patchseq_population(population(ctx["config"], ctx["atlas"], seed), ctx["patchseq"])
    names = list(protocol["conditions"])
    rows = []
    for model, model_spec in ctx["models"].items():
        pop, cfg, sim = apply_model_spec(base, ctx["config"], model_spec)
        graph = connect("D", pop, cfg, seed)
        params = {**sim, "baseline_ms": protocol["duration_ms"], "cue_ms": 0}
        noise, initial = trial_stimuli(pop, params, seed, protocol["trial"])
        n = len(pop.cell_type)
        dt_s = sim["dt_ms"] / 1000.0
        burn = ticks(protocol["burn_in_ms"], sim["dt_ms"])
        n_windows = int((ticks(protocol["duration_ms"], sim["dt_ms"]) - burn) * dt_s // spec_m["window_s"])
        windows = burn * dt_s + np.column_stack((np.arange(n_windows), np.arange(n_windows) + 1)) * spec_m["window_s"]
        for start in range(0, len(names), protocol["conditions_per_stack"]):
            chunk = names[start:start + protocol["conditions_per_stack"]]
            built = [build_condition(pop, graph, ctx["depletion"], protocol["conditions"][c]) for c in chunk]
            bpop, bgraph, bew = _stack_populations([b[0] for b in built], graph, [b[1] for b in built])
            k = len(chunk)
            spikes = simulate(bpop, bgraph, params, np.tile(noise, (1, k)), np.tile(initial, k), np.zeros(k * n), edge_weights=bew)
            for ci, cname in enumerate(chunk):
                row = {"seed": seed, "model": model, "condition": cname}
                for g in GROUPS:
                    cells = np.flatnonzero(pop.cell_type == pop.names.index(g))
                    metrics = [unit_metrics((np.flatnonzero(spikes[:, ci * n + c]) + 0.5) * dt_s, windows, spec_m, restrict_to_span=False)
                               for c in cells]
                    for key in (("rate_hz", "isi_cv", "beta_index") if g == "GPi" else ("rate_hz",)):
                        vals = [m[key] for m in metrics if m is not None and np.isfinite(m[key])]
                        row[f"{g}_{key}"] = float(np.median(vals)) if vals else float("nan")
                rows.append(row)
    print(f"depletion seed {seed}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    atlas, _ = load_atlas(ROOT / "atlas_data")
    _CTX.update({"config": load(ROOT / "config.json"), "patchseq": load(ROOT / "validation/patchseq_tau.json"), "atlas": atlas,
                 "protocol": protocol, "metrics": load(ROOT / "heldout/protocol.json")["metrics"],
                 "depletion": load(ROOT / "parkinsonian/protocol.json")["dopamine_depletion"],
                 "models": {"baseline_lif": None, **{k: load(ROOT / v)["selected_spec"] for k, v in protocol["specs"].items()}}})
    workers = min(len(protocol["seeds"]), os.cpu_count() or 1)
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for chunk in pool.map(seed_rows, protocol["seeds"]) for r in chunk]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "conditions.csv", rows)
