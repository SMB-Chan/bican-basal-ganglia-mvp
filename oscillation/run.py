"""Exploratory loop-gain sweep: does the open-loop GPe-STN circuit produce PD-specific 8-15 Hz oscillation?

The frozen Stage 4A model is varied in three ways only: GPe->STN edge weights, STN->GPe edge weights and the
STN rebound amplitude. For each variant the normal-state GPe, STN and GPi biases are re-calibrated on seed 800
to the frozen model's target rates, so that loop gain changes rhythm rather than mean drive. Normal and
dopamine-depleted spontaneous activity are then simulated on 12 fresh seeds.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import apply_model_spec
from model import connect, population, simulate, ticks
from parkinsonian.run import _stack_populations, configure_pd_condition, oscillation_metrics
from run import trial_stimuli, write_csv
from validation.run_validation import patchseq_population


HERE = Path(__file__).resolve().parent
GROUPS = ("GPe", "STN", "GPi")
_CTX = {}


def variant_spec(base_spec, rebound_mv):
    spec = deepcopy(base_spec)
    if rebound_mv > 0:
        spec["groups"]["STN"]["rebound_mv"] = float(rebound_mv)
    else:
        spec["groups"]["STN"].pop("rebound_mv", None)
    return spec


def loop_weights(pop, graph, gpe_to_stn, stn_to_gpe, base=None):
    """Scale GPe->STN and STN->GPe edges only. `base` may already hold other edge changes."""
    weights = (np.broadcast_to(graph.weights[:, None], graph.targets.shape).astype(float).copy()
               if base is None else base.astype(float, copy=True))
    valid = graph.targets >= 0
    gpe, stn = pop.names.index("GPe"), pop.names.index("STN")
    post = np.where(valid, pop.cell_type[np.maximum(graph.targets, 0)], -1)
    weights[(pop.cell_type[:, None] == gpe) & (post == stn)] *= gpe_to_stn
    weights[(pop.cell_type[:, None] == stn) & (post == gpe)] *= stn_to_gpe
    return weights


def tonic_rates(pop, graph, sim, weights, tonic_spec, seed):
    params = {**sim, "baseline_ms": tonic_spec["duration_ms"], "cue_ms": 0}
    noise, initial = trial_stimuli(pop, params, seed, 1000)
    spikes = simulate(pop, graph, params, noise, initial, np.zeros(len(pop.cell_type)), edge_weights=weights)
    spikes = spikes[ticks(tonic_spec["burn_in_ms"], sim["dt_ms"]):]
    return {g: float(spikes[:, pop.cell_type == pop.names.index(g)].mean() * 1000 / sim["dt_ms"]) for g in GROUPS}


def calibrate(item):
    key, g_es, g_se, rebound_mv = item
    ctx = _CTX
    spec = variant_spec(ctx["base_spec"], rebound_mv)
    pop, cfg, sim = apply_model_spec(ctx["cal_base"], ctx["config"], spec)
    graph = connect("D", pop, cfg, ctx["cal_seed"])
    weights = loop_weights(pop, graph, g_es, g_se)
    bias = pop.bias.astype(float, copy=True)
    cal = ctx["protocol"]["calibration"]
    for it in range(cal["network_iterations"]):
        measured = tonic_rates(replace(pop, bias=bias), graph, sim, weights, ctx["tonic_spec"], ctx["cal_seed"])
        errors = {g: ctx["targets"][g] - measured[g] for g in GROUPS}
        if max(abs(e) for e in errors.values()) <= cal["tolerance_hz"]:
            break
        step = 0.35 if it < 6 else 0.25
        for g, err in errors.items():
            bias[pop.cell_type == pop.names.index(g)] += step * err
    measured = tonic_rates(replace(pop, bias=bias), graph, sim, weights, ctx["tonic_spec"], ctx["cal_seed"])
    biases = {g: float(bias[pop.cell_type == pop.names.index(g)][0]) for g in GROUPS}
    for g, value in biases.items():
        spec["groups"][g]["bias_mv"] = value
    return key, {"gpe_to_stn_scale": g_es, "stn_to_gpe_scale": g_se, "stn_rebound_mv": rebound_mv, "spec": spec,
                 "calibrated_rates_hz": measured,
                 "max_calibration_error_hz": max(abs(ctx["targets"][g] - measured[g]) for g in GROUPS)}


def evaluate_seed(seed):
    ctx = _CTX
    sp_spec = ctx["pd_protocol"]["spontaneous"]
    depletion = ctx["pd_protocol"]["dopamine_depletion"]
    base = patchseq_population(population(ctx["config"], ctx["atlas"], seed), ctx["patchseq"])
    rows = []
    for key, entry in ctx["variants"].items():
        pop, cfg, sim = apply_model_spec(base, ctx["config"], entry["spec"])
        graph = connect("D", pop, cfg, seed)
        normal_w = loop_weights(pop, graph, entry["gpe_to_stn_scale"], entry["stn_to_gpe_scale"])
        pd_pop, pd_base, _ = configure_pd_condition(pop, graph, depletion, "pd_intact")
        pd_w = loop_weights(pop, graph, entry["gpe_to_stn_scale"], entry["stn_to_gpe_scale"], base=pd_base)
        params = {**sim, "baseline_ms": sp_spec["duration_ms"], "cue_ms": 0}
        noise, initial = trial_stimuli(pop, params, seed, 4000)
        bpop, bgraph, bew = _stack_populations([pop, pd_pop], graph, [normal_w, pd_w])
        n = len(pop.cell_type)
        spikes = simulate(bpop, bgraph, params, np.tile(noise, (1, 2)), np.tile(initial, 2), np.zeros(2 * n),
                          edge_weights=bew)[ticks(sp_spec["burn_in_ms"], sim["dt_ms"]):].reshape(-1, 2, n)
        row = {"seed": seed, "variant": key, "gpe_to_stn_scale": entry["gpe_to_stn_scale"],
               "stn_to_gpe_scale": entry["stn_to_gpe_scale"], "stn_rebound_mv": entry["stn_rebound_mv"]}
        for ci, state in enumerate(("normal", "pd")):
            for g in GROUPS:
                m, _, _ = oscillation_metrics(spikes[:, ci, :], pop.cell_type == pop.names.index(g), sim["dt_ms"], sp_spec)
                for k in ("rate_hz", "isi_cv", "pop_beta_index", "unit_beta_index", "peak_freq_hz"):
                    row[f"{state}_{g}_{k}"] = m[k]
        d = ctx["protocol"]["definitions"]
        row["pd_specific_oscillation"] = int(
            row["pd_STN_pop_beta_index"] >= 1.5 and row["pd_GPi_pop_beta_index"] >= 1.5
            and row["pd_STN_pop_beta_index"] >= 1.25 * row["normal_STN_pop_beta_index"]
            and row["pd_GPi_pop_beta_index"] >= 1.25 * row["normal_GPi_pop_beta_index"]
            and row["normal_STN_pop_beta_index"] < 1.5)
        rows.append(row)
    print(f"oscillation seed {seed}: {len(ctx['variants'])} variants", flush=True)
    return rows


def pool_map(function, items):
    workers = min(len(items), os.cpu_count() or 1)
    if "fork" in mp.get_all_start_methods() and workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(function, items)
    return [function(item) for item in items]


def execute(config, protocol, pd_protocol, tonic_spec, patchseq, selection, calibration, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True)
    atlas, atlas_info = load_atlas(ROOT / "atlas_data")
    cal_seed = protocol["calibration"]["seed"]
    _CTX.update({"config": config, "protocol": protocol, "pd_protocol": pd_protocol, "tonic_spec": tonic_spec,
                 "patchseq": patchseq, "atlas": atlas, "base_spec": selection["selected_spec"], "cal_seed": cal_seed,
                 "cal_base": patchseq_population(population(config, atlas, cal_seed), patchseq),
                 "targets": calibration["target_tonic_rates_hz"]})
    grid = protocol["grid"]
    items = [(f"es{a:g}|se{b:g}|reb{r:g}", a, b, r) for r in grid["stn_rebound_mv"]
             for a in grid["gpe_to_stn_scale"] for b in grid["stn_to_gpe_scale"]]
    variants = dict(pool_map(calibrate, items))
    _CTX["variants"] = variants
    print(f"calibrated {len(variants)} variants", flush=True)
    rows = [row for chunk in pool_map(evaluate_seed, seeds) for row in chunk]
    write_csv(out / "evaluations.csv", rows)
    (out / "variants.json").write_text(json.dumps(variants, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
            load(ROOT / "validation/protocol.json")["tonic"], load(ROOT / "validation/patchseq_tau.json"),
            load(ROOT / "rebound/results/selection.json"), load(ROOT / "rebound/results/calibration.json"),
            args.out, args.seeds or protocol["seeds"])
