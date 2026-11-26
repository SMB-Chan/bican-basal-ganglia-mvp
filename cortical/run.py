"""Evaluate pre-frozen models on the held-out Stage 3 cortical (M1) stimulation and blockade benchmark."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from model import connect, population, simulate, ticks
from run import trial_stimuli, write_csv
from validation.run_validation import patchseq_population


HERE = Path(__file__).resolve().parent
RECORDED_GROUPS = ("GPi", "GPe", "STN", "D1", "D2")


def apply_model_spec(pop, config, spec):
    """Apply a frozen model specification without consulting any cortical benchmark data."""
    if spec is None:
        return pop, {**config, "matched_channel_gain": 1.0}, dict(config["simulation"])
    cfg = {**config, "matched_channel_gain": 1.0}
    if spec.get("projection_propensity"):
        cfg["projection_propensity"] = {**config["projection_propensity"], **spec["projection_propensity"]}
    bias = pop.bias.astype(float, copy=True)
    jump = pop.jump.astype(float, copy=True)
    noise_scale = np.ones(len(bias)) if pop.noise_scale is None else pop.noise_scale.astype(float, copy=True)
    slow_noise = np.zeros(len(bias)) if pop.slow_noise_mv is None else pop.slow_noise_mv.astype(float, copy=True)
    rebound = np.zeros(len(bias)) if pop.rebound_mv is None else pop.rebound_mv.astype(float, copy=True)
    has_slow = pop.slow_noise_mv is not None
    has_rebound = pop.rebound_mv is not None
    for group, values in spec.get("groups", {}).items():
        mask = pop.cell_type == pop.names.index(group)
        if "bias_mv" in values:
            bias[mask] = float(values["bias_mv"])
        if "jump_mv" in values:
            jump[mask] = float(values["jump_mv"])
        if "noise_multiplier" in values:
            noise_scale[mask] = float(values["noise_multiplier"])
        if "slow_noise_mv" in values and float(values["slow_noise_mv"]) > 0:
            slow_noise[mask] = float(values["slow_noise_mv"])
            has_slow = True
        if "rebound_mv" in values and float(values["rebound_mv"]) > 0:
            rebound[mask] = float(values["rebound_mv"])
            has_rebound = True
    updated = replace(pop, bias=bias, jump=jump, noise_scale=noise_scale,
                      slow_noise_mv=slow_noise if has_slow else None,
                      rebound_mv=rebound if has_rebound else None)
    sim = {**config["simulation"], **spec.get("simulation", {})}
    return updated, cfg, sim


def block_edges(pop, graph, source_groups, target_groups):
    src = np.isin(pop.cell_type, [pop.names.index(g) for g in source_groups])
    dst = np.isin(pop.cell_type, [pop.names.index(g) for g in target_groups])
    weights = np.broadcast_to(graph.weights[:, None], graph.targets.shape).copy()
    blocked = src[:, None] & (graph.targets >= 0) & dst[graph.targets]
    weights[blocked] = 0.0
    return weights


def cortical_drive(pop, dt, spec, seed, include_stn=True, include_striatum=True):
    boundary = ticks(spec["stim_at_ms"], dt)
    after = ticks(spec["after_ms"], dt)
    total = boundary + after
    n = len(pop.cell_type)
    drive = np.zeros((total, n))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8]))
    stn = np.flatnonzero(pop.cell_type == pop.names.index("STN"))
    driven_stn = rng.choice(stn, round(len(stn) * spec["fraction_stimulated"]), replace=False)
    driven_str = []
    for name in spec["striatum"]["groups"]:
        members = np.flatnonzero(pop.cell_type == pop.names.index(name))
        driven_str.append(rng.choice(members, round(len(members) * spec["fraction_stimulated"]), replace=False))
    driven_str = np.concatenate(driven_str)

    if include_stn:
        s = spec["stn"]
        t0 = boundary + ticks(s["delay_ms"], dt)
        decay = np.exp(-dt / s["tau_ms"])
        k = np.arange(total - t0)[:, None]
        drive[t0:, driven_stn] += s["amplitude_mv"] * (1.0 - decay) * (decay ** k)
    if include_striatum:
        s = spec["striatum"]
        t0 = boundary + ticks(s["delay_ms"], dt)
        decay = np.exp(-dt / s["tau_ms"])
        k = np.arange(total - t0)[:, None]
        drive[t0:, driven_str] += s["amplitude_mv"] * (1.0 - decay) * (decay ** k)
    return drive


def onset_latencies(curve, dt, windows):
    w = max(1, ticks(windows["onset_smoothing_ms"], dt))
    smooth = np.convolve(curve, np.ones(w) / w, mode="same")
    thr = windows["onset_threshold_hz"]

    def first_crossing(span_ms, positive, min_step=0):
        lo, hi = (ticks(t, dt) for t in span_ms)
        lo = max(lo, min_step)
        if lo >= hi:
            return None
        segment = smooth[lo:hi]
        hits = np.flatnonzero(segment >= thr if positive else segment <= -thr)
        return float((lo + int(hits[0])) * dt) if len(hits) else None

    early = first_crossing(windows["early_search_ms"], True)
    inh_min = ticks(early, dt) + 1 if early is not None else 0
    inh = first_crossing(windows["inhibition_search_ms"], False, inh_min)
    late_min = ticks(inh, dt) + 1 if inh is not None else (ticks(early, dt) + 5 if early is not None else 0)
    late = first_crossing(windows["late_search_ms"], True, late_min)
    return {"early_onset_ms": early, "inhibition_onset_ms": inh, "late_onset_ms": late}


def window_metrics(curve, dt, windows):
    sl = lambda key: slice(*(ticks(t, dt) for t in windows[key]))
    early = float(curve[sl("early_excitation_ms")].mean())
    inh = float(curve[sl("inhibition_ms")].mean())
    late = float(curve[sl("late_excitation_ms")].mean())
    eps = windows["sign_tolerance_hz"]
    has_early = int(early > eps)
    has_inh = int(inh < -eps)
    has_late = int(late > eps)
    return {"early_hz": early, "inhibition_hz": inh, "late_hz": late,
            "has_early_excitation": has_early, "has_inhibition": has_inh, "has_late_excitation": has_late,
            "triphasic_signs": int(has_early and has_inh and has_late),
            **onset_latencies(curve, dt, windows)}


def evaluate_seed_model(pop, graph, sim, protocol, seed):
    stim = protocol["m1_stimulus"]
    dt = sim["dt_ms"]
    boundary = ticks(stim["stim_at_ms"], dt)
    after = ticks(stim["after_ms"], dt)
    params = {**sim, "baseline_ms": stim["stim_at_ms"], "cue_ms": stim["after_ms"]}
    masks = {name: pop.cell_type == pop.names.index(name) for name in RECORDED_GROUPS}
    drives = {
        "both": cortical_drive(pop, dt, stim, seed, True, True),
        "stn": cortical_drive(pop, dt, stim, seed, True, False),
        "str": cortical_drive(pop, dt, stim, seed, False, True),
    }
    specs = {
        "intact": ("both", None),
        "hyperdirect_only": ("stn", None),
        "hyperdirect_gpe_blocked": ("stn", block_edges(pop, graph, ("STN",), ("GPe",))),
        "direct_only": ("str", None),
        "d1_gpi_blocked": ("both", block_edges(pop, graph, ("D1",), ("GPi",))),
        "stn_blocked": ("both", block_edges(pop, graph, ("STN",), pop.names)),
        "gpi_gaba_blocked": ("both", block_edges(pop, graph, ("D1", "D2", "SIn", "GPe"), ("GPi",))),
    }
    accum = {cond: {g: np.zeros(after) for g in RECORDED_GROUPS} for cond in protocol["interventions"]}
    zeros = np.zeros(len(pop.cell_type))
    for trial in range(stim["repetitions"]):
        noise, initial = trial_stimuli(pop, params, seed, 3000 + trial)
        sham_intact = simulate(pop, graph, params, noise, initial, zeros)
        for cond in protocol["interventions"]:
            drive_key, weights = specs[cond]
            sham = sham_intact if weights is None else simulate(pop, graph, params, noise, initial, zeros, edge_weights=weights)
            evoked = simulate(pop, graph, params, noise + drives[drive_key], initial, zeros, edge_weights=weights)
            for g, mask in masks.items():
                accum[cond][g] += evoked[boundary:, mask].mean(axis=1) - sham[boundary:, mask].mean(axis=1)
    scale = 1000.0 / (dt * stim["repetitions"])
    return {cond: {g: vals * scale for g, vals in groups.items()} for cond, groups in accum.items()}


def summarize_seed(seed, model_name, responses, dt, protocol):
    win = protocol["windows"]
    ref = protocol["iwamuro_m1_forelimb_gpi_ms"]
    eps = win["sign_tolerance_hz"]
    by_cond = {cond: window_metrics(responses[cond]["GPi"], dt, win) for cond in protocol["interventions"]}
    intact = by_cond["intact"]
    hd_only = by_cond["hyperdirect_only"]
    hd_blk = by_cond["hyperdirect_gpe_blocked"]
    d_only = by_cond["direct_only"]
    d1_blk = by_cond["d1_gpi_blocked"]
    stn_blk = by_cond["stn_blocked"]
    gaba_blk = by_cond["gpi_gaba_blocked"]

    non_syn_inh = hd_blk["inhibition_hz"]
    indirect_inh = hd_only["inhibition_hz"] - hd_blk["inhibition_hz"]
    direct_inh = intact["inhibition_hz"] - d1_blk["inhibition_hz"]
    intact_inh_depth = max(0.0, -intact["inhibition_hz"])
    non_syn_depth = max(0.0, -non_syn_inh)
    non_syn_frac = (non_syn_depth / intact_inh_depth) if intact_inh_depth > 1e-9 else None
    pathway_inh = int(intact["has_inhibition"] == 1 and non_syn_depth <= protocol["non_synaptic_inhibition_max_fraction"] * intact_inh_depth)
    no_early_direct = int(d_only["early_hz"] <= eps)
    nambu_pass = int((intact["early_hz"] - stn_blk["early_hz"] > eps) and (intact["late_hz"] - stn_blk["late_hz"] > eps))
    tachibana_pass = int((gaba_blk["inhibition_hz"] >= -eps) and (gaba_blk["inhibition_hz"] - intact["inhibition_hz"] > eps))

    key_map = {"early_excitation": "early_onset_ms", "inhibition": "inhibition_onset_ms", "late_excitation": "late_onset_ms"}
    lat_errs = [abs(intact[col] - ref[name]["mean"]) for name, col in key_map.items() if intact[col] is not None]
    eval_row = {
        "seed": seed, "model": model_name,
        **{f"intact_{k}": v for k, v in intact.items()},
        "non_synaptic_inh_hz": non_syn_inh,
        "indirect_feedforward_inh_hz": indirect_inh,
        "direct_d1_inh_hz": direct_inh,
        "non_synaptic_inh_fraction": non_syn_frac,
        "pathway_mediated_inhibition": pathway_inh,
        "genuine_triphasic": int(intact["triphasic_signs"] == 1 and pathway_inh == 1),
        "direct_only_no_early": no_early_direct,
        "stn_block_early_drop_hz": intact["early_hz"] - stn_blk["early_hz"],
        "stn_block_late_drop_hz": intact["late_hz"] - stn_blk["late_hz"],
        "nambu_stn_block_pass": nambu_pass,
        "gaba_block_inh_hz": gaba_blk["inhibition_hz"],
        "tachibana_gaba_block_pass": tachibana_pass,
        "all_causal_checks": int(intact["triphasic_signs"] == 1 and pathway_inh == 1
                                 and no_early_direct == 1 and nambu_pass == 1 and tachibana_pass == 1),
        "triphasic_latency_mae_ms": float(np.mean(lat_errs)) if len(lat_errs) == 3 else None,
    }
    cond_rows = [{"seed": seed, "model": model_name, "intervention": cond, **metrics}
                 for cond, metrics in by_cond.items()]
    return eval_row, cond_rows


def execute(config, patchseq, protocol, biophysics_selection, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    atlas, atlas_info = load_atlas(ROOT / "atlas_data")
    models = {"baseline_lif": None, "selected_biophysics": biophysics_selection["selected_spec"]}
    dt = config["simulation"]["dt_ms"]
    after = ticks(protocol["m1_stimulus"]["after_ms"], dt)
    eval_rows, cond_rows = [], []
    curve_sum = {(m, c, g): np.zeros(after) for m in models for c in protocol["interventions"] for g in RECORDED_GROUPS}

    for seed in seeds:
        base_pop = patchseq_population(population(config, atlas, seed), patchseq)
        for model_name, spec in models.items():
            pop, cfg, sim = apply_model_spec(base_pop, config, spec)
            graph = connect("D", pop, cfg, seed)
            responses = evaluate_seed_model(pop, graph, sim, protocol, seed)
            erow, crows = summarize_seed(seed, model_name, responses, sim["dt_ms"], protocol)
            eval_rows.append(erow)
            cond_rows.extend(crows)
            for cond, groups in responses.items():
                for g, curve in groups.items():
                    curve_sum[model_name, cond, g] += curve / len(seeds)
        print(f"cortical seed {seed}: {len(models)} models x {len(protocol['interventions'])} interventions", flush=True)

    curve_rows = [
        {"model": m, "intervention": c, "group": g, "time_ms": t * dt, "response_hz": float(val)}
        for (m, c, g), arr in curve_sum.items() for t, val in enumerate(arr)
    ]
    for name, rows in (("evaluations", eval_rows), ("interventions", cond_rows), ("mean_curves", curve_rows)):
        write_csv(out / f"{name}.csv", rows)
    manifest = {
        "seeds": seeds, "protocol": protocol, "biophysics_selection": biophysics_selection,
        "model_config": config, "patchseq_summary": patchseq, "atlas": atlas_info,
        "python": platform.python_version(), "numpy": np.__version__,
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.csv"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biophysics", type=Path, default=ROOT / "biophysics/results/selection.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    execute(load(ROOT / "config.json"), load(ROOT / "validation/patchseq_tau.json"),
            protocol, load(args.biophysics), args.out, args.seeds or protocol["seeds"])
