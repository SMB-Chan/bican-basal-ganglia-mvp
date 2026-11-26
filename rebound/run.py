"""Stage 4A factorial test of STN Cav3 (IFB) post-inhibitory rebound and pallidal-subthalamic routing on normal-state benchmarks."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from biophysics.run import match_network_tonic
from cortical.run import RECORDED_GROUPS, apply_model_spec, block_edges, cortical_drive, summarize_seed
from model import Graph, Population, connect, population, simulate, ticks
from run import trial_stimuli, write_csv
from validation.run_validation import block_stn_to_gpe, measure_tonic, patchseq_population, pulse_metrics


HERE = Path(__file__).resolve().parent
_WORK_CTX = {}


def tile_copies(pop, graph, k, edge_weights_list=None):
    """Stack k independent copies of (pop, graph) for cache-friendly vectorized simulation."""
    n = len(pop.cell_type)
    tile1 = lambda x: None if x is None else np.tile(x, k)
    bpop = Population(
        pop.names, np.tile(pop.cell_type, k), np.tile(pop.channel, k), np.tile(pop.xyz, (k, 1)),
        tile1(pop.sign), tile1(pop.jump), tile1(pop.bias), tile1(pop.tau),
        threshold_mv=tile1(pop.threshold_mv), cue_gain=tile1(pop.cue_gain),
        noise_scale=tile1(pop.noise_scale), slow_noise_mv=tile1(pop.slow_noise_mv),
        rebound_mv=tile1(pop.rebound_mv),
    )
    offsets = (np.arange(k, dtype=np.int32) * n)[:, None]
    bsources = (offsets + graph.sources[None, :]).ravel()
    btargets = np.tile(graph.targets, (k, 1)).copy()
    valid = btargets >= 0
    rep_off = np.repeat(np.arange(k, dtype=np.int32) * n, n)[:, None]
    btargets[valid] += np.broadcast_to(rep_off, btargets.shape)[valid]
    bgraph = Graph(bsources, btargets, np.tile(graph.weights, k),
                   np.tile(graph.delay, (k, 1)), np.tile(graph.lengths, (k, 1)))
    if edge_weights_list is None:
        bew = None
    else:
        bew = np.vstack(edge_weights_list)
    return bpop, bgraph, bew


def fast_pulse_groups(pop, graph, sim, spec, seed, groups, pre_trials=None):
    """Bit-for-bit identical batched equivalent of validation.run_validation.measure_pulse_groups."""
    params = {**sim, "baseline_ms": spec["pulse_at_ms"], "cue_ms": spec["after_ms"]}
    boundary = ticks(spec["pulse_at_ms"], sim["dt_ms"])
    n_steps = ticks(spec["after_ms"], sim["dt_ms"])
    stn = np.flatnonzero(pop.cell_type == pop.names.index("STN"))
    masks = {name: pop.cell_type == pop.names.index(name) for name in groups}
    base_w = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
    blocked = block_stn_to_gpe(pop, graph)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    driven = rng.choice(stn, round(len(stn) * spec["fraction_stn"]), replace=False)

    reps = spec["repetitions"]
    if pre_trials is None:
        pre_trials = [trial_stimuli(pop, params, seed, 2000 + tr) for tr in range(reps)]
    # 4 runs per repetition: (sham_intact, evoked_intact, sham_blocked, evoked_blocked) x reps
    bpop, bgraph, bew = tile_copies(pop, graph, 4 * reps, [base_w, base_w, blocked, blocked] * reps)
    noise_cols, init_cols = [], []
    for noise, initial in pre_trials:
        pulse = noise.copy()
        pulse[boundary, driven] += spec["voltage_jump_mv"]
        noise_cols.extend((noise, pulse, noise, pulse))
        init_cols.extend((initial, initial, initial, initial))
    n = len(pop.cell_type)
    bspikes = simulate(bpop, bgraph, params, np.hstack(noise_cols), np.concatenate(init_cols),
                       np.zeros(4 * reps * n), edge_weights=bew)
    bspikes = bspikes[boundary:].reshape(n_steps, reps, 4, n)
    scale = 1000.0 / sim["dt_ms"]
    out = {}
    for name, mask in masks.items():
        m = bspikes[:, :, :, mask].mean(axis=3)  # (n_steps, reps, 4)
        intact = (m[:, :, 1] - m[:, :, 0]).mean(axis=1) * scale
        blk = (m[:, :, 3] - m[:, :, 2]).mean(axis=1) * scale
        out[name] = np.vstack((intact, blk))
    return out


def fast_m1_evaluate(pop, graph, sim, protocol, seed, pre_trials=None):
    """Bit-for-bit identical batched equivalent of cortical.run.evaluate_seed_model."""
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
    base_w = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
    w_hd_blk = block_edges(pop, graph, ("STN",), ("GPe",))
    w_d1_blk = block_edges(pop, graph, ("D1",), ("GPi",))
    w_stn_blk = block_edges(pop, graph, ("STN",), pop.names)
    w_gaba_blk = block_edges(pop, graph, ("D1", "D2", "SIn", "GPe"), ("GPi",))

    # 12 runs per trial:
    # 0: sham_intact, 1: intact_ev, 2: hd_only_ev, 3: dir_only_ev,
    # 4: hd_blk_sham, 5: hd_blk_ev, 6: d1_blk_sham, 7: d1_blk_ev,
    # 8: stn_blk_sham, 9: stn_blk_ev, 10: gaba_blk_sham, 11: gaba_blk_ev
    weights_per_trial = [
        base_w, base_w, base_w, base_w,
        w_hd_blk, w_hd_blk, w_d1_blk, w_d1_blk,
        w_stn_blk, w_stn_blk, w_gaba_blk, w_gaba_blk,
    ]
    bpop, bgraph, bew = tile_copies(pop, graph, 12, weights_per_trial)
    n = len(pop.cell_type)
    zeros = np.zeros(12 * n)
    reps = stim["repetitions"]
    if pre_trials is None:
        pre_trials = [trial_stimuli(pop, params, seed, 3000 + tr) for tr in range(reps)]

    pairs = {
        "intact": (1, 0),
        "hyperdirect_only": (2, 0),
        "direct_only": (3, 0),
        "hyperdirect_gpe_blocked": (5, 4),
        "d1_gpi_blocked": (7, 6),
        "stn_blocked": (9, 8),
        "gpi_gaba_blocked": (11, 10),
    }
    accum = {cond: {g: np.zeros(after) for g in RECORDED_GROUPS} for cond in protocol["interventions"]}
    for noise, initial in pre_trials:
        nb = np.hstack((
            noise, noise + drives["both"], noise + drives["stn"], noise + drives["str"],
            noise, noise + drives["stn"], noise, noise + drives["both"],
            noise, noise + drives["both"], noise, noise + drives["both"],
        ))
        ib = np.tile(initial, 12)
        sp = simulate(bpop, bgraph, params, nb, ib, zeros, edge_weights=bew)[boundary:].reshape(after, 12, n)
        for g, mask in masks.items():
            mg = sp[:, :, mask].mean(axis=2)
            for cond in protocol["interventions"]:
                ev_i, sh_i = pairs[cond]
                accum[cond][g] += mg[:, ev_i] - mg[:, sh_i]
    scale = 1000.0 / (dt * reps)
    return {cond: {g: vals * scale for g, vals in groups.items()} for cond, groups in accum.items()}


def measure_isolated_rebound(config, protocol, rebound_mv, seed):
    iso = protocol["isolated_stn_rebound"]
    fl = protocol["fluctuation_spec"]
    kin = protocol["rebound_kinetics"]
    stn_cfg = config["groups"]["STN"]
    dt = config["simulation"]["dt_ms"]
    n = iso["cells"]
    ones = np.ones(n)
    reb_arr = (rebound_mv * ones) if rebound_mv > 0 else None
    pop = Population(
        ("STN",), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros((n, 3)),
        stn_cfg["sign"] * ones, stn_cfg["jump_mv"] * ones, 16.0 * ones, stn_cfg["tau_ms"] * ones,
        noise_scale=fl["noise_multiplier"] * ones, slow_noise_mv=fl["slow_noise_mv"] * ones,
        rebound_mv=reb_arr,
    )
    graph = Graph(np.array([], dtype=int), np.full((n, 1), -1), np.zeros(n),
                  np.zeros((n, 1), dtype=int), np.empty((0, 1)))
    params = {**config["simulation"], "baseline_ms": iso["duration_ms"], "cue_ms": 0,
              "slow_noise_tau_ms": fl["slow_noise_tau_ms"], **kin}
    noise, initial = trial_stimuli(pop, params, seed, 5000)
    sham = simulate(pop, graph, params, noise, initial, np.zeros(n))
    hyp = noise.copy()
    h0, h1 = (ticks(t, dt) for t in iso["hyperpol_window_ms"])
    hyp[h0:h1, :] += float(iso["hyperpol_step_mv"])
    evoked = simulate(pop, graph, params, hyp, initial, np.zeros(n))
    diff_hz = (evoked.mean(axis=1) - sham.mean(axis=1)) * (1000.0 / dt)
    r0, r1 = (ticks(t, dt) for t in iso["readout_window_ms"])
    rebound_hz = float(diff_hz[r0:r1].mean())
    return rebound_hz, diff_hz


def build_variant_spec(pathway_spec, syn_spec, rebound_spec, fluct_spec, rebound_kinetics, biases=None):
    reb_mv = float(rebound_spec["rebound_mv"])
    groups = {
        "GPe": {
            "noise_multiplier": fluct_spec["noise_multiplier"],
            "slow_noise_mv": fluct_spec["slow_noise_mv"],
            "jump_mv": pathway_spec["gpe_jump_mv"],
        },
        "STN": {
            "noise_multiplier": fluct_spec["noise_multiplier"],
            "slow_noise_mv": fluct_spec["slow_noise_mv"],
            "jump_mv": pathway_spec["stn_jump_mv"],
            **({"rebound_mv": reb_mv} if reb_mv > 0 else {}),
        },
        "GPi": {
            "noise_multiplier": fluct_spec["noise_multiplier"],
            "slow_noise_mv": fluct_spec["slow_noise_mv"],
        },
    }
    if biases is not None:
        for g, val in biases.items():
            groups[g]["bias_mv"] = float(val)
    return {
        "projection_propensity": pathway_spec["projection_propensity"],
        "groups": groups,
        "simulation": {"slow_noise_tau_ms": fluct_spec["slow_noise_tau_ms"], **syn_spec, **rebound_kinetics},
    }


def select_rebound_variant(rows, rule):
    keys = sorted({(r["pathway"], r["synapse"], r["rebound"]) for r in rows})
    candidates = []
    for pathway, synapse, rebound in keys:
        sub = [r for r in rows if (r["pathway"], r["synapse"], r["rebound"]) == (pathway, synapse, rebound)]
        joint_frac = float(np.mean([int(r["joint_normal_pass"]) for r in sub]))
        kita_frac = float(np.mean([int(r["kita_strict_pass"]) for r in sub]))
        m1_frac = float(np.mean([int(r["m1_all_causal_pass"]) for r in sub]))
        rmae = float(np.mean([float(r["tonic_rate_mae_hz"]) for r in sub]))
        cvmae = float(np.mean([float(r["tonic_cv_mae"]) for r in sub]))
        lat_vals = [float(r["triphasic_latency_mae_ms"]) for r in sub if r["triphasic_latency_mae_ms"] not in (None, "")]
        lat_mae = float(np.mean(lat_vals)) if lat_vals else 20.0
        tie_score = -(cvmae + 0.1 * rmae + 0.05 * lat_mae)
        candidates.append((joint_frac, 0.5 * (kita_frac + m1_frac), tie_score,
                           pathway, synapse, rebound, kita_frac, m1_frac, rmae, cvmae, lat_mae))
    best = max(candidates)
    (_, _, _, pathway, synapse, rebound, kita_frac, m1_frac, rmae, cvmae, lat_mae) = best
    return {
        "pathway": pathway,
        "synapse": synapse,
        "rebound": rebound,
        "variant": f"{pathway}|{synapse}|{rebound}",
        "joint_normal_pass_fraction": best[0],
        "kita_strict_pass_fraction": kita_frac,
        "m1_all_causal_pass_fraction": m1_frac,
        "mean_tonic_rate_mae_hz": rmae,
        "mean_tonic_cv_mae": cvmae,
        "mean_triphasic_latency_mae_ms": lat_mae,
    }


def _calibrate_one(item):
    label, p_name, s_name, r_name, init_spec = item
    ctx = _WORK_CTX
    cpop, ccfg, csim = apply_model_spec(ctx["cal_base"], ctx["config"], init_spec)
    cgraph = connect("D", cpop, ccfg, ctx["cal_seed"])
    biases = match_network_tonic(cpop, cgraph, csim, ctx["tonic_spec"], ctx["protocol"]["groups"],
                                 ctx["target_tonic"], ctx["protocol"]["tonic_calibration"], ctx["cal_seed"])
    full_spec = build_variant_spec(
        ctx["protocol"]["pathway_modes"][p_name],
        ctx["protocol"]["synapse_modes"][s_name],
        ctx["protocol"]["rebound_modes"][r_name],
        ctx["protocol"]["fluctuation_spec"],
        ctx["protocol"]["rebound_kinetics"],
        biases=biases,
    )
    return label, {"pathway": p_name, "synapse": s_name, "rebound": r_name, "biases": biases, "spec": full_spec}


def _evaluate_one_seed(seed):
    ctx = _WORK_CTX
    config = ctx["config"]
    protocol = ctx["protocol"]
    cortical_protocol = ctx["cortical_protocol"]
    tonic_spec = ctx["tonic_spec"]
    pulse_spec = ctx["pulse_spec"]
    calibrated = ctx["calibrated"]
    refs = ctx["refs"]
    dt = config["simulation"]["dt_ms"]
    saved_conds = ("intact", "stn_blocked", "gpi_gaba_blocked")
    saved_groups = ("GPi", "GPe", "STN")

    iso_by_reb, iso_curves = {}, {}
    for r_name, r_spec in protocol["rebound_modes"].items():
        reb_hz, diff_curve = measure_isolated_rebound(config, protocol, float(r_spec["rebound_mv"]), seed)
        iso_by_reb[r_name] = reb_hz
        iso_curves[r_name] = diff_curve

    seed_base = patchseq_population(population(config, ctx["atlas"], seed), ctx["patchseq"])
    graphs_by_pathway = {}
    first_spec = next(iter(calibrated.values()))["spec"]
    ref_pop, _, ref_sim = apply_model_spec(seed_base, config, first_spec)
    for p_name, p_spec in protocol["pathway_modes"].items():
        tmp_spec = build_variant_spec(p_spec, protocol["synapse_modes"]["delta"], protocol["rebound_modes"]["passive"],
                                      protocol["fluctuation_spec"], protocol["rebound_kinetics"])
        tpop, tcfg, _ = apply_model_spec(seed_base, config, tmp_spec)
        graphs_by_pathway[p_name] = connect("D", tpop, tcfg, seed)

    # All 12 variants share identical tau, noise_multiplier, slow_noise_mv, and slow_noise_tau_ms
    pulse_params = {**ref_sim, "baseline_ms": pulse_spec["pulse_at_ms"], "cue_ms": pulse_spec["after_ms"]}
    pre_pulse = [trial_stimuli(ref_pop, pulse_params, seed, 2000 + tr) for tr in range(pulse_spec["repetitions"])]
    stim_m1 = cortical_protocol["m1_stimulus"]
    m1_params = {**ref_sim, "baseline_ms": stim_m1["stim_at_ms"], "cue_ms": stim_m1["after_ms"]}
    pre_m1 = [trial_stimuli(ref_pop, m1_params, seed, 3000 + tr) for tr in range(stim_m1["repetitions"])]
    tonic_params = {**ref_sim, "baseline_ms": tonic_spec["duration_ms"], "cue_ms": 0}
    tonic_noise, tonic_init = trial_stimuli(ref_pop, tonic_params, seed, 1000)
    tonic_burn = ticks(tonic_spec["burn_in_ms"], dt)
    tonic_dur_s = (ticks(tonic_spec["duration_ms"], dt) - tonic_burn) * dt / 1000.0

    seed_rows = []
    seed_m1_curves = {}
    late_sl = slice(*(ticks(t, dt) for t in cortical_protocol["windows"]["late_excitation_ms"]))
    for label, entry in calibrated.items():
        pop, _, sim = apply_model_spec(seed_base, config, entry["spec"])
        graph = graphs_by_pathway[entry["pathway"]]

        sp_tonic = simulate(pop, graph, {**sim, "baseline_ms": tonic_spec["duration_ms"], "cue_ms": 0},
                            tonic_noise, tonic_init, np.zeros(len(pop.cell_type)))[tonic_burn:]
        measured = {}
        for gname in tonic_spec["rate_hz"]:
            memb = np.flatnonzero(pop.cell_type == pop.names.index(gname))
            ivs = [np.diff(np.flatnonzero(sp_tonic[:, i])) for i in memb]
            val = [s.std() / s.mean() for s in ivs if len(s) >= tonic_spec["min_intervals"]]
            measured[gname] = {
                "rate_hz": float(sp_tonic[:, memb].mean() * len(sp_tonic) / tonic_dur_s),
                "isi_cv": float(np.mean(val)),
            }
        rmae = float(np.mean([abs(measured[g]["rate_hz"] - refs["rate_hz"][g]) for g in refs["rate_hz"]]))
        cvmae = float(np.mean([abs(measured[g]["isi_cv"] - refs["isi_cv"][g]) for g in refs["isi_cv"]]))

        pulse_resp = fast_pulse_groups(pop, graph, sim, pulse_spec, seed, ("GPi", "STN"), pre_trials=pre_pulse)
        pm = pulse_metrics(pulse_resp["GPi"], dt, pulse_spec)
        kita_strict = int(
            rmae <= protocol["selection"]["max_tonic_rate_mae_hz"]
            and pm["pulse_features"] == 4
            and abs(pm["blocked_late_hz"]) <= protocol["selection"]["max_blocked_late_abs_hz"]
        )

        m1_resp = fast_m1_evaluate(pop, graph, sim, cortical_protocol, seed, pre_trials=pre_m1)
        m1_eval, _ = summarize_seed(seed, label, m1_resp, dt, cortical_protocol)
        for c in saved_conds:
            for g in saved_groups:
                seed_m1_curves[label, c, g] = m1_resp[c][g]

        stn_m1_late_hz = float(m1_resp["intact"]["STN"][late_sl].mean())
        gpe_m1_late_hz = float(m1_resp["intact"]["GPe"][late_sl].mean())
        iso_hz = iso_by_reb[entry["rebound"]]
        iso_pass = int(iso_hz >= protocol["selection"]["min_isolated_rebound_hz"])
        m1_pass = int(m1_eval["all_causal_checks"])
        joint_pass = int(kita_strict == 1 and m1_pass == 1 and iso_pass == 1)

        seed_rows.append({
            "seed": seed,
            "variant": label,
            "pathway": entry["pathway"],
            "synapse": entry["synapse"],
            "rebound": entry["rebound"],
            "isolated_stn_rebound_hz": iso_hz,
            "isolated_rebound_pass": iso_pass,
            "tonic_rate_mae_hz": rmae,
            "tonic_cv_mae": cvmae,
            **{f"{g}_{k}": measured[g][k] for g in measured for k in ("rate_hz", "isi_cv")},
            "pulse_early_hz": pm["pulse_early_hz"],
            "pulse_late_hz": pm["pulse_late_hz"],
            "blocked_late_hz": pm["blocked_late_hz"],
            "pulse_features": pm["pulse_features"],
            "kita_strict_pass": kita_strict,
            "intact_early_hz": m1_eval["intact_early_hz"],
            "intact_inhibition_hz": m1_eval["intact_inhibition_hz"],
            "intact_late_hz": m1_eval["intact_late_hz"],
            "stn_m1_late_hz": stn_m1_late_hz,
            "gpe_m1_late_hz": gpe_m1_late_hz,
            "non_synaptic_inh_hz": m1_eval["non_synaptic_inh_hz"],
            "stn_block_early_drop_hz": m1_eval["stn_block_early_drop_hz"],
            "stn_block_late_drop_hz": m1_eval["stn_block_late_drop_hz"],
            "nambu_stn_block_pass": m1_eval["nambu_stn_block_pass"],
            "gaba_block_inh_hz": m1_eval["gaba_block_inh_hz"],
            "tachibana_gaba_block_pass": m1_eval["tachibana_gaba_block_pass"],
            "triphasic_latency_mae_ms": m1_eval["triphasic_latency_mae_ms"],
            "m1_all_causal_pass": m1_pass,
            "joint_normal_pass": joint_pass,
        })
    print(f"rebound seed {seed}: {len(calibrated)} variants evaluated", flush=True)
    return seed, seed_rows, iso_curves, seed_m1_curves


def execute(config, validation, patchseq, cortical_protocol, protocol, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    atlas, atlas_info = load_atlas(ROOT / "atlas_data")
    cal_seed = protocol["calibration_seed"]
    if cal_seed in seeds:
        raise ValueError("Calibration seed must be disjoint from evaluation seeds")

    tonic_spec = validation["tonic"]
    pulse_spec = validation["stn_pulse"]
    cal_base = patchseq_population(population(config, atlas, cal_seed), patchseq)
    cal_graph0 = connect("D", cal_base, {**config, "matched_channel_gain": 1.0}, cal_seed)
    target_tonic = {g: v["rate_hz"] for g, v in measure_tonic(cal_base, cal_graph0, config["simulation"], tonic_spec, cal_seed).items()}
    refs = {k: {g: float(np.mean(v)) for g, v in tonic_spec[k].items()} for k in ("rate_hz", "isi_cv")}

    cal_items = []
    for p_name, p_spec in protocol["pathway_modes"].items():
        for s_name, s_spec in protocol["synapse_modes"].items():
            for r_name, r_spec in protocol["rebound_modes"].items():
                label = f"{p_name}|{s_name}|{r_name}"
                init_spec = build_variant_spec(p_spec, s_spec, r_spec, protocol["fluctuation_spec"], protocol["rebound_kinetics"])
                cal_items.append((label, p_name, s_name, r_name, init_spec))

    _WORK_CTX.update({
        "atlas": atlas, "config": config, "patchseq": patchseq, "protocol": protocol,
        "cortical_protocol": cortical_protocol, "tonic_spec": tonic_spec, "pulse_spec": pulse_spec,
        "cal_seed": cal_seed, "cal_base": cal_base, "target_tonic": target_tonic, "refs": refs,
    })
    workers = min(len(cal_items), os.cpu_count() or 1)
    use_fork = "fork" in mp.get_all_start_methods() and workers > 1
    if use_fork:
        with mp.get_context("fork").Pool(workers) as pool:
            cal_pairs = pool.map(_calibrate_one, cal_items)
    else:
        cal_pairs = [_calibrate_one(it) for it in cal_items]
    calibrated = dict(cal_pairs)
    _WORK_CTX["calibrated"] = calibrated
    print(f"calibrated {len(calibrated)} rebound factorial variants on seed {cal_seed}", flush=True)

    seed_workers = min(len(seeds), os.cpu_count() or 1)
    if "fork" in mp.get_all_start_methods() and seed_workers > 1:
        with mp.get_context("fork").Pool(seed_workers) as pool:
            seed_results = pool.map(_evaluate_one_seed, seeds)
    else:
        seed_results = [_evaluate_one_seed(s) for s in seeds]

    dt = config["simulation"]["dt_ms"]
    iso_dur = ticks(protocol["isolated_stn_rebound"]["duration_ms"], dt)
    iso_curve_sum = {r_name: np.zeros(iso_dur) for r_name in protocol["rebound_modes"]}
    after_m1 = ticks(cortical_protocol["m1_stimulus"]["after_ms"], dt)
    saved_conds = ("intact", "stn_blocked", "gpi_gaba_blocked")
    saved_groups = ("GPi", "GPe", "STN")
    m1_curve_sum = {(lbl, c, g): np.zeros(after_m1) for lbl in calibrated for c in saved_conds for g in saved_groups}
    rows = []
    for _, srows, iso_c, m1_c in seed_results:
        rows.extend(srows)
        for r_name, arr in iso_c.items():
            iso_curve_sum[r_name] += arr / len(seeds)
        for k, arr in m1_c.items():
            m1_curve_sum[k] += arr / len(seeds)

    iso_rows = [
        {"rebound": r_name, "time_ms": t * dt, "response_hz": float(val)}
        for r_name, arr in iso_curve_sum.items() for t, val in enumerate(arr)
        if 170.0 <= t * dt <= 290.0
    ]
    m1_rows = [
        {"variant": lbl, "intervention": c, "group": g, "time_ms": t * dt, "response_hz": float(val)}
        for (lbl, c, g), arr in m1_curve_sum.items() for t, val in enumerate(arr)
    ]
    for name, data in (
        ("evaluations", rows),
        ("isolated_rebound_curves", iso_rows),
        ("mean_m1_curves", m1_rows),
    ):
        write_csv(out / f"{name}.csv", data)

    chosen = select_rebound_variant(rows, protocol["selection"])
    chosen_entry = calibrated[chosen["variant"]]
    selection = {
        **chosen,
        "selected_spec": chosen_entry["spec"],
        "rule": protocol["selection"],
    }
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps({
        "calibration_seed": cal_seed, "target_tonic_rates_hz": target_tonic, "variants": calibrated,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "seeds": seeds, "protocol": protocol, "cortical_protocol": cortical_protocol,
        "model_config": config, "validation": validation, "patchseq_summary": patchseq, "atlas": atlas_info,
        "python": platform.python_version(), "numpy": np.__version__,
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected rebound variant: {chosen['variant']} (joint_normal_pass={chosen['joint_normal_pass_fraction']:.2f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    execute(load(ROOT / "config.json"), load(ROOT / "validation/protocol.json"),
            load(ROOT / "validation/patchseq_tau.json"), load(ROOT / "cortical/protocol.json"),
            protocol, args.out, args.seeds or protocol["seeds"])
