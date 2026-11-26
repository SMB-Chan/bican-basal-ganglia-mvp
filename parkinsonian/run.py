"""Blind held-out Stage 4B evaluation on awake MPTP parkinsonian macaque oscillations and cortical responses."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import RECORDED_GROUPS, apply_model_spec, window_metrics
from model import connect, population, simulate, ticks
from run import trial_stimuli, write_csv
from validation.run_validation import patchseq_population


HERE = Path(__file__).resolve().parent
SPONTANEOUS_GROUPS = ("GPe", "STN", "GPi", "D1", "D2")


def welch_psd(signal, dt_ms, window_ms, overlap_ms):
    """One-sided Welch PSD (in units^2 / Hz) along axis 0 for 1D or 2D signals."""
    arr = np.asarray(signal, dtype=float)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    nperseg = ticks(window_ms, dt_ms)
    step = nperseg - ticks(overlap_ms, dt_ms)
    if step <= 0 or len(arr) < nperseg:
        raise ValueError("Invalid Welch window/overlap configuration")
    fs = 1000.0 / dt_ms
    win = np.hanning(nperseg)[:, None]
    norm = fs * float((win[:, 0] ** 2).sum())
    starts = range(0, len(arr) - nperseg + 1, step)
    psd_acc = np.zeros((nperseg // 2 + 1, arr.shape[1]))
    for s in starts:
        seg = arr[s:s + nperseg]
        seg = (seg - seg.mean(axis=0, keepdims=True)) * win
        fft = np.fft.rfft(seg, axis=0)
        p = (np.abs(fft) ** 2) / norm
        if nperseg % 2 == 0:
            p[1:-1] *= 2.0
        else:
            p[1:] *= 2.0
        psd_acc += p
    psd = psd_acc / len(starts)
    freqs = np.fft.rfftfreq(nperseg, d=dt_ms / 1000.0)
    return freqs, (psd[:, 0] if squeeze else psd)


def oscillation_metrics(spikes, mask, dt_ms, spec):
    """Rate-invariant 8–15 Hz spectral indices and firing statistics for a cell group."""
    sub = spikes[:, mask]
    scale_hz = 1000.0 / dt_ms
    pop_rate = sub.mean(axis=1) * scale_hz
    rate_hz = float(pop_rate.mean())
    freqs, pop_psd = welch_psd(pop_rate, dt_ms, spec["welch_window_ms"], spec["welch_overlap_ms"])
    df = float(freqs[1] - freqs[0])
    b_lo, b_hi = spec["low_beta_band_hz"]
    t_lo, t_hi = spec["total_band_hz"]
    r_lo, r_hi = spec["poisson_ref_band_hz"]
    beta_m = (freqs >= b_lo) & (freqs <= b_hi)
    tot_m = (freqs >= t_lo) & (freqs <= t_hi)
    ref_m = (freqs >= r_lo) & (freqs <= r_hi)

    pop_beta_power = float(pop_psd[beta_m].sum() * df)
    pop_ref_mean = float(pop_psd[ref_m].mean())
    pop_beta_index = float(pop_psd[beta_m].mean() / max(pop_ref_mean, 1e-12))
    pop_beta_frac = float(pop_psd[beta_m].sum() / max(pop_psd[tot_m].sum(), 1e-12))
    peak_freq = float(freqs[tot_m][int(np.argmax(pop_psd[tot_m]))])

    counts = sub.sum(axis=0)
    active = np.flatnonzero(counts >= spec["min_spikes_single_unit"])
    if len(active):
        _, unit_psd = welch_psd(sub[:, active].astype(float) * scale_hz, dt_ms,
                                spec["welch_window_ms"], spec["welch_overlap_ms"])
        unit_ref = np.maximum(unit_psd[ref_m].mean(axis=0), 1e-12)
        unit_beta_index = float((unit_psd[beta_m].mean(axis=0) / unit_ref).mean())
    else:
        unit_beta_index = 1.0

    cvs = []
    for col in range(sub.shape[1]):
        isi = np.diff(np.flatnonzero(sub[:, col]))
        if len(isi) >= 5:
            cvs.append(float(isi.std() / isi.mean()))
    isi_cv = float(np.mean(cvs)) if cvs else float("nan")

    return {
        "rate_hz": rate_hz,
        "isi_cv": isi_cv,
        "pop_beta_power_hz2": pop_beta_power,
        "pop_beta_index": pop_beta_index,
        "pop_beta_fraction": pop_beta_frac,
        "unit_beta_index": unit_beta_index,
        "peak_freq_hz": peak_freq,
    }, freqs, pop_psd


def configure_pd_condition(pop, graph, pd_spec, condition):
    """Apply pre-registered dopamine-depletion state and selective pathway blockades."""
    if condition == "normal_intact":
        return pop, None, {"D1": 1.0, "D2": 1.0, "SIn": 1.0}
    bias = pop.bias.astype(float, copy=True)
    d1_idx = pop.names.index("D1")
    d2_idx = pop.names.index("D2")
    gpe_idx = pop.names.index("GPe")
    stn_idx = pop.names.index("STN")
    bias[pop.cell_type == d1_idx] += float(pd_spec["d1_bias_shift_mv"])
    bias[pop.cell_type == d2_idx] += float(pd_spec["d2_bias_shift_mv"])
    updated_pop = replace(pop, bias=bias)

    weights = np.broadcast_to(graph.weights[:, None], graph.targets.shape).astype(float).copy()
    valid = graph.targets >= 0
    gpe_to_stn = (pop.cell_type[:, None] == gpe_idx) & valid & (pop.cell_type[graph.targets] == stn_idx)
    if condition != "pd_striatal_only":
        weights[gpe_to_stn] *= float(pd_spec["gpe_to_stn_weight_scale"])
    if condition == "pd_stn_blocked":
        weights[(pop.cell_type[:, None] == stn_idx) & valid] = 0.0
    elif condition == "pd_gpe_to_stn_blocked":
        weights[gpe_to_stn] = 0.0
    elif condition not in ("pd_intact", "pd_striatal_only"):
        raise ValueError(f"Unknown PD condition {condition}")

    gains = {"D1": float(pd_spec["d1_cortical_gain"]), "D2": float(pd_spec["d2_cortical_gain"]), "SIn": 1.0}
    return updated_pop, weights, gains


def cortical_drive_pd(pop, dt, stim_spec, seed, striatal_gains):
    boundary = ticks(stim_spec["stim_at_ms"], dt)
    after = ticks(stim_spec["after_ms"], dt)
    total = boundary + after
    drive = np.zeros((total, len(pop.cell_type)))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8]))
    stn = np.flatnonzero(pop.cell_type == pop.names.index("STN"))
    driven_stn = rng.choice(stn, round(len(stn) * stim_spec["fraction_stimulated"]), replace=False)
    driven_by_group = {}
    for name in stim_spec["striatum"]["groups"]:
        members = np.flatnonzero(pop.cell_type == pop.names.index(name))
        driven_by_group[name] = rng.choice(members, round(len(members) * stim_spec["fraction_stimulated"]), replace=False)

    s = stim_spec["stn"]
    t0 = boundary + ticks(s["delay_ms"], dt)
    decay = np.exp(-dt / s["tau_ms"])
    k = np.arange(total - t0)[:, None]
    drive[t0:, driven_stn] += s["amplitude_mv"] * (1.0 - decay) * (decay ** k)

    st = stim_spec["striatum"]
    t0_s = boundary + ticks(st["delay_ms"], dt)
    decay_s = np.exp(-dt / st["tau_ms"])
    ks = np.arange(total - t0_s)[:, None]
    base_wave = st["amplitude_mv"] * (1.0 - decay_s) * (decay_s ** ks)
    for name, cols in driven_by_group.items():
        drive[t0_s:, cols] += striatal_gains.get(name, 1.0) * base_wave
    return drive


def _stack_populations(pops, graph, weights_list):
    k = len(pops)
    n = len(pops[0].cell_type)
    cat1 = lambda attr: (
        None if all(getattr(p, attr) is None for p in pops)
        else np.concatenate([np.zeros(n) if getattr(p, attr) is None else getattr(p, attr) for p in pops])
    )
    bpop = replace(
        pops[0],
        cell_type=np.tile(pops[0].cell_type, k),
        channel=np.tile(pops[0].channel, k),
        xyz=np.tile(pops[0].xyz, (k, 1)),
        sign=np.concatenate([p.sign for p in pops]),
        jump=np.concatenate([p.jump for p in pops]),
        bias=np.concatenate([p.bias for p in pops]),
        tau=np.concatenate([p.tau for p in pops]),
        threshold_mv=cat1("threshold_mv"),
        cue_gain=cat1("cue_gain"),
        noise_scale=cat1("noise_scale"),
        slow_noise_mv=cat1("slow_noise_mv"),
        rebound_mv=cat1("rebound_mv"),
    )
    offsets = (np.arange(k, dtype=np.int32) * n)[:, None]
    bsources = (offsets + graph.sources[None, :]).ravel()
    btargets = np.tile(graph.targets, (k, 1)).copy()
    valid = btargets >= 0
    rep_off = np.repeat(np.arange(k, dtype=np.int32) * n, n)[:, None]
    btargets[valid] += np.broadcast_to(rep_off, btargets.shape)[valid]
    bgraph = type(graph)(bsources, btargets, np.concatenate([p.sign * p.jump for p in pops]),
                         np.tile(graph.delay, (k, 1)), np.tile(graph.lengths, (k, 1)))
    base_w = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
    bew = np.vstack([base_w if w is None else w for w in weights_list])
    return bpop, bgraph, bew


def evaluate_spontaneous(base_pop, graph, sim, protocol, seed):
    sp_spec = protocol["spontaneous"]
    dt = sim["dt_ms"]
    params = {**sim, "baseline_ms": sp_spec["duration_ms"], "cue_ms": 0}
    burn = ticks(sp_spec["burn_in_ms"], dt)
    noise, initial = trial_stimuli(base_pop, params, seed, 4000)
    conds = sp_spec["conditions"]
    pops, weights_list = [], []
    for cond in conds:
        cpop, weights, _ = configure_pd_condition(base_pop, graph, protocol["dopamine_depletion"], cond)
        pops.append(cpop)
        weights_list.append(weights)
    bpop, bgraph, bew = _stack_populations(pops, graph, weights_list)
    n = len(base_pop.cell_type)
    k = len(conds)
    bspikes = simulate(bpop, bgraph, params, np.tile(noise, (1, k)), np.tile(initial, k),
                       np.zeros(k * n), edge_weights=bew)[burn:].reshape(-1, k, n)
    masks = {g: base_pop.cell_type == base_pop.names.index(g) for g in SPONTANEOUS_GROUPS}
    out_metrics, out_psd = {}, {}
    freqs_out = None
    for ci, cond in enumerate(conds):
        spikes = bspikes[:, ci, :]
        out_metrics[cond], out_psd[cond] = {}, {}
        for g, mask in masks.items():
            m, freqs, psd = oscillation_metrics(spikes, mask, dt, sp_spec)
            out_metrics[cond][g] = m
            if g in ("GPe", "STN", "GPi"):
                out_psd[cond][g] = psd
                freqs_out = freqs
    return out_metrics, freqs_out, out_psd


def evaluate_cortical_pd(base_pop, graph, sim, protocol, cortical_protocol, seed):
    stim = cortical_protocol["m1_stimulus"]
    dt = sim["dt_ms"]
    boundary = ticks(stim["stim_at_ms"], dt)
    after = ticks(stim["after_ms"], dt)
    params = {**sim, "baseline_ms": stim["stim_at_ms"], "cue_ms": stim["after_ms"]}
    masks = {g: base_pop.cell_type == base_pop.names.index(g) for g in RECORDED_GROUPS}
    conds = protocol["cortical_evoked"]["conditions"]
    pops, weights_list, drives = [], [], {}
    for cond in conds:
        cpop, weights, gains = configure_pd_condition(base_pop, graph, protocol["dopamine_depletion"], cond)
        pops.extend((cpop, cpop))
        weights_list.extend((weights, weights))
        drives[cond] = cortical_drive_pd(cpop, dt, stim, seed, gains)

    bpop, bgraph, bew = _stack_populations(pops, graph, weights_list)
    n = len(base_pop.cell_type)
    k = len(pops)
    zeros = np.zeros(k * n)
    accum = {cond: {g: np.zeros(after) for g in RECORDED_GROUPS} for cond in conds}
    for trial in range(stim["repetitions"]):
        noise, initial = trial_stimuli(base_pop, params, seed, 3000 + trial)
        nb = np.hstack([arr for cond in conds for arr in (noise, noise + drives[cond])])
        ib = np.tile(initial, k)
        sp = simulate(bpop, bgraph, params, nb, ib, zeros, edge_weights=bew)[boundary:].reshape(after, len(conds), 2, n)
        for ci, cond in enumerate(conds):
            for g, mask in masks.items():
                accum[cond][g] += sp[:, ci, 1, mask].mean(axis=1) - sp[:, ci, 0, mask].mean(axis=1)
    scale = 1000.0 / (dt * stim["repetitions"])
    return {cond: {g: vals * scale for g, vals in groups.items()} for cond, groups in accum.items()}


def summarize_pd_seed(seed, model_name, sp_metrics, cort_responses, dt, protocol, cortical_protocol):
    sp_spec = protocol["spontaneous"]
    win = cortical_protocol["windows"]
    eps = protocol["cortical_evoked"]["sign_tolerance_hz"]
    r_min = sp_spec["rate_shift_min_hz"]
    b_ratio = sp_spec["min_beta_index_ratio"]
    b_min = sp_spec["min_pd_beta_index"]

    norm = sp_metrics["normal_intact"]
    pd = sp_metrics["pd_intact"]
    pd_str = sp_metrics["pd_striatal_only"]
    stn_blk = sp_metrics["pd_stn_blocked"]
    gpe_blk = sp_metrics["pd_gpe_to_stn_blocked"]

    gpi_cort = {c: window_metrics(cort_responses[c]["GPi"], dt, win) for c in cort_responses}
    gpe_cort = {c: window_metrics(cort_responses[c]["GPe"], dt, win) for c in cort_responses}
    stn_cort = {c: window_metrics(cort_responses[c]["STN"], dt, win) for c in cort_responses}

    c1_rate = int((norm["GPe"]["rate_hz"] - pd["GPe"]["rate_hz"] >= r_min)
                  and (pd["STN"]["rate_hz"] - norm["STN"]["rate_hz"] >= r_min))
    c2_beta = int((pd["STN"]["pop_beta_index"] >= b_min
                   and pd["STN"]["pop_beta_index"] >= b_ratio * norm["STN"]["pop_beta_index"])
                  and (pd["GPi"]["pop_beta_index"] >= b_min
                       and pd["GPi"]["pop_beta_index"] >= b_ratio * norm["GPi"]["pop_beta_index"]))
    c3_stn_inact = int((stn_blk["GPi"]["pop_beta_power_hz2"] < pd["GPi"]["pop_beta_power_hz2"])
                       and (stn_blk["GPi"]["pop_beta_index"] < pd["GPi"]["pop_beta_index"])
                       and (pd["GPi"]["rate_hz"] - stn_blk["GPi"]["rate_hz"] >= r_min))
    c4_gpe_blk = int((gpe_blk["STN"]["pop_beta_power_hz2"] < pd["STN"]["pop_beta_power_hz2"])
                     and (gpe_blk["STN"]["pop_beta_index"] < pd["STN"]["pop_beta_index"])
                     and (gpe_blk["STN"]["rate_hz"] - pd["STN"]["rate_hz"] >= r_min))
    c5_cort_shift = int((gpi_cort["pd_intact"]["inhibition_hz"] - gpi_cort["normal_intact"]["inhibition_hz"] > eps)
                        and (gpe_cort["pd_intact"]["late_hz"] - gpe_cort["normal_intact"]["late_hz"] > eps))
    c6_stn_rescue = int((gpi_cort["pd_stn_blocked"]["inhibition_hz"] < -eps)
                        and (gpi_cort["pd_intact"]["inhibition_hz"] - gpi_cort["pd_stn_blocked"]["inhibition_hz"] > eps)
                        and (gpi_cort["pd_intact"]["early_hz"] - gpi_cort["pd_stn_blocked"]["early_hz"] > eps)
                        and (gpi_cort["pd_intact"]["late_hz"] - gpi_cort["pd_stn_blocked"]["late_hz"] > eps))
    checks = (c1_rate, c2_beta, c3_stn_inact, c4_gpe_blk, c5_cort_shift, c6_stn_rescue)

    return {
        "seed": seed,
        "model": model_name,
        "normal_gpe_rate_hz": norm["GPe"]["rate_hz"],
        "pd_gpe_rate_hz": pd["GPe"]["rate_hz"],
        "normal_stn_rate_hz": norm["STN"]["rate_hz"],
        "pd_stn_rate_hz": pd["STN"]["rate_hz"],
        "normal_gpi_rate_hz": norm["GPi"]["rate_hz"],
        "pd_gpi_rate_hz": pd["GPi"]["rate_hz"],
        "pd_minus_normal_gpi_rate_hz": pd["GPi"]["rate_hz"] - norm["GPi"]["rate_hz"],
        "normal_stn_beta_index": norm["STN"]["pop_beta_index"],
        "pd_stn_beta_index": pd["STN"]["pop_beta_index"],
        "pd_striatal_only_stn_beta_index": pd_str["STN"]["pop_beta_index"],
        "normal_gpi_beta_index": norm["GPi"]["pop_beta_index"],
        "pd_gpi_beta_index": pd["GPi"]["pop_beta_index"],
        "pd_striatal_only_gpi_beta_index": pd_str["GPi"]["pop_beta_index"],
        "normal_gpe_beta_index": norm["GPe"]["pop_beta_index"],
        "pd_gpe_beta_index": pd["GPe"]["pop_beta_index"],
        "pd_stn_unit_beta_index": pd["STN"]["unit_beta_index"],
        "pd_gpi_unit_beta_index": pd["GPi"]["unit_beta_index"],
        "pd_stn_peak_freq_hz": pd["STN"]["peak_freq_hz"],
        "pd_gpi_peak_freq_hz": pd["GPi"]["peak_freq_hz"],
        "normal_gpi_beta_power_hz2": norm["GPi"]["pop_beta_power_hz2"],
        "pd_gpi_beta_power_hz2": pd["GPi"]["pop_beta_power_hz2"],
        "pd_stn_blk_gpi_beta_power_hz2": stn_blk["GPi"]["pop_beta_power_hz2"],
        "pd_stn_blk_gpi_beta_index": stn_blk["GPi"]["pop_beta_index"],
        "pd_stn_blk_gpi_rate_hz": stn_blk["GPi"]["rate_hz"],
        "pd_stn_beta_power_hz2": pd["STN"]["pop_beta_power_hz2"],
        "pd_gpe_blk_stn_beta_power_hz2": gpe_blk["STN"]["pop_beta_power_hz2"],
        "pd_gpe_blk_stn_beta_index": gpe_blk["STN"]["pop_beta_index"],
        "pd_gpe_blk_stn_rate_hz": gpe_blk["STN"]["rate_hz"],
        "normal_cort_gpi_inh_hz": gpi_cort["normal_intact"]["inhibition_hz"],
        "pd_cort_gpi_inh_hz": gpi_cort["pd_intact"]["inhibition_hz"],
        "pd_stn_blk_cort_gpi_inh_hz": gpi_cort["pd_stn_blocked"]["inhibition_hz"],
        "normal_cort_gpi_late_hz": gpi_cort["normal_intact"]["late_hz"],
        "pd_cort_gpi_late_hz": gpi_cort["pd_intact"]["late_hz"],
        "pd_stn_blk_cort_gpi_late_hz": gpi_cort["pd_stn_blocked"]["late_hz"],
        "normal_cort_gpe_late_hz": gpe_cort["normal_intact"]["late_hz"],
        "pd_cort_gpe_late_hz": gpe_cort["pd_intact"]["late_hz"],
        "normal_cort_stn_late_hz": stn_cort["normal_intact"]["late_hz"],
        "pd_cort_stn_late_hz": stn_cort["pd_intact"]["late_hz"],
        "pd_rate_shift_pass": c1_rate,
        "pd_beta_emergence_pass": c2_beta,
        "tachibana_stn_inactivation_pass": c3_stn_inact,
        "tachibana_gpe_stn_block_pass": c4_gpe_blk,
        "chiken_pd_cortical_shift_pass": c5_cort_shift,
        "chiken_stn_block_rescue_pass": c6_stn_rescue,
        "pd_checks_passed": int(sum(checks)),
        "all_six_pd_checks": int(all(checks)),
    }


_PD_CTX = {}


def _pd_seed_worker(seed):
    ctx = _PD_CTX
    config = ctx["config"]
    patchseq = ctx["patchseq"]
    protocol = ctx["protocol"]
    cortical_protocol = ctx["cortical_protocol"]
    models = ctx["models"]
    dt = config["simulation"]["dt_ms"]
    sp_conds = protocol["spontaneous"]["conditions"]
    cort_conds = protocol["cortical_evoked"]["conditions"]

    base_pop = patchseq_population(population(config, ctx["atlas"], seed), patchseq)
    s_eval, s_sp, s_cort = [], [], []
    s_psd, s_curves = {}, {}
    freqs_ref = None
    for model_name, spec in models.items():
        pop, cfg, sim = apply_model_spec(base_pop, config, spec)
        graph = connect("D", pop, cfg, seed)
        sp_metrics, freqs, psds = evaluate_spontaneous(pop, graph, sim, protocol, seed)
        freqs_ref = freqs
        for cond in sp_conds:
            for g, vals in sp_metrics[cond].items():
                s_sp.append({"seed": seed, "model": model_name, "condition": cond, "group": g, **vals})
            for g, arr in psds[cond].items():
                s_psd[model_name, cond, g] = arr

        cort_resp = evaluate_cortical_pd(pop, graph, sim, protocol, cortical_protocol, seed)
        for cond in cort_conds:
            for g in ("GPi", "GPe", "STN"):
                wm = window_metrics(cort_resp[cond][g], dt, cortical_protocol["windows"])
                s_cort.append({"seed": seed, "model": model_name, "condition": cond, "group": g, **wm})
            for g, arr in cort_resp[cond].items():
                s_curves[model_name, cond, g] = arr

        s_eval.append(summarize_pd_seed(seed, model_name, sp_metrics, cort_resp, dt, protocol, cortical_protocol))
    print(f"parkinsonian seed {seed}: {len(models)} models evaluated", flush=True)
    return s_eval, s_sp, s_cort, freqs_ref, s_psd, s_curves


def execute(config, patchseq, cortical_protocol, protocol, biophysics_sel, rebound_sel, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    atlas, atlas_info = load_atlas(ROOT / "atlas_data")
    models = {
        "baseline_lif": None,
        "selected_biophysics": biophysics_sel["selected_spec"],
        "selected_rebound": rebound_sel["selected_spec"],
    }
    dt = config["simulation"]["dt_ms"]
    after = ticks(cortical_protocol["m1_stimulus"]["after_ms"], dt)
    cort_conds = protocol["cortical_evoked"]["conditions"]
    eval_rows, sp_rows, cort_rows = [], [], []
    curve_sum = {(m, c, g): np.zeros(after) for m in models for c in cort_conds for g in RECORDED_GROUPS}
    psd_sum = {}
    freqs_ref = None

    _PD_CTX.update({
        "atlas": atlas, "config": config, "patchseq": patchseq,
        "protocol": protocol, "cortical_protocol": cortical_protocol, "models": models,
    })
    workers = min(len(seeds), os.cpu_count() or 1)
    if "fork" in mp.get_all_start_methods() and workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_pd_seed_worker, seeds)
    else:
        results = [_pd_seed_worker(s) for s in seeds]

    for s_eval, s_sp, s_cort, freqs, s_psd, s_curves in results:
        eval_rows.extend(s_eval)
        sp_rows.extend(s_sp)
        cort_rows.extend(s_cort)
        freqs_ref = freqs
        for k, arr in s_psd.items():
            psd_sum[k] = psd_sum.get(k, np.zeros_like(arr)) + arr / len(seeds)
        for k, arr in s_curves.items():
            curve_sum[k] += arr / len(seeds)

    max_freq_plot = 60.0
    freq_idx = np.flatnonzero(freqs_ref <= max_freq_plot)
    spectra_rows = [
        {"model": m, "condition": c, "group": g, "freq_hz": float(freqs_ref[i]), "pop_psd_hz2_per_hz": float(arr[i])}
        for (m, c, g), arr in psd_sum.items() for i in freq_idx
    ]
    curve_rows = [
        {"model": m, "condition": c, "group": g, "time_ms": t * dt, "response_hz": float(val)}
        for (m, c, g), arr in curve_sum.items() for t, val in enumerate(arr)
    ]
    for name, data in (
        ("evaluations", eval_rows),
        ("spontaneous", sp_rows),
        ("cortical_windows", cort_rows),
        ("mean_spectra", spectra_rows),
        ("mean_cortical_curves", curve_rows),
    ):
        write_csv(out / f"{name}.csv", data)

    manifest = {
        "seeds": seeds, "protocol": protocol, "cortical_protocol": cortical_protocol,
        "biophysics_selection": biophysics_sel, "rebound_selection": rebound_sel,
        "model_config": config, "patchseq_summary": patchseq, "atlas": atlas_info,
        "python": platform.python_version(), "numpy": np.__version__,
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.csv"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biophysics", type=Path, default=ROOT / "biophysics/results/selection.json")
    parser.add_argument("--rebound", type=Path, default=ROOT / "rebound/results/selection.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    execute(load(ROOT / "config.json"), load(ROOT / "validation/patchseq_tau.json"),
            load(ROOT / "cortical/protocol.json"), protocol,
            load(args.biophysics), load(args.rebound), args.out, args.seeds or protocol["seeds"])
