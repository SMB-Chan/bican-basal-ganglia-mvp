"""Factorial test of two-timescale fluctuations, synaptic decay, and indirect-pathway coupling."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from model import Graph, Population, connect, population, simulate, ticks
from run import trial_stimuli, write_csv
from validation.run_validation import measure_pulse_groups, measure_tonic, patchseq_population, pulse_metrics


HERE = Path(__file__).resolve().parent


def isolated_grid(config, group, biases, fast_scale, slow_sd, slow_tau, cal_spec, seed):
    spec = config["groups"][group]
    m = cal_spec["isolated_cells_per_bias"]
    dur = cal_spec["isolated_duration_ms"]
    burn = ticks(cal_spec["isolated_burn_in_ms"], config["simulation"]["dt_ms"])
    b_arr = np.repeat(biases, m)
    n = len(b_arr)
    ones = np.ones(n)
    pop = Population((group,), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros((n, 3)),
                     spec["sign"] * ones, spec["jump_mv"] * ones, b_arr, spec["tau_ms"] * ones,
                     noise_scale=fast_scale * ones, slow_noise_mv=slow_sd * ones if slow_sd > 0 else None)
    base_ones = np.ones(m)
    base_pop = Population((group,), np.zeros(m, dtype=int), np.zeros(m, dtype=int), np.zeros((m, 3)),
                          spec["sign"] * base_ones, spec["jump_mv"] * base_ones, np.zeros(m), spec["tau_ms"] * base_ones,
                          noise_scale=fast_scale * base_ones, slow_noise_mv=slow_sd * base_ones if slow_sd > 0 else None)
    params = {**config["simulation"], "baseline_ms": dur, "cue_ms": 0, "slow_noise_tau_ms": slow_tau}
    noise_m, init_m = trial_stimuli(base_pop, params, seed, 100 + list(config["groups"]).index(group))
    graph = Graph(np.array([], dtype=int), np.full((n, 1), -1), np.zeros(n), np.zeros((n, 1), dtype=int), np.empty((0, 1)))
    spikes = simulate(pop, graph, params, np.tile(noise_m, (1, len(biases))), np.tile(init_m, len(biases)), np.zeros(n))[burn:]
    spikes = spikes.reshape(len(spikes), len(biases), m)
    rates = spikes.mean(axis=(0, 2)) * (1000.0 / config["simulation"]["dt_ms"])
    cvs = []
    for bi in range(len(biases)):
        vals = [np.diff(np.flatnonzero(spikes[:, bi, j])).std() / np.diff(np.flatnonzero(spikes[:, bi, j])).mean()
                for j in range(m) if spikes[:, bi, j].sum() >= 6]
        cvs.append(float(np.mean(vals)) if vals else float("nan"))
    return rates, np.array(cvs)


def calibrate_fluctuations(config, validation, protocol):
    seed = protocol["calibration_seed"]
    cal = protocol["tonic_calibration"]
    slow_tau = protocol["slow_noise_tau_ms"]
    refs_cv = {g: float(np.mean(v)) for g, v in validation["tonic"]["isi_cv"].items()}
    biases_grid = np.linspace(-20.0, 28.0, 17)
    regimes = {}
    for name, spec in protocol["fluctuation_regimes"].items():
        mult = float(spec["noise_multiplier"])
        mode = spec["slow_noise_mode"]
        regimes[name] = {}
        for g in protocol["groups"]:
            if mode == "zero":
                slow_sd = 0.0
            elif mode == "fixed":
                slow_sd = float(spec["slow_noise_mv"])
            elif mode == "match_reference_cv":
                r0, _ = isolated_grid(config, g, np.array([config["groups"][g]["bias_mv"]]), 1.0, 0.0, slow_tau, cal, seed)
                target_r = float(r0[0])
                lo, hi = 0.0, float(spec["max_slow_noise_mv"])
                for _ in range(cal["cv_bisection_steps"]):
                    mid = 0.5 * (lo + hi)
                    rs, _ = isolated_grid(config, g, biases_grid, mult, mid, slow_tau, cal, seed)
                    b_est = float(np.interp(target_r, rs, biases_grid))
                    _, cv_est = isolated_grid(config, g, np.array([b_est]), mult, mid, slow_tau, cal, seed)
                    if cv_est[0] < refs_cv[g]:
                        lo = mid
                    else:
                        hi = mid
                slow_sd = 0.5 * (lo + hi)
            else:
                raise ValueError(f"Unknown slow_noise_mode {mode}")
            regimes[name][g] = {"noise_multiplier": mult, "slow_noise_mv": slow_sd}
    return regimes


def configure_variant(pop, config, pathway_spec, syn_spec, fluct_map, slow_tau):
    cfg = {**config, "matched_channel_gain": 1.0}
    if pathway_spec["projection_propensity"]:
        cfg["projection_propensity"] = {**config["projection_propensity"], **pathway_spec["projection_propensity"]}
    jump = pop.jump.astype(float, copy=True)
    jump[pop.cell_type == pop.names.index("GPe")] = float(pathway_spec["gpe_jump_mv"])
    noise_scale = np.ones(len(jump))
    slow_noise = np.zeros(len(jump))
    has_slow = False
    for g, vals in fluct_map.items():
        mask = pop.cell_type == pop.names.index(g)
        noise_scale[mask] = vals["noise_multiplier"]
        if vals["slow_noise_mv"] > 0:
            slow_noise[mask] = vals["slow_noise_mv"]
            has_slow = True
    updated = replace(pop, jump=jump, noise_scale=noise_scale, slow_noise_mv=slow_noise if has_slow else None)
    sim = {**config["simulation"], "slow_noise_tau_ms": slow_tau, **syn_spec}
    return updated, cfg, sim


def match_network_tonic(pop, graph, sim, tonic_spec, groups, target_rates, cal_spec, seed):
    bias = pop.bias.astype(float, copy=True)
    for it in range(cal_spec["network_iterations"]):
        cur = replace(pop, bias=bias)
        measured = measure_tonic(cur, graph, sim, tonic_spec, seed)
        errs = {g: target_rates[g] - measured[g]["rate_hz"] for g in groups}
        if max(abs(e) for e in errs.values()) <= cal_spec["tolerance_hz"]:
            break
        step = 0.35 if it < 6 else 0.25
        for g, err in errs.items():
            bias[pop.cell_type == pop.names.index(g)] += step * err
    return {g: float(bias[pop.cell_type == pop.names.index(g)][0]) for g in groups}


def with_biases(pop, group_biases):
    bias = pop.bias.astype(float, copy=True)
    for g, val in group_biases.items():
        bias[pop.cell_type == pop.names.index(g)] = val
    return replace(pop, bias=bias)


def stn_immediate_hz(pop, graph, sim, pulse_spec, seed, jump_mv):
    spec = {**pulse_spec, "voltage_jump_mv": jump_mv}
    stn_curve = measure_pulse_groups(pop, graph, sim, spec, seed, ("STN",))["STN"]
    steps = ticks(2.0, sim["dt_ms"])
    return float(stn_curve[0, :steps].mean())


def calibrate_stn_jump(pop, graph, sim, pulse_spec, target_hz, decomp_spec, seed):
    lo, hi = decomp_spec["jump_search_mv"]
    for _ in range(decomp_spec["jump_iterations"]):
        mid = 0.5 * (lo + hi)
        if stn_immediate_hz(pop, graph, sim, pulse_spec, seed, mid) < target_hz:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def select_condition(rows, rule):
    factorial = [r for r in rows if r["kind"] == "factorial"]
    keys = sorted({(r["pathway"], r["synapse"], r["fluctuation"]) for r in factorial})
    candidates = []
    for pathway, synapse, fluctuation in keys:
        sub = [r for r in factorial if (r["pathway"], r["synapse"], r["fluctuation"]) == (pathway, synapse, fluctuation)]
        strict_frac = float(np.mean([int(r["strict_candidate"]) for r in sub]))
        coarse_frac = float(np.mean([int(r["coarse_candidate"]) for r in sub]))
        rmae = float(np.mean([float(r["tonic_rate_mae_hz"]) for r in sub]))
        cvmae = float(np.mean([float(r["tonic_cv_mae"]) for r in sub]))
        tie_score = -(cvmae + 0.1 * rmae)
        candidates.append((strict_frac, tie_score, pathway, synapse, fluctuation, coarse_frac, rmae, cvmae))
    best = max(candidates)
    _, _, pathway, synapse, fluctuation, coarse_frac, rmae, cvmae = best
    return {
        "pathway": pathway, "synapse": synapse, "fluctuation": fluctuation,
        "strict_candidate_fraction": best[0], "coarse_candidate_fraction": coarse_frac,
        "mean_tonic_rate_mae_hz": rmae, "mean_tonic_cv_mae": cvmae,
    }


def execute(config, validation, patchseq, ident_protocol, protocol, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    atlas, atlas_info = load_atlas(ROOT / "atlas_data")
    cal_seed = protocol["calibration_seed"]
    if cal_seed in seeds:
        raise ValueError("Calibration seed must be disjoint from evaluation seeds")

    fluctuations = calibrate_fluctuations(config, validation, protocol)
    cal_base = patchseq_population(population(config, atlas, cal_seed), patchseq)
    cal_graph0 = connect("D", cal_base, {**config, "matched_channel_gain": 1.0}, cal_seed)
    tonic_spec = validation["tonic"]
    pulse_spec = validation["stn_pulse"]
    target_tonic = {g: v["rate_hz"] for g, v in measure_tonic(cal_base, cal_graph0, config["simulation"], tonic_spec, cal_seed).items()}
    print("calibrated fluctuations and baseline tonic target", flush=True)

    variants = []
    for pathway_name, pathway_spec in protocol["pathway_modes"].items():
        for syn_name, syn_spec in protocol["synapse_modes"].items():
            for fl_name in protocol["fluctuation_regimes"]:
                variants.append(("factorial", f"{pathway_name}|{syn_name}|{fl_name}",
                                 pathway_name, syn_name, fl_name, fluctuations[fl_name], pulse_spec["voltage_jump_mv"]))

    decomp_regime = protocol["recruitment_decomposition"]["regime"]
    for single_group in ("STN", "GPi"):
        fmap = {g: (fluctuations[decomp_regime][g] if g == single_group else fluctuations["pacemaker"][g])
                for g in protocol["groups"]}
        tag = f"{single_group.lower()}_only_unmatched" if single_group == "STN" else "gpi_only"
        variants.append(("decomposition", tag, "baseline", "delta", tag, fmap, pulse_spec["voltage_jump_mv"]))

    calibrated_specs = {}
    for kind, label, p_name, s_name, f_name, fmap, jump_mv in variants:
        p_spec = protocol["pathway_modes"][p_name]
        s_spec = protocol["synapse_modes"][s_name]
        cpop, ccfg, csim = configure_variant(cal_base, config, p_spec, s_spec, fmap, protocol["slow_noise_tau_ms"])
        cgraph = connect("D", cpop, ccfg, cal_seed)
        if p_name == "baseline" and s_name == "delta" and f_name == "pacemaker":
            biases = {g: float(cal_base.bias[cal_base.cell_type == cal_base.names.index(g)][0]) for g in protocol["groups"]}
        else:
            biases = match_network_tonic(cpop, cgraph, csim, tonic_spec, protocol["groups"],
                                         target_tonic, protocol["tonic_calibration"], cal_seed)
        calibrated_specs[label] = {"kind": kind, "pathway": p_name, "synapse": s_name, "fluctuation": f_name,
                                   "fluct_map": fmap, "biases": biases, "jump_mv": jump_mv}

    # Calibrate evoked-matched STN pulse jump on calibration seed 800
    base_cpop, base_ccfg, base_csim = configure_variant(
        cal_base, config, protocol["pathway_modes"]["baseline"], protocol["synapse_modes"]["delta"],
        fluctuations["pacemaker"], protocol["slow_noise_tau_ms"])
    target_stn_imm = stn_immediate_hz(base_cpop, cal_graph0, base_csim, pulse_spec, cal_seed, pulse_spec["voltage_jump_mv"])
    stn_unmatched = calibrated_specs["stn_only_unmatched"]
    s_cpop, _, s_csim = configure_variant(
        cal_base, config, protocol["pathway_modes"]["baseline"], protocol["synapse_modes"]["delta"],
        stn_unmatched["fluct_map"], protocol["slow_noise_tau_ms"])
    s_cpop = with_biases(s_cpop, stn_unmatched["biases"])
    matched_jump = calibrate_stn_jump(s_cpop, cal_graph0, s_csim, pulse_spec, target_stn_imm,
                                      protocol["recruitment_decomposition"], cal_seed)
    calibrated_specs["stn_only_matched"] = {**stn_unmatched, "fluctuation": "stn_only_matched", "jump_mv": matched_jump}
    print(f"calibrated all {len(calibrated_specs)} variants (matched STN jump = {matched_jump:.2f} mV)", flush=True)

    refs = {k: {g: float(np.mean(v)) for g, v in tonic_spec[k].items()} for k in ("rate_hz", "isi_cv")}
    dt = config["simulation"]["dt_ms"]
    after = ticks(pulse_spec["after_ms"], dt)
    imm_steps = ticks(2.0, dt)
    late_slice = slice(*(ticks(t, dt) for t in pulse_spec["late_window_ms"]))
    rows, tonic_rows = [], []
    curve_sum = {(label, g): np.zeros((2, after)) for label in calibrated_specs for g in ("GPi", "STN")}

    for seed in seeds:
        seed_base = patchseq_population(population(config, atlas, seed), patchseq)
        graphs_by_pathway = {}
        for p_name, p_spec in protocol["pathway_modes"].items():
            tmp_pop, pcfg, _ = configure_variant(seed_base, config, p_spec, protocol["synapse_modes"]["delta"],
                                                 fluctuations["pacemaker"], protocol["slow_noise_tau_ms"])
            graphs_by_pathway[p_name] = connect("D", tmp_pop, pcfg, seed)

        for label, cspec in calibrated_specs.items():
            p_spec = protocol["pathway_modes"][cspec["pathway"]]
            s_spec = protocol["synapse_modes"][cspec["synapse"]]
            pop, _, sim = configure_variant(seed_base, config, p_spec, s_spec, cspec["fluct_map"], protocol["slow_noise_tau_ms"])
            pop = with_biases(pop, cspec["biases"])
            graph = graphs_by_pathway[cspec["pathway"]]
            measured = measure_tonic(pop, graph, sim, tonic_spec, seed)
            keys = {"seed": seed, "variant": label, "kind": cspec["kind"],
                    "pathway": cspec["pathway"], "synapse": cspec["synapse"], "fluctuation": cspec["fluctuation"]}
            tonic_rows.extend({**keys, "group": g, **vals} for g, vals in measured.items())
            rmae = float(np.mean([abs(measured[g]["rate_hz"] - refs["rate_hz"][g]) for g in refs["rate_hz"]]))
            cvmae = float(np.mean([abs(measured[g]["isi_cv"] - refs["isi_cv"][g]) for g in refs["isi_cv"]]))
            pspec = {**pulse_spec, "voltage_jump_mv": cspec["jump_mv"]}
            responses = measure_pulse_groups(pop, graph, sim, pspec, seed, ("GPi", "STN"))
            pm = pulse_metrics(responses["GPi"], dt, pspec)
            coarse = int(rmae <= protocol["selection"]["max_tonic_rate_mae_hz"] and pm["pulse_features"] == 4)
            strict = int(coarse and abs(pm["blocked_late_hz"]) <= protocol["selection"]["max_blocked_late_abs_hz"])
            stn = responses["STN"]
            rows.append({
                **keys, "pulse_jump_mv": cspec["jump_mv"],
                "tonic_rate_mae_hz": rmae, "tonic_cv_mae": cvmae,
                **{f"{g}_{k}": measured[g][k] for g in measured for k in ("rate_hz", "isi_cv")},
                **pm, "stn_immediate_hz": float(stn[0, :imm_steps].mean()),
                "stn_blocked_immediate_hz": float(stn[1, :imm_steps].mean()),
                "stn_late_hz": float(stn[0, late_slice].mean()),
                "stn_blocked_late_hz": float(stn[1, late_slice].mean()),
                "coarse_candidate": coarse, "strict_candidate": strict,
            })
            for g in ("GPi", "STN"):
                curve_sum[label, g] += responses[g] / len(seeds)
        print(f"biophysics seed {seed}: {len(calibrated_specs)} variants", flush=True)

    curve_rows = [
        {"variant": label, "group": g, "time_ms": t * dt, "intact_hz": float(a), "blocked_hz": float(b)}
        for (label, g), arr in curve_sum.items() for t, (a, b) in enumerate(arr.T)
    ]
    for name, data in (("network", rows), ("tonic", tonic_rows), ("mean_pulse_curves", curve_rows)):
        write_csv(out / f"{name}.csv", data)

    chosen = select_condition(rows, protocol["selection"])
    chosen_label = f"{chosen['pathway']}|{chosen['synapse']}|{chosen['fluctuation']}"
    cspec = calibrated_specs[chosen_label]
    p_spec = protocol["pathway_modes"][chosen["pathway"]]
    s_spec = protocol["synapse_modes"][chosen["synapse"]]
    selected_spec = {
        "projection_propensity": p_spec["projection_propensity"],
        "groups": {
            g: {
                "bias_mv": cspec["biases"][g],
                "noise_multiplier": cspec["fluct_map"][g]["noise_multiplier"],
                "slow_noise_mv": cspec["fluct_map"][g]["slow_noise_mv"],
                **({"jump_mv": p_spec["gpe_jump_mv"]} if g == "GPe" else {}),
            }
            for g in protocol["groups"]
        },
        "simulation": {"slow_noise_tau_ms": protocol["slow_noise_tau_ms"], **s_spec},
    }
    selection = {**chosen, "variant": chosen_label, "selected_spec": selected_spec, "rule": protocol["selection"]}
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps({
        "calibration_seed": cal_seed, "target_tonic_rates_hz": target_tonic,
        "target_stn_immediate_hz": target_stn_imm, "matched_stn_jump_mv": matched_jump,
        "variants": calibrated_specs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "seeds": seeds, "protocol": protocol, "identifiability_protocol": ident_protocol,
        "model_config": config, "validation": validation, "patchseq_summary": patchseq, "atlas": atlas_info,
        "python": platform.python_version(), "numpy": np.__version__,
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected biophysical variant: {chosen_label} (strict={chosen['strict_candidate_fraction']:.2f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    execute(load(ROOT / "config.json"), load(ROOT / "validation/protocol.json"),
            load(ROOT / "validation/patchseq_tau.json"), load(ROOT / "identifiability/protocol.json"),
            protocol, args.out, args.seeds or protocol["seeds"])
