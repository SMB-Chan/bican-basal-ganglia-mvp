"""Test whether pacemaker-like LIF dynamics create the GPi inhibition that survives STN→GPe blockade.

Three parts:
1. Isolated cells. For GPe, STN and GPi, scale the membrane noise and re-solve the bias so the isolated
   firing rate stays at its default value. Measure regularity (ISI CV) and the response to one brief
   excitatory jump with no synapses present.
2. Network sweep. Apply each noise regime to GPe, STN and GPi of the projection-constrained graph (E)
   and the Patch-seq variant (F). A pre-declared rule picks one regime from tonic rate and CV only.
3. Stage-1 ensemble. Re-evaluate the 192 identifiability graphs under the selected regime.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from identifiability import run as identifiability
from model import Graph, Population, connect, population, simulate, ticks
from run import trial_stimuli, write_csv
from validation.run_validation import measure_pulse_groups, measure_tonic, patchseq_population, pulse_metrics


HERE = Path(__file__).resolve().parent


def isolated_population(config, group, n, bias, multiplier):
    spec = config["groups"][group]
    ones = np.ones(n)
    return Population((group,), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros((n, 3)),
                      spec["sign"] * ones, spec["jump_mv"] * ones, bias * ones, spec["tau_ms"] * ones,
                      noise_scale=multiplier * ones)


def empty_graph(n):
    return Graph(np.array([], dtype=int), np.full((n, 1), -1), np.zeros(n), np.zeros((n, 1), dtype=int),
                 np.empty((0, 1)))


def isolated_spikes(config, group, bias, multiplier, spec, jump_mv=0.0):
    """Synapse-free cells sharing one noise realization across bias values, multipliers and probe/sham."""
    sim = config["simulation"]
    n = spec["cells"]
    params = {**sim, "baseline_ms": spec["duration_ms"], "cue_ms": 0}
    pop = isolated_population(config, group, n, bias, multiplier)
    noise, initial = trial_stimuli(pop, params, spec["seed"], 100 + list(config["groups"]).index(group))
    if jump_mv:
        noise[ticks(spec["probe_at_ms"], sim["dt_ms"])] += jump_mv
    return simulate(pop, empty_graph(n), params, noise, initial, np.zeros(n))


def rate_and_cv(spikes, dt, burn_ms, min_intervals):
    data = spikes[ticks(burn_ms, dt):]
    rate = float(data.mean() * 1000 / dt)
    cvs = []
    for column in data.T:
        intervals = np.diff(np.flatnonzero(column))
        if len(intervals) >= min_intervals:
            cvs.append(intervals.std() / intervals.mean())
    return rate, (float(np.mean(cvs)) if cvs else float("nan"))


def isolated_rate(config, group, bias, multiplier, spec):
    # Exclude the probe region so calibration does not depend on it.
    spikes = isolated_spikes(config, group, bias, multiplier, spec)[:ticks(spec["probe_at_ms"], config["simulation"]["dt_ms"])]
    return rate_and_cv(spikes, config["simulation"]["dt_ms"], spec["burn_in_ms"], spec["min_intervals"])


def calibrate_bias(config, group, multiplier, target_hz, spec):
    """Bisection on bias. Rate is non-decreasing in bias under shared noise."""
    low, high = spec["bias_search_mv"]
    if not isolated_rate(config, group, low, multiplier, spec)[0] <= target_hz <= isolated_rate(config, group, high, multiplier, spec)[0]:
        raise ValueError(f"{group}: target {target_hz} Hz lies outside the bias search range")
    for _ in range(spec["bias_iterations"]):
        middle = (low + high) / 2
        if isolated_rate(config, group, middle, multiplier, spec)[0] < target_hz:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def probe_response(config, group, bias, multiplier, spec):
    dt = config["simulation"]["dt_ms"]
    start = ticks(spec["probe_at_ms"], dt)
    window = ticks(spec["probe_window_ms"], dt)
    evoked = isolated_spikes(config, group, bias, multiplier, spec, spec["probe_jump_mv"])
    sham = isolated_spikes(config, group, bias, multiplier, spec)
    return (evoked[start:start + window].mean(axis=1) - sham[start:start + window].mean(axis=1)) * 1000 / dt


def window_mean(curve, window_ms, dt):
    return float(curve[slice(*(ticks(t, dt) for t in window_ms))].mean())


def binned_min(curve, window_ms, width_ms, dt):
    start, stop = (ticks(t, dt) for t in window_ms)
    width = ticks(width_ms, dt)
    if (stop - start) % width:
        raise ValueError("The trough window must hold whole bins")
    return float(curve[start:stop].reshape(-1, width).mean(axis=1).min())


def isolated_experiment(config, protocol):
    spec = protocol["isolated"]
    dt = config["simulation"]["dt_ms"]
    regimes, rows, curves = {}, [], []
    for group in protocol["groups"]:
        default_bias = config["groups"][group]["bias_mv"]
        target, _ = isolated_rate(config, group, default_bias, 1.0, spec)
        for multiplier in protocol["noise_multipliers"]:
            bias = default_bias if multiplier == 1.0 else calibrate_bias(config, group, multiplier, target, spec)
            rate, cv = isolated_rate(config, group, bias, multiplier, spec)
            curve = probe_response(config, group, bias, multiplier, spec)
            regimes.setdefault(multiplier, {})[group] = {"bias_mv": bias, "noise_multiplier": multiplier}
            rows.append({"group": group, "noise_multiplier": multiplier, "bias_mv": bias,
                         "equilibrium_minus_threshold_mv": config["simulation"]["rest_mv"] + bias - config["simulation"]["threshold_mv"],
                         "target_rate_hz": target, "rate_hz": rate, "isi_cv": cv,
                         "immediate_hz": window_mean(curve, spec["immediate_window_ms"], dt),
                         "post_window_hz": window_mean(curve, spec["post_window_ms"], dt),
                         "trough_5ms_hz": binned_min(curve, spec["trough_window_ms"], 5, dt)})
            curves.extend({"group": group, "noise_multiplier": multiplier, "time_ms": t * dt, "response_hz": value}
                          for t, value in enumerate(curve))
        print(f"isolated {group}: {len(protocol['noise_multipliers'])} regimes", flush=True)
    return regimes, rows, curves


def regime_population(pop, regime):
    bias = pop.bias.astype(float, copy=True)
    scale = np.ones(len(bias)) if pop.noise_scale is None else pop.noise_scale.astype(float, copy=True)
    for group, values in regime.items():
        use = pop.cell_type == pop.names.index(group)
        bias[use] = values["bias_mv"]
        scale[use] = values["noise_multiplier"]
    return replace(pop, bias=bias, noise_scale=scale)


def network_sweep(config, validation, patchseq, protocol, regimes, atlas):
    sim = config["simulation"]
    tonic = validation["tonic"]
    pulse = validation["stn_pulse"]
    references = {key: {name: np.mean(values) for name, values in tonic[key].items()} for key in ("rate_hz", "isi_cv")}
    late = slice(*(ticks(t, sim["dt_ms"]) for t in pulse["late_window_ms"]))
    everyone = "+".join(protocol["groups"])
    variants = [(multiplier, everyone, regime) for multiplier, regime in regimes.items()]
    split = protocol["network"]["decomposition"]
    variants += [(split["noise_multiplier"], group, {group: regimes[split["noise_multiplier"]][group]})
                 for group in split["single_groups"]]
    rows, firing, curves = [], [], []
    for seed in protocol["network"]["seeds"]:
        base = population(config, atlas, seed)
        graph = connect("D", base, {**config, "matched_channel_gain": 1.0}, seed)
        for multiplier, changed, regime in variants:
            dynamic = regime_population(base, regime)
            for membrane in protocol["network"]["membrane_conditions"]:
                pop = dynamic if membrane == "E" else patchseq_population(dynamic, patchseq)
                keys = {"seed": seed, "noise_multiplier": multiplier, "changed_groups": changed, "membrane": membrane}
                measured = measure_tonic(pop, graph, sim, tonic, seed)
                firing.extend({**keys, "group": group, **values} for group, values in measured.items())
                errors = {key: float(np.mean([abs(values[key] - references[key][group]) for group, values in measured.items()]))
                          for key in references}
                responses = measure_pulse_groups(pop, graph, sim, pulse, seed, protocol["network"]["pulse_groups"])
                stn = responses["STN"]
                rows.append({**keys, "tonic_rate_mae_hz": errors["rate_hz"], "tonic_cv_mae": errors["isi_cv"],
                             **{f"{group}_{key}": measured[group][key] for group in measured for key in ("rate_hz", "isi_cv")},
                             **pulse_metrics(responses["GPi"], sim["dt_ms"], pulse),
                             "stn_late_hz": float(stn[0, late].mean()), "stn_blocked_late_hz": float(stn[1, late].mean())})
                for group, values in responses.items():
                    curves.extend({**keys, "group": group, "time_ms": t * sim["dt_ms"], "intact_hz": a, "blocked_hz": b}
                                  for t, (a, b) in enumerate(values.T))
        print(f"network seed {seed}: {len(variants)} regimes", flush=True)
    return rows, firing, curves


def select_regime(rows, selection, everyone):
    """Pick a noise multiplier from tonic summaries only. Other columns are ignored on purpose."""
    by_multiplier = {}
    for row in rows:
        if row["membrane"] != selection["membrane"] or row["changed_groups"] != everyone:
            continue
        by_multiplier.setdefault(float(row["noise_multiplier"]), []).append(
            (float(row["tonic_rate_mae_hz"]), float(row[selection["minimize"]])))
    candidates = []
    for multiplier, values in sorted(by_multiplier.items()):
        rate_mae, objective = np.mean(values, axis=0)
        if rate_mae <= selection["max_tonic_rate_mae_hz"]:
            candidates.append((objective, multiplier, rate_mae))
    if not candidates:
        raise ValueError("No regime satisfies the tonic rate constraint")
    objective, multiplier, rate_mae = min(candidates)
    return {"noise_multiplier": multiplier, "mean_tonic_rate_mae_hz": rate_mae, f"mean_{selection['minimize']}": objective,
            "eligible_multipliers": [c[1] for c in candidates], "grid_edge": multiplier == max(by_multiplier)}


def execute(config, validation, patchseq, ident_protocol, protocol, out, skip_ensemble=False):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True)
    atlas, info = load_atlas(ROOT / "atlas_data")
    regimes, isolated, isolated_curves = isolated_experiment(config, protocol)
    network, firing, curves = network_sweep(config, validation, patchseq, protocol, regimes, atlas)
    selected = select_regime(network, protocol["selection"], "+".join(protocol["groups"]))
    regime = regimes[selected["noise_multiplier"]]
    for name, rows in (("isolated", isolated), ("isolated_curves", isolated_curves), ("network", network),
                       ("network_tonic", firing), ("network_pulse_curves", curves)):
        write_csv(out / f"{name}.csv", rows)
    selection = {**selected, "regime": regime, "rule": protocol["selection"]}
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected noise multiplier {selected['noise_multiplier']}", flush=True)
    if not skip_ensemble:
        identifiability.execute(config, validation, patchseq, ident_protocol, out / "ensemble", ident_protocol["placements"],
                                transform=lambda pop: regime_population(pop, regime), extra={"dynamics_regime": selection})
    manifest = {"protocol": protocol, "identifiability_protocol": ident_protocol, "model_config": config,
                "validation": validation, "patchseq_summary": patchseq, "atlas": info,
                "python": platform.python_version(), "numpy": np.__version__,
                "raw_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(out.iterdir()) if path.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="Network seeds (default: protocol)")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip the 384-evaluation stage-1 re-run")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    if args.seeds:
        protocol["network"]["seeds"] = args.seeds
    execute(load(ROOT / "config.json"), load(ROOT / "validation/protocol.json"), load(ROOT / "validation/patchseq_tau.json"),
            load(ROOT / "identifiability/protocol.json"), protocol, args.out, args.skip_ensemble)
