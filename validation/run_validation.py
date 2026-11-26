"""Run the paired constraint ladder and evaluate external macaque benchmarks."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from model import connect, graph_statistics, population, simulate, ticks
from run import cue_input, score_trial, trial_plan, trial_stimuli, write_csv


HERE = Path(__file__).resolve().parent


def mask_only(propensity):
    return {source: {target: float(odds > 0) for target, odds in targets.items()}
            for source, targets in propensity.items()}


def condition_config(config, specification):
    return {
        **config,
        "projection_propensity": config["projection_propensity"] if specification["odds"] else mask_only(config["projection_propensity"]),
        "matched_channel_gain": config["matched_channel_gain"] if specification["channel"] else 1.0,
    }


def gaussian_population(pop, config, moments, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    regions = np.array([config["groups"][name]["region"] for name in pop.names])[pop.cell_type]
    xyz = np.empty_like(pop.xyz)
    for region, (center, cov) in moments.items():
        mask = regions == region
        xyz[mask] = rng.multivariate_normal(center, cov, size=int(mask.sum()))
    return replace(pop, xyz=xyz)


def patchseq_population(pop, patchseq):
    tau = pop.tau.astype(float, copy=True)
    for name, group in patchseq["groups"].items():
        tau[pop.cell_type == pop.names.index(name)] = group["median_tau_ms"]
    return replace(pop, tau=tau)


def measure_tonic(pop, graph, sim, spec, seed):
    params = {**sim, "baseline_ms": spec["duration_ms"], "cue_ms": 0}
    burn = ticks(spec["burn_in_ms"], sim["dt_ms"])
    noise, initial = trial_stimuli(pop, params, seed, 1000)
    spikes = simulate(pop, graph, params, noise, initial, np.zeros(len(pop.cell_type)))[burn:]
    duration_s = len(spikes) * sim["dt_ms"] / 1000
    result = {}
    for name in spec["rate_hz"]:
        membership = np.flatnonzero(pop.cell_type == pop.names.index(name))
        intervals = [np.diff(np.flatnonzero(spikes[:, i])) for i in membership]
        valid = [series.std() / series.mean() for series in intervals if len(series) >= spec["min_intervals"]]
        result[name] = {"rate_hz": float(spikes[:, membership].mean() * len(spikes) / duration_s),
                        "isi_cv": float(np.mean(valid)), "cv_valid_fraction": len(valid) / len(membership)}
    return result


def block_stn_to_gpe(pop, graph):
    stn = pop.cell_type == pop.names.index("STN")
    gpe = pop.cell_type == pop.names.index("GPe")
    weights = np.broadcast_to(graph.weights[:, None], graph.targets.shape).copy()
    blocked = stn[:, None] & (graph.targets >= 0) & gpe[graph.targets]
    weights[blocked] = 0
    return weights


def measure_pulse(pop, graph, sim, spec, seed):
    params = {**sim, "baseline_ms": spec["pulse_at_ms"], "cue_ms": spec["after_ms"]}
    boundary = ticks(spec["pulse_at_ms"], sim["dt_ms"])
    n_steps = ticks(spec["after_ms"], sim["dt_ms"])
    stn = np.flatnonzero(pop.cell_type == pop.names.index("STN"))
    gpi = pop.cell_type == pop.names.index("GPi")
    blocked = block_stn_to_gpe(pop, graph)
    n_stimulated = round(len(stn) * spec["fraction_stn"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    driven = rng.choice(stn, n_stimulated, replace=False)
    responses = np.zeros((2, n_steps))
    zeros = np.zeros(len(pop.cell_type))
    for trial in range(spec["repetitions"]):
        noise, initial = trial_stimuli(pop, params, seed, 2000 + trial)
        pulse = noise.copy()
        pulse[boundary, driven] += spec["voltage_jump_mv"]
        for i, weights in enumerate((None, blocked)):
            sham = simulate(pop, graph, params, noise, initial, zeros, edge_weights=weights)
            evoked = simulate(pop, graph, params, pulse, initial, zeros, edge_weights=weights)
            responses[i] += evoked[boundary:, gpi].mean(axis=1) - sham[boundary:, gpi].mean(axis=1)
    return responses * (1000 / (sim["dt_ms"] * spec["repetitions"]))


def pulse_metrics(responses, dt, spec):
    early = slice(*(ticks(t, dt) for t in spec["early_window_ms"]))
    late = slice(*(ticks(t, dt) for t in spec["late_window_ms"]))
    early_hz, blocked_early_hz = responses[:, early].mean(axis=1).tolist()
    late_hz, blocked_late_hz = responses[:, late].mean(axis=1).tolist()
    attenuation = blocked_late_hz - late_hz
    epsilon = spec["sign_tolerance_hz"]
    signs = {"early_excitation": early_hz > epsilon, "late_inhibition": late_hz < -epsilon,
             "blocked_early_excitation": blocked_early_hz > epsilon, "late_attenuation": attenuation > epsilon}
    return {"pulse_early_hz": early_hz, "pulse_late_hz": late_hz, "blocked_early_hz": blocked_early_hz,
            "blocked_late_hz": blocked_late_hz, "late_attenuation_hz": attenuation,
            "pulse_features": sum(signs.values()), **{name: int(passed) for name, passed in signs.items()}}


def measure_choices(pop, graph, sim, seed):
    gpi = pop.cell_type == pop.names.index("GPi")
    channel_masks = [gpi & (pop.channel == channel) for channel in (0, 1)]
    cue_span = ticks(sim["cue_readout_ms"], sim["dt_ms"])
    by_delta = {delta: [] for delta in sim["cue_deltas_mv"]}
    for trial, (delta, side) in enumerate(trial_plan(sim, seed)):
        if side is None:
            continue
        noise, initial = trial_stimuli(pop, sim, seed, trial)
        spikes = simulate(pop, graph, sim, noise, initial, cue_input(pop, sim, delta, side))
        gpi_rates = [float(spikes[-cue_span:, mask].mean()) * 1000 / sim["dt_ms"] for mask in channel_masks]
        _, accuracy, margin = score_trial(gpi_rates, side)
        by_delta[delta].append((accuracy, margin))
    low, high = min(by_delta), max(by_delta)
    return {"choice_weak_accuracy": float(np.mean(by_delta[low], axis=0)[0]),
            "choice_strong_accuracy": float(np.mean(by_delta[high], axis=0)[0]),
            "choice_weak_margin_hz": float(np.mean(by_delta[low], axis=0)[1]),
            "choice_strong_margin_hz": float(np.mean(by_delta[high], axis=0)[1])}


def execute(config, protocol, patchseq, data, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    atlas, atlas_info = load_atlas(data)
    moments = {name: (points.mean(axis=0), np.cov(points, rowvar=False)) for name, points in atlas.items()}
    out.mkdir(parents=True, exist_ok=True)
    sim = config["simulation"]
    tonic = protocol["tonic"]
    tone_keys = ("rate_hz", "isi_cv")
    means = {key: {name: np.mean(values) for name, values in tonic[key].items()} for key in tone_keys}
    pulse = protocol["stn_pulse"]
    metrics, firing, curves = [], [], []
    for seed in seeds:
        atlas_pop = population(config, atlas, seed)
        geometries = {"atlas": atlas_pop, "gaussian": gaussian_population(atlas_pop, config, moments, seed)}
        for name, condition in protocol["conditions"].items():
            pop = geometries[condition["geometry"]]
            if condition["patchseq"]:
                pop = patchseq_population(pop, patchseq)
            graph = connect(condition["wiring"], pop, condition_config(config, condition), seed)
            stats = graph_statistics(pop, graph, config)
            keys = {"seed": seed, "condition": name}
            measured = measure_tonic(pop, graph, sim, tonic, seed)
            firing.extend({**keys, "group": region, **values} for region, values in measured.items())
            errors = [float(np.mean([abs(values[key] - means[key][region]) for region, values in measured.items()])) for key in tone_keys]
            responses = measure_pulse(pop, graph, sim, pulse, seed)
            curves.extend({**keys, "time_ms": t * sim["dt_ms"], "intact_hz": intact, "blocked_hz": blocked}
                          for t, (intact, blocked) in enumerate(responses.T))
            metrics.append({**keys, **stats, "tonic_rate_mae_hz": errors[0], "tonic_cv_mae": errors[1],
                            **pulse_metrics(responses, sim["dt_ms"], pulse), **measure_choices(pop, graph, sim, seed)})
        print(f"seed {seed}: {len(protocol['conditions'])} conditions", flush=True)

    for name, rows in (("metrics", metrics), ("tonic", firing), ("pulse_curves", curves)):
        write_csv(out / f"{name}.csv", rows)
    manifest = {"seeds": seeds, "python": platform.python_version(), "numpy": np.__version__,
                "model_config": config, "protocol": protocol, "patchseq_summary": patchseq, "atlas": atlas_info,
                "raw_csv_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.glob("*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    patchseq = json.loads((HERE / "patchseq_tau.json").read_text(encoding="utf-8"))
    execute(config, protocol, patchseq, ROOT / "atlas_data", args.out, args.seeds or protocol["seeds"])
