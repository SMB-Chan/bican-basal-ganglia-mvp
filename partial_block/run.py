"""Matched-rate comparison of intrinsic GPi drive and GPe-to-GPi synaptic gain."""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import apply_model_spec
from depletion.run import build_condition
from model import connect, population, simulate, ticks
from parkinsonian.run import _stack_populations
from run import trial_stimuli
from validation.run_validation import patchseq_population


HERE = Path(__file__).resolve().parent
CONDITIONS = ("normal", "pd", "h1", "h2")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def calibration(protocol):
    """Interpolate only within the historical bias sweep, separately per model."""
    with (ROOT / protocol["prior_table"]).open(newline="") as f:
        table = {(r["model"], r["condition"]): float(r["GPi_rate_hz_level"]) for r in csv.DictReader(f)}
    shifts = protocol["calibration_shifts_mv"]
    gains = {}
    for model in protocol["models"]:
        rates = [table[model, "pd_full" if mv == 0 else f"pd_gpi_bias_{mv}"] for mv in shifts]
        target = table[model, f"pd_gpe_gpi_x{protocol['h2_gain']}"]
        if not np.all(np.diff(rates) < 0) or not rates[-1] <= target <= rates[0]:
            raise ValueError(f"No monotone interpolation bracket for {model}")
        gains[model] = float(np.interp(target, rates[::-1], shifts[::-1]))
    return gains


def load_context(protocol):
    atlas, _ = load_atlas(ROOT / "atlas_data")
    return (protocol, atlas, read_json(ROOT / "config.json"), read_json(ROOT / "validation/patchseq_tau.json"),
            read_json(ROOT / "parkinsonian/protocol.json")["dopamine_depletion"], calibration(protocol))


def init_worker(protocol):
    global CONTEXT
    CONTEXT = load_context(protocol)


def condition_specs(shift, gain):
    pd = dict(d1=True, d2=True, gpe_stn=True)
    return [dict(d1=False, d2=False, gpe_stn=False), pd, {**pd, "gpi_bias_mv": shift},
            {**pd, "gpi_input_gain": {"GPe": gain}}]


def attenuate(weights, mask, fraction):
    out = weights.copy()
    out[mask] *= 1.0 - fraction
    return out


def simulate_seed(seed):
    protocol, atlas, config, patchseq, depletion, shifts = CONTEXT
    base = patchseq_population(population(config, atlas, seed), patchseq)
    rows = []
    for model, spec_path in protocol["models"].items():
        spec = read_json(ROOT / spec_path)["selected_spec"] if spec_path else None
        pop, cfg, sim = apply_model_spec(base, config, spec)
        graph = connect("D", pop, cfg, seed)
        gpi = pop.cell_type == pop.names.index("GPi")
        if gpi[graph.sources].any():
            raise ValueError("The output population has recurrent or outgoing projections")
        target_mask = (graph.targets >= 0) & gpi[graph.targets] & (graph.weights[:, None] < 0)
        if not np.any(target_mask):
            raise ValueError("No inhibitory GPi inputs to attenuate")
        conditions = condition_specs(shifts[model], protocol["h2_gain"])
        builds = [build_condition(pop, graph, depletion, s) for s in conditions]
        pops, weights = zip(*builds)
        duration_ms = protocol["burn_ms"] + protocol["readout_ms"]
        params = {**sim, "baseline_ms": duration_ms, "cue_ms": 0}
        burn = ticks(protocol["burn_ms"], sim["dt_ms"])
        noise, initial = trial_stimuli(pop, params, seed, protocol["trial"])
        noise = np.tile(noise, (1, len(builds)))
        initial = np.tile(initial, len(builds))
        for fraction in protocol["block_fractions"]:
            changed = [attenuate(w, target_mask, fraction) for w in weights]
            bpop, bgraph, bweights = _stack_populations(pops, graph, changed)
            spikes = simulate(bpop, bgraph, params, noise, initial, np.zeros_like(initial),
                              edge_weights=bweights).reshape(-1, len(builds), len(gpi))
            for index in (2, 3):
                if np.any(spikes[:, 1, :] != spikes[:, index, :], where=np.broadcast_to(~gpi, spikes[:, 1, :].shape)):
                    raise AssertionError("GPi-only perturbation affected an upstream spike")
            if fraction == 1 and not np.array_equal(spikes[:, 1], spikes[:, 3]):
                raise AssertionError("H2 differs from PD despite complete block of its extra input")
            for index, condition in enumerate(CONDITIONS):
                counts = spikes[burn:, index, :].sum(axis=0)
                median = np.median(counts[gpi]) * 1000.0 / protocol["readout_ms"]
                rows.append(dict(seed=seed, model=model, condition=condition, block=fraction,
                                 gpi_rate_hz=float(median)))
    return rows


def summarize(rows, protocol):
    values = {(r["model"], r["condition"], r["block"], r["seed"]): r["gpi_rate_hz"] for r in rows}
    seeds = protocol["seeds"]
    bootstrap = np.random.default_rng(protocol["bootstrap_seed"]).integers(
        len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    summary, decisions = [], {}
    for model in protocol["models"]:
        def rate(condition, fraction):
            return np.array([values[model, condition, fraction, s] for s in seeds])

        def report(measure, first_block, second_block, difference):
            lo, hi = np.quantile(difference[bootstrap].mean(axis=1), [0.025, 0.975])
            summary.append(dict(model=model, measure=measure, block=first_block, comparison_block=second_block,
                                mean_hz=float(difference.mean()), low_hz=float(lo), high_hz=float(hi)))
            return lo, hi

        margin = protocol["sham_match_margin_hz"]
        lo, hi = report("sham_h2_minus_h1", 0, 0, rate("h2", 0) - rate("h1", 0))
        matched = bool(lo >= -margin and hi <= margin)
        for fraction in protocol["block_fractions"]:
            responses = {}
            for condition in CONDITIONS:
                responses[condition] = rate(condition, fraction) - rate(condition, 0)
                report(condition + "_response", fraction, 0, responses[condition])
            lo, _ = report("h2_minus_h1_response", fraction, fraction, responses["h2"] - responses["h1"])
            if fraction == protocol["primary_block"]:
                separated = bool(lo > protocol["separation_margin_hz"])
        low, high = protocol["efficacy_sensitivity"]
        for f2, f1 in [(low, high), (high, low)]:
            report("unequal_efficacy_h2_minus_h1", f2, f1,
                   (rate("h2", f2) - rate("h2", 0)) - (rate("h1", f1) - rate("h1", 0)))
        decisions[model] = dict(matched_sham=matched, separated_at_half=separated, passed=matched and separated)
    return summary, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    protocol = read_json(HERE / "protocol.json")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.workers == 1:
        init_worker(protocol)
        groups = [simulate_seed(s) for s in protocol["seeds"]]
    else:
        with mp.get_context("spawn").Pool(min(args.workers, len(protocol["seeds"])),
                                          initializer=init_worker, initargs=(protocol,)) as pool:
            groups = pool.map(simulate_seed, protocol["seeds"])
    rows = [row for group in groups for row in group]
    summary, decisions = summarize(rows, protocol)
    write_rows(args.out / "rates.csv", rows)
    write_rows(args.out / "contrasts.csv", summary)
    inputs = [HERE / "protocol.json", ROOT / protocol["prior_table"], ROOT / "config.json",
              ROOT / "parkinsonian/protocol.json", ROOT / "validation/patchseq_tau.json",
              *(ROOT / path for path in protocol["models"].values() if path)]
    manifest = dict(base=protocol["base_commit"], python=platform.python_version(), numpy=np.__version__,
                    interpolation_mv=calibration(protocol), runs=len(rows), decisions=decisions,
                    inputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
