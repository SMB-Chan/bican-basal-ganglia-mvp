"""Simulate paired GPi measurements for simultaneous intrinsic and synaptic changes."""
import argparse
import csv
import gzip
import io
import itertools
import multiprocessing as mp
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import apply_model_spec
from model import Graph, Population, connect, population, simulate, ticks
from parkinsonian.run import configure_pd_condition
from partial_block.run import read_json
from run import trial_stimuli
from validation.run_validation import patchseq_population

HERE = Path(__file__).resolve().parent


def init_worker():
    global ATLAS
    ATLAS, _ = load_atlas(ROOT / "atlas_data")


def make_network(model, seed, atlas, setup):
    prior = read_json(ROOT / "partial_block/protocol.json")
    model_path = prior["models"][model]
    spec = read_json(ROOT / model_path)["selected_spec"] if model_path else None
    cfg = read_json(ROOT / "config.json")
    pop = patchseq_population(population(cfg, atlas, seed), read_json(ROOT / "validation/patchseq_tau.json"))
    pop, cfg, sim = apply_model_spec(pop, cfg, spec)
    graph = connect("D", pop, cfg, seed)
    pd, weights, _ = configure_pd_condition(pop, graph,
        read_json(ROOT / "parkinsonian/protocol.json")["dopamine_depletion"], "pd_intact")
    ids = np.flatnonzero(pd.cell_type == pd.names.index("GPi"))
    cells = np.random.default_rng([seed, 611]).choice(ids, size=setup["gpi_sample"], replace=False)
    if np.intersect1d(ids, graph.sources).size or sim.get("exc_tau_ms", 0) or sim.get("inh_tau_ms", 0):
        raise ValueError("Replay requires output-only GPi and instantaneous synapses")
    if pd.rebound_mv is not None and np.any(pd.rebound_mv[cells]):
        raise ValueError("Selected GPi cells must have zero rebound gain")
    sim = {**sim, "baseline_ms": setup["burn_ms"] + setup["measure_ms"], "cue_ms": 0}
    return pd, graph, weights, sim, cells


def incoming_edges(pop, graph, weights, cells):
    index = np.full(len(pop.cell_type), -1, dtype=int)
    index[cells] = np.arange(len(cells))
    source, slot = np.where((graph.targets >= 0) & (index[graph.targets] >= 0))
    target = index[graph.targets[source, slot]]
    return source, target, graph.delay[source, slot], weights[source, slot]


def currents(pop, edges, spikes, n_cells):
    """Sum delayed inputs to recorded cells in three groups whose gains can differ."""
    source, target, delays, weights = edges
    exc, inhib, gpe = np.zeros((3, len(spikes), n_cells))
    gpe_id = pop.names.index("GPe")
    for pre, post, delay, weight in zip(source, target, delays, weights):
        array = gpe if pop.cell_type[pre] == gpe_id else (inhib if weight < 0 else exc)
        array[delay:, post] += weight * spikes[:-delay, pre]
    return exc, inhib, gpe


def replay(pop, sim, cells, noise, initial, bias, burn):
    """Run the repository's LIF integrator on sampled cells with supplied net input."""
    bias = np.broadcast_to(bias, (len(bias), len(cells)))
    k = len(bias)
    n = bias.size
    flat = Population(pop.names, np.tile(pop.cell_type[cells], k), np.tile(pop.channel[cells], k),
        np.tile(pop.xyz[cells], (k, 1)), np.tile(pop.sign[cells], k), np.tile(pop.jump[cells], k),
        (pop.bias[cells] + bias).ravel(), np.tile(pop.tau[cells], k),
        threshold_mv=None if pop.threshold_mv is None else np.tile(pop.threshold_mv[cells], k))
    empty = Graph(np.empty(0, dtype=int), np.full((n, 1), -1, dtype=int), np.zeros(n),
                  np.zeros((n, 1), dtype=int), np.empty((0, 1)))
    spikes = simulate(flat, empty, sim, noise.reshape(len(noise), n), np.tile(initial[cells], k), np.zeros(n))
    rate = spikes[burn:].sum(axis=0).reshape(k, len(cells)) * 1000 / ((len(spikes) - burn) * sim["dt_ms"])
    return np.median(rate, axis=1)


def probe(pop, cells, edges, seed, params, dt):
    """Average time-locked voltage deflection from repeated isolated GPe volleys."""
    source, target, delays, weights = edges
    gpe_id = pop.names.index("GPe")
    gpe = np.flatnonzero(pop.cell_type == gpe_id)
    keep = pop.cell_type[source] == gpe_id
    pre, post, delay, weight = (v[keep] for v in (source, target, delays, weights))
    rng = np.random.default_rng([seed, 642])
    repetitions = params["repetitions"]
    prefix = ticks(5, dt)
    shape = (repetitions, prefix + ticks(params["last_ms"], dt) + 1, len(cells))
    traces = np.zeros(shape)
    decay = np.exp(-dt / pop.tau[cells])
    for trial in range(repetitions):
        recruited = rng.choice(gpe, size=round(len(gpe) * params["gpe_fraction"]), replace=False)
        lag = np.full(len(pop.bias), -1, dtype=int)
        lag[recruited] = rng.integers(ticks(params["max_jitter_ms"], dt) + 1, size=len(recruited))
        use = lag[pre] >= 0
        arrival = prefix + lag[pre[use]] + delay[use]
        np.add.at(traces[trial], (arrival, post[use]), weight[use])
        for t in range(prefix + 1, shape[1]):
            traces[trial, t] += decay * traces[trial, t - 1]
    signal = traces.mean(axis=(0, 2))
    pre_noise, post_noise = [rng.normal(0, params["noise_mv_sd"], shape).mean(axis=(0, 2)) for _ in range(2)]

    def peak(gain, blocked, noise):
        trace = gain * (1 - blocked) * signal + noise
        return max(0.0, float(trace[:prefix].mean() - trace[prefix:].min()))

    return lambda gain, fraction: (peak(gain, 0, pre_noise), peak(gain, fraction, post_noise))


def simulate_model(model, seed, split, setup):
    pop, graph, weights, sim, cells = make_network(model, seed, ATLAS, setup)
    shifts = read_json(ROOT / "partial_block/results/manifest.json")["interpolation_mv"]
    strengths = setup[split + "_strengths"]
    fractions = setup[split + "_block"]
    edges = incoming_edges(pop, graph, weights, cells)
    burn = ticks(setup["burn_ms"], sim["dt_ms"])
    space = list(itertools.product(strengths, strengths))
    a, b = np.array(space).T
    rates, isolated = {}, {}
    for epoch in ("pre", "post"):
        noise, initial = trial_stimuli(pop, sim, seed, setup["trials"][epoch])
        spikes = simulate(pop, graph, sim, noise, initial, np.zeros(len(pop.bias)), edge_weights=weights)
        exc, inhib, gpe = currents(pop, edges, spikes, len(cells))
        common = (noise[:, cells] + exc)[:, None, :]
        inhibitory = inhib[:, None, :] + (1 + b[None, :, None]) * gpe[:, None, :]
        for fraction in ([0] if epoch == "pre" else fractions):
            rates[fraction] = replay(pop, sim, cells, common + (1 - fraction) * inhibitory,
                                     initial, a[:, None] * shifts[model], burn)
    for epoch, step in zip(("isol_low", "isol_high"), setup["isolated_steps_mv"]):
        noise, initial = trial_stimuli(pop, sim, seed, setup["trials"][epoch])
        drive = np.broadcast_to(noise[:, None, cells], (len(noise), len(strengths), len(cells)))
        medians = replay(pop, sim, cells, drive, initial, np.array(strengths)[:, None] * shifts[model] + step, burn)
        isolated[epoch] = dict(zip(strengths, medians))
    measure = probe(pop, cells, edges, seed, setup["probe"], sim["dt_ms"])
    rows = []
    for j, (intrinsic, synaptic) in enumerate(space):
        for fraction in fractions:
            pre, post = measure(1 + synaptic, fraction)
            estimated = float(np.clip(1 - post / pre, 0, 1)) if pre > 1e-6 else 0.5
            rows.append(dict(split=split, model=model, seed=seed, h1=intrinsic, h2=synaptic,
                             eff_true=fraction, rate_pre=rates[0][j], rate_delta=rates[fraction][j] - rates[0][j],
                             isol_low=isolated["isol_low"][intrinsic], isol_high=isolated["isol_high"][intrinsic],
                             peak_pre=pre, peak_post=post, eff_proxy=estimated))
    return rows


def simulate_seed(job):
    split, seed = job
    setup = read_json(HERE / "protocol.json")
    models = read_json(ROOT / "partial_block/protocol.json")["models"]
    return [r for model in models for r in simulate_model(model, seed, split, setup)]


def execute(output, workers):
    setup = read_json(HERE / "protocol.json")
    jobs = [(split, s) for split in ("train", "test") for s in setup[split + "_seeds"]]
    if workers < 1:
        raise ValueError("worker count must be positive")
    if set(setup["train_seeds"]) & set(setup["test_seeds"]):
        raise ValueError("Overlapping training and evaluation seeds")
    output.mkdir(parents=True, exist_ok=False)
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), initializer=init_worker) as pool:
        with (output / "observations.csv.gz").open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as f:
                    writer = None
                    for rows in pool.imap(simulate_seed, jobs):
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                            writer.writeheader()
                        writer.writerows(rows)
    from joint_inference.score import score
    score(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    execute(args.out, args.workers)
