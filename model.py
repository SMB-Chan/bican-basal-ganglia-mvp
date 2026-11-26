"""Directed, fixed-out-degree pseudoconnectomes and a delayed LIF simulator."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Population:
    names: tuple
    cell_type: np.ndarray
    channel: np.ndarray
    xyz: np.ndarray
    sign: np.ndarray
    jump: np.ndarray
    bias: np.ndarray
    tau: np.ndarray
    threshold_mv: np.ndarray | None = None
    cue_gain: np.ndarray | None = None
    noise_scale: np.ndarray | None = None
    slow_noise_mv: np.ndarray | None = None
    rebound_mv: np.ndarray | None = None


@dataclass(frozen=True)
class Graph:
    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    delay: np.ndarray
    lengths: np.ndarray


def population(config, atlas, seed):
    groups = config["groups"]
    names = tuple(groups)
    counts = [groups[name]["n"] for name in names]
    if any(n < 2 or n % 2 for n in counts):
        raise ValueError("Each group must have a positive, even count for the two channels")
    cell_type = np.repeat(np.arange(len(names)), counts)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    channel = np.concatenate([rng.permutation(np.repeat([0, 1], n // 2)) for n in counts])
    xyz = np.empty((len(cell_type), 3))
    regions = np.array([groups[name]["region"] for name in names])[cell_type]
    for region in dict.fromkeys(regions):
        use = regions == region
        size = int(use.sum())
        if atlas is None:
            assumed = config["synthetic_geometry_mm"][region]
            xyz[use] = rng.normal(assumed["center"], assumed["sd"], (size, 3))
        else:
            xyz[use] = atlas[region][rng.choice(len(atlas[region]), size, replace=False)]
    return Population(
        names, cell_type, channel, xyz,
        *[np.array([groups[name][key] for name in names])[cell_type] for key in ("sign", "jump_mv", "bias_mv", "tau_ms")],
    )


def type_matrix(names, values, default):
    return np.array([[values.get(source, {}).get(target, default) for target in names] for source in names])


def pathway_matrix(pop, config):
    return type_matrix(pop.names, config["projection_propensity"], 0.0)


def connect(condition, pop, config, seed):
    if condition not in ("A", "B", "C", "D"):
        raise ValueError(f"Unknown condition {condition}")
    cell_type = pop.cell_type
    n = len(cell_type)
    active = np.flatnonzero(cell_type != pop.names.index(config["output_group"]))
    degree = config["out_degree"]
    if not 0 < degree < n:
        raise ValueError("Out-degree must lie between 1 and N−1")

    distances = np.linalg.norm(pop.xyz[active, None] - pop.xyz[None], axis=2)
    # Common random Gumbel scores implement weighted sampling without replacement
    # and pair the wiring draws across conditions within each seed.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    score = rng.gumbel(size=distances.shape)
    pre_type, post_type = cell_type[active, None], cell_type[None]
    if condition != "A":
        score -= distances / config["distance_scale_mm"]
    if condition in ("C", "D"):
        affinity = type_matrix(pop.names, config["local_cell_affinity"], 1.0)
        score += np.log(affinity[pre_type, post_type])
    if condition == "D":
        with np.errstate(divide="ignore"):
            score += np.log(pathway_matrix(pop, config)[pre_type, post_type])
        topography = {(source, target) for source, target in config["topographic_projections"]}
        selective = np.array([[(source, target) in topography for target in pop.names] for source in pop.names])
        same_channel = pop.channel[active, None] == pop.channel[None]
        score += np.log(config["matched_channel_gain"]) * (same_channel & selective[pre_type, post_type])

    row = np.arange(len(active))
    score[row, active] = -np.inf
    choices = np.sort(np.argpartition(score, -degree, axis=1)[:, -degree:], axis=1)
    if not np.isfinite(score[row[:, None], choices]).all():
        raise ValueError(f"{condition}: a source has fewer than {degree} permitted distinct targets")

    targets = np.full((n, degree), -1, dtype=np.int32)
    targets[active] = choices
    lengths = distances[row[:, None], choices]
    sim = config["simulation"]
    physical_delay = sim["synaptic_offset_ms"] + lengths / sim["conduction_mm_per_ms"]
    delay = np.zeros((n, degree), dtype=np.int32)
    delay[active] = np.maximum(1, np.ceil(physical_delay / sim["dt_ms"])).astype(np.int32)
    weights = pop.sign * pop.jump
    return Graph(active, targets, weights, delay, lengths)


def ticks(ms, dt):
    steps = round(ms / dt)
    if not np.isclose(steps * dt, ms):
        raise ValueError(f"{ms} ms must be a multiple of dt={dt} ms")
    return steps


def simulate(pop, graph, sim, noise, initial, cue_bias, *, edge_weights=None):
    dt = sim["dt_ms"]
    baseline = ticks(sim["baseline_ms"], dt)
    duration = baseline + ticks(sim["cue_ms"], dt)
    refractory_steps = ticks(sim["refractory_ms"], dt)
    n = len(pop.cell_type)
    if noise.shape != (duration, n):
        raise ValueError("Noise array must have one independent draw per time step and cell")
    decay = np.exp(-dt / pop.tau)
    v = initial.copy()
    rest = sim["rest_mv"]
    equilibrium = rest + pop.bias
    cue_equilibrium = equilibrium + (cue_bias if pop.cue_gain is None else cue_bias * pop.cue_gain)
    thresholds = sim["threshold_mv"] if pop.threshold_mv is None else pop.threshold_mv
    reset = sim["reset_mv"]
    refractory = np.zeros(n, dtype=np.int32)
    slots = int(graph.delay.max()) + 1
    pending = np.zeros((slots, n))
    active_source = np.zeros(n, dtype=bool)
    active_source[graph.sources] = True
    spikes = np.zeros((duration, n), dtype=bool)
    if edge_weights is None:
        edge_weights = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
    elif edge_weights.shape != graph.targets.shape:
        raise ValueError("Edge weights must match the directed target array")

    exc_tau = float(sim.get("exc_tau_ms", 0.0))
    inh_tau = float(sim.get("inh_tau_ms", 0.0))
    if exc_tau < 0 or inh_tau < 0:
        raise ValueError("Synaptic time constants must be nonnegative")
    has_rebound = pop.rebound_mv is not None
    if exc_tau == 0.0 and inh_tau == 0.0 and not has_rebound:
        for t in range(duration):
            slot = t % slots
            ready = refractory == 0
            drive = equilibrium if t < baseline else cue_equilibrium
            v[:] = np.where(ready, drive + (v - drive) * decay + noise[t] + pending[slot], reset)
            pending[slot].fill(0)
            fired = ready & (v >= thresholds)
            spikes[t] = fired
            v[fired] = reset
            refractory[~ready] -= 1
            refractory[fired] = refractory_steps

            sources = np.flatnonzero(fired & active_source)
            np.add.at(
                pending,
                ((t + graph.delay[sources]) % slots, graph.targets[sources]),
                edge_weights[sources],
            )
        return spikes

    if has_rebound:
        vh = float(sim.get("rebound_vh_mv", -68.5))
        va = float(sim.get("rebound_va_mv", -68.0))
        tau_plus = float(sim.get("rebound_tau_plus_ms", 25.0))
        tau_minus = float(sim.get("rebound_tau_minus_ms", 20.0))
        if tau_plus <= 0 or tau_minus <= 0:
            raise ValueError("Rebound time constants must be positive")
        h_grow = np.exp(-dt / tau_plus)
        h_decay = np.exp(-dt / tau_minus)
        h_hold = np.exp(-dt / (3.0 * tau_minus))
        rebound_gain = (1.0 - decay) * pop.rebound_mv
        h = np.where(initial < vh, 0.2, 0.0)

    exc_decay = np.exp(-dt / exc_tau) if exc_tau > 0 else 0.0
    inh_decay = np.exp(-dt / inh_tau) if inh_tau > 0 else 0.0
    exc_scale = (1.0 - exc_decay) if exc_tau > 0 else 1.0
    inh_scale = (1.0 - inh_decay) if inh_tau > 0 else 1.0
    pending_exc = np.zeros((slots, n))
    pending_inh = np.zeros((slots, n))
    syn_exc = np.zeros(n)
    syn_inh = np.zeros(n)
    pos_weights = np.maximum(edge_weights, 0.0)
    neg_weights = np.minimum(edge_weights, 0.0)
    for t in range(duration):
        slot = t % slots
        syn_exc = syn_exc * exc_decay + exc_scale * pending_exc[slot]
        syn_inh = syn_inh * inh_decay + inh_scale * pending_inh[slot]
        pending_exc[slot].fill(0)
        pending_inh[slot].fill(0)
        ready = refractory == 0
        drive = equilibrium if t < baseline else cue_equilibrium
        v_next = drive + (v - drive) * decay + noise[t] + syn_exc + syn_inh
        if has_rebound:
            v_next = v_next + (ready & (v_next >= va)) * (rebound_gain * h)
        v[:] = np.where(ready, v_next, reset)
        fired = ready & (v >= thresholds)
        spikes[t] = fired
        v[fired] = reset
        if has_rebound:
            deinact = ready & (~fired) & (v < vh)
            inact_fast = (~ready) | fired | (v >= va)
            h = np.where(deinact, 1.0 + (h - 1.0) * h_grow, np.where(inact_fast, h * h_decay, h * h_hold))
        refractory[~ready] -= 1
        refractory[fired] = refractory_steps

        sources = np.flatnonzero(fired & active_source)
        idx = ((t + graph.delay[sources]) % slots, graph.targets[sources])
        np.add.at(pending_exc, idx, pos_weights[sources])
        np.add.at(pending_inh, idx, neg_weights[sources])
    return spikes


def graph_statistics(pop, graph, config):
    n = len(pop.cell_type)
    source = np.repeat(graph.sources, graph.targets.shape[1])
    target = graph.targets[graph.sources].ravel()
    edge_count = len(target)
    forward = source * n + target
    reverse = target * n + source
    in_degree = np.bincount(target, minlength=n)
    pre_type, post_type = pop.cell_type[source], pop.cell_type[target]
    allowed = pathway_matrix(pop, config)[pre_type, post_type] > 0
    signs = pop.sign[source]
    return {
        "neurons": n,
        "edges": edge_count,
        "density": edge_count / (n * (n - 1)),
        "eligible_density": edge_count / (len(graph.sources) * (n - 1)),
        "excitatory_edges": int(np.count_nonzero(signs > 0)),
        "inhibitory_edges": int(np.count_nonzero(signs < 0)),
        "mean_length_mm": float(graph.lengths.mean()),
        "mean_delay_ms": float(graph.delay[graph.sources].mean() * config["simulation"]["dt_ms"]),
        "reciprocal_edge_fraction": float(np.isin(forward, reverse).mean()),
        "same_channel_edge_fraction": float((pop.channel[source] == pop.channel[target]).mean()),
        "permitted_pathway_fraction": float(allowed.mean()),
        "in_degree_sd": float(in_degree.std()),
        "max_in_degree": int(in_degree.max()),
    }


def edge_matrix(pop, graph):
    source = np.repeat(graph.sources, graph.targets.shape[1])
    target = graph.targets[graph.sources].ravel()
    n_types = len(pop.names)
    pair = pop.cell_type[source] * n_types + pop.cell_type[target]
    return np.bincount(pair, minlength=n_types**2).reshape(n_types, n_types)
