"""Cross-check one complete seed with individually simulated, unstacked networks."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import apply_model_spec
from model import connect, population, simulate, ticks
from parkinsonian.run import configure_pd_condition
from run import trial_stimuli
from validation.run_validation import patchseq_population


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(results):
    protocol = load(Path(__file__).with_name("protocol.json"))
    seed = protocol["seeds"][0]
    with (results / "rates.csv").open() as f:
        all_rows = list(csv.DictReader(f))
    keys = {(int(r["seed"]), r["model"], float(r["block"]), r["condition"]): float(r["gpi_rate_hz"])
            for r in all_rows}
    states = ("normal", "pd", "h1", "h2")
    expected = {(s, m, b, c) for s in protocol["seeds"] for m in protocol["models"]
                for b in protocol["block_fractions"] for c in states}
    if expected != keys.keys() or len(all_rows) != len(expected):
        raise AssertionError("Missing, repeated or unexpected condition")
    config = load(ROOT / "config.json")
    atlas, _ = load_atlas(ROOT / "atlas_data")
    base = patchseq_population(population(config, atlas, seed), load(ROOT / "validation/patchseq_tau.json"))
    depletion = load(ROOT / "parkinsonian/protocol.json")["dopamine_depletion"]
    shifts = load(results / "manifest.json")["interpolation_mv"]
    count = 0
    for model, path in protocol["models"].items():
        spec = load(ROOT / path)["selected_spec"] if path else None
        pop, cfg, sim = apply_model_spec(base, config, spec)
        graph = connect("D", pop, cfg, seed)
        total = protocol["burn_ms"] + protocol["readout_ms"]
        sim.update(baseline_ms=total, cue_ms=0)
        noise, initial = trial_stimuli(pop, sim, seed, protocol["trial"])
        gpi = pop.cell_type == pop.names.index("GPi")
        pd, pd_weights, _ = configure_pd_condition(pop, graph, depletion, "pd_intact")
        standard_weights = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
        for state in states:
            cells = pop if state == "normal" else pd
            if state == "h1":
                cells = replace(pd, bias=pd.bias + shifts[model] * gpi)
            for fraction in protocol["block_fractions"]:
                weights = np.array(standard_weights if state == "normal" else pd_weights, copy=True)
                for i in graph.sources:
                    if pop.sign[i] < 0:
                        selected = gpi[graph.targets[i]]
                        if state == "h2" and pop.cell_type[i] == pop.names.index("GPe"):
                            weights[i, selected] *= protocol["h2_gain"]
                        weights[i, selected] *= 1 - fraction
                spikes = simulate(cells, graph, sim, noise, initial, np.zeros(len(gpi)), edge_weights=weights)
                kept = spikes[ticks(protocol["burn_ms"], sim["dt_ms"]):, gpi]
                rate = np.median(np.count_nonzero(kept, axis=0)) * (1000.0 / protocol["readout_ms"])
                np.testing.assert_allclose(rate, keys[seed, model, fraction, state], atol=1e-12, rtol=0)
                count += 1
    print(f"{len(expected)} unique rows; seed {seed}: {count} unstacked rates matched within 1e-12 Hz.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    verify(parser.parse_args().results)
