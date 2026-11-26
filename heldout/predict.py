"""Model predictions for the DANDI:000947 test, computed with the same windowed metrics as the data.

The three frozen models of Stage 4B are simulated in the normal and dopamine-depleted states on shared noise.
Every GPi cell is scored with heldout.metrics on consecutive 1 s windows.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from cortical.run import apply_model_spec
from heldout.metrics import unit_metrics
from model import connect, population, simulate, ticks
from parkinsonian.run import _stack_populations, configure_pd_condition
from run import trial_stimuli, write_csv
from validation.run_validation import patchseq_population


HERE = Path(__file__).resolve().parent
_CTX = {}


def cell_rows(seed):
    ctx = _CTX
    spec_m, spec_p = ctx["protocol"]["metrics"], ctx["protocol"]["model_predictions"]
    base = patchseq_population(population(ctx["config"], ctx["atlas"], seed), ctx["patchseq"])
    rows = []
    for name, spec in ctx["models"].items():
        pop, cfg, sim = apply_model_spec(base, ctx["config"], spec)
        graph = connect("D", pop, cfg, seed)
        normal, _, _ = configure_pd_condition(pop, graph, ctx["depletion"], "normal_intact")
        pd, pd_w, _ = configure_pd_condition(pop, graph, ctx["depletion"], "pd_intact")
        params = {**sim, "baseline_ms": spec_p["duration_ms"], "cue_ms": 0}
        noise, initial = trial_stimuli(pop, params, seed, spec_p["trial"])
        bpop, bgraph, bew = _stack_populations([normal, pd], graph, [None, pd_w])
        n = len(pop.cell_type)
        spikes = simulate(bpop, bgraph, params, np.tile(noise, (1, 2)), np.tile(initial, 2), np.zeros(2 * n), edge_weights=bew)
        dt_s = sim["dt_ms"] / 1000.0
        burn = ticks(spec_p["burn_in_ms"], sim["dt_ms"])
        n_windows = int((spikes.shape[0] - burn) * dt_s // spec_m["window_s"])
        start = burn * dt_s
        windows = np.column_stack((start + np.arange(n_windows) * spec_m["window_s"],
                                   start + (np.arange(n_windows) + 1) * spec_m["window_s"]))
        gpi = np.flatnonzero(pop.cell_type == pop.names.index("GPi"))
        for state, offset in (("normal", 0), ("pd", n)):
            for cell in gpi:
                times = (np.flatnonzero(spikes[:, offset + cell]) + 0.5) * dt_s
                m = unit_metrics(times, windows, spec_m, restrict_to_span=False)
                rows.append({"seed": seed, "model": name, "state": state, "cell": int(cell), **(m or {})})
    print(f"predict seed {seed}", flush=True)
    return rows


def summarize(rows, protocol):
    stats = protocol["statistics"]
    rng = np.random.default_rng(stats["bootstrap_seed"])
    seeds = sorted({r["seed"] for r in rows})
    draws = rng.integers(0, len(seeds), size=(stats["replicates"], len(seeds)))
    out = []
    for model in protocol["model_predictions"]["models"]:
        for metric in ("rate_hz", "isi_cv", "beta_index"):
            per_seed = {state: [] for state in ("normal", "pd")}
            for seed in seeds:
                for state in per_seed:
                    v = np.array([r[metric] for r in rows if r["seed"] == seed and r["model"] == model and r["state"] == state
                                  and r.get(metric) is not None and np.isfinite(r[metric])])
                    per_seed[state].append(float(np.median(v)) if len(v) else float("nan"))
            normal, pd = np.array(per_seed["normal"]), np.array(per_seed["pd"])
            change = pd - normal
            low, high = np.quantile(change[draws].mean(axis=1), [0.025, 0.975])
            margin = protocol["directions"]["minimum_effect"][metric]
            mean = float(change.mean())
            direction = ("increase" if low > 0 and mean >= margin else
                         "decrease" if high < 0 and mean <= -margin else "no reliable change")
            out.append({"model": model, "metric": metric, "seeds": len(seeds), "normal_median": float(normal.mean()),
                        "pd_median": float(pd.mean()), "change": float(change.mean()), "ci_low": float(low),
                        "ci_high": float(high), "direction": direction})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    protocol = load(HERE / "protocol.json")
    spec = protocol["model_predictions"]
    atlas, _ = load_atlas(ROOT / "atlas_data")
    _CTX.update({"config": load(ROOT / "config.json"), "patchseq": load(ROOT / "validation/patchseq_tau.json"),
                 "atlas": atlas, "protocol": protocol, "depletion": load(ROOT / "parkinsonian/protocol.json")["dopamine_depletion"],
                 "models": {"baseline_lif": None, **{k: load(ROOT / v)["selected_spec"] for k, v in spec["specs"].items()}}})
    workers = min(len(spec["seeds"]), os.cpu_count() or 1)
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for chunk in pool.map(cell_rows, spec["seeds"]) for r in chunk]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "model_cells.csv", rows)
    summary = summarize(rows, protocol)
    write_csv(args.out / "predictions.csv", summary)
    for row in summary:
        print(row["model"], row["metric"], f"{row['change']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]", row["direction"])
