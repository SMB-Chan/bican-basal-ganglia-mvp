"""Replace assumed group sizes with MERSCOPE-derived composition and score external macaque benchmarks."""

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from model import connect, graph_statistics, population
from run import write_csv
from validation.run_validation import (condition_config, measure_pulse, measure_tonic, patchseq_population,
                                       pulse_metrics)


HERE = Path(__file__).resolve().parent


def even(x):
    return int(2 * round(x / 2))


def composition_counts(config, census, protocol):
    """Group sizes for every composition. Striatal (500) and extra-striatal (500) totals stay fixed."""
    assumed = {name: group["n"] for name, group in config["groups"].items()}
    measured = census["model_counts_from_measurement"]
    striatal = ("D1", "D2", "SIn")
    pallidal = ("GPe", "STN", "GPi")
    striatal_total = sum(assumed[k] for k in striatal)
    sin = even(striatal_total * protocol["literature_interneuron_fraction"])
    d1 = even((striatal_total - sin) / 2)
    literature = {**assumed, "SIn": sin, "D1": d1, "D2": striatal_total - sin - d1}
    counts = {
        "assumed": assumed,
        "striatal_measured": {**assumed, **{k: measured[k] for k in striatal}},
        "pallidal_measured": {**assumed, **{k: measured[k] for k in pallidal}},
        "both_measured": dict(measured),
        "striatal_literature": literature,
    }
    for name, values in counts.items():
        if sum(values.values()) != sum(assumed.values()) or any(v < 2 or v % 2 for v in values.values()):
            raise ValueError(f"{name}: counts must be positive even numbers with the original total")
    return {name: counts[name] for name in protocol["compositions"]}


def with_counts(config, counts):
    changed = copy.deepcopy(config)
    for name, n in counts.items():
        changed["groups"][name]["n"] = n
    return changed


def in_degree(pop, graph):
    target = graph.targets[graph.sources].ravel()
    degree = np.bincount(target, minlength=len(pop.cell_type))
    return {f"in_degree_{name}": float(degree[pop.cell_type == i].mean()) for i, name in enumerate(pop.names)}


def pair_in_degree(pop, graph):
    """Mean number of inputs from each source group onto one cell of each target group."""
    n_types = len(pop.names)
    source = np.repeat(graph.sources, graph.targets.shape[1])
    target = graph.targets[graph.sources].ravel()
    counts = np.bincount(pop.cell_type[source] * n_types + pop.cell_type[target], minlength=n_types ** 2)
    sizes = np.bincount(pop.cell_type, minlength=n_types)
    return counts.reshape(n_types, n_types) / sizes[None, :]


def drive_matched_weights(pop, graph, reference_in_degree):
    """Per-edge weights that restore the reference mean input count × weight for every group pair."""
    ratio = np.ones_like(reference_in_degree)
    realized = pair_in_degree(pop, graph)
    usable = realized > 0
    ratio[usable] = reference_in_degree[usable] / realized[usable]
    weights = np.broadcast_to(graph.weights[:, None], graph.targets.shape).copy()
    rows = graph.sources
    weights[rows] *= ratio[pop.cell_type[rows, None], pop.cell_type[graph.targets[rows]]]
    unmatched = int(((reference_in_degree > 0) & ~usable).sum())
    return weights, float(np.abs(np.log(ratio[usable & (reference_in_degree > 0)])).max()), unmatched


def execute(config, validation, protocol, census, patchseq, data, out, seeds):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    atlas, atlas_info = load_atlas(data)
    out.mkdir(parents=True, exist_ok=True)
    sim = config["simulation"]
    tonic = validation["tonic"]
    targets = {key: {name: np.mean(values) for name, values in tonic[key].items()} for key in ("rate_hz", "isi_cv")}
    counts = composition_counts(config, census, protocol)
    if next(iter(counts)) != "assumed":
        raise ValueError("The assumed composition must come first; it is the drive-matching reference")
    metrics, firing = [], []
    for seed in seeds:
        reference = {}
        for composition, group_counts in counts.items():
            local = with_counts(config, group_counts)
            base = population(local, atlas, seed)
            stn = int(np.count_nonzero(base.cell_type == base.names.index("STN")))
            for wiring in protocol["wirings"]:
                condition = validation["conditions"][wiring]
                pop = patchseq_population(base, patchseq) if condition["patchseq"] else base
                graph = connect(condition["wiring"], pop, condition_config(local, condition), seed)
                if composition == "assumed":
                    reference[wiring] = pair_in_degree(pop, graph)
                matched, max_log_ratio, unmatched = drive_matched_weights(pop, graph, reference[wiring])
                for drive, weights in (("raw", None), ("drive_matched", matched)):
                    keys = {"seed": seed, "composition": composition, "wiring": wiring, "drive": drive}
                    measured = measure_tonic(pop, graph, sim, tonic, seed, weights)
                    firing.extend({**keys, "group": region, **values} for region, values in measured.items())
                    errors = {f"tonic_{key}_mae": float(np.mean([abs(values[key] - targets[key][region])
                                                                 for region, values in measured.items()]))
                              for key in ("rate_hz", "isi_cv")}
                    rates = {f"rate_{region}_hz": values["rate_hz"] for region, values in measured.items()}
                    pulse = pulse_metrics(measure_pulse(pop, graph, sim, validation["stn_pulse"], seed, weights),
                                          sim["dt_ms"], validation["stn_pulse"])
                    fixed_spec = {**validation["stn_pulse"],
                                  "fraction_stn": protocol["fixed_stimulated_stn_cells"] / stn}
                    fixed = pulse_metrics(measure_pulse(pop, graph, sim, fixed_spec, seed, weights),
                                          sim["dt_ms"], fixed_spec)
                    metrics.append({**keys, **group_counts, **graph_statistics(pop, graph, local),
                                    **in_degree(pop, graph), "max_abs_log_weight_ratio": max_log_ratio,
                                    "unmatched_pairs": unmatched, **errors, **rates, **pulse,
                                    **{f"fixed60_{k}": v for k, v in fixed.items()}})
        print(f"seed {seed}: {len(counts)} compositions x {len(protocol['wirings'])} wirings x 2 drives", flush=True)
    write_csv(out / "composition_metrics.csv", metrics)
    write_csv(out / "composition_tonic.csv", firing)
    manifest = {"seeds": seeds, "python": platform.python_version(), "numpy": np.__version__,
                "counts": counts, "protocol": protocol, "validation_protocol": validation, "model_config": config,
                "census_sha256": census["source_sha256"], "atlas": atlas_info,
                "raw_csv_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(out.glob("composition_*.csv"))}}
    (out / "composition_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--census", type=Path, default=HERE / "results" / "census.json")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    protocol = read(HERE / "composition_protocol.json")
    execute(read(ROOT / "config.json"), read(ROOT / "validation" / "protocol.json"), protocol, read(args.census),
            read(ROOT / "validation" / "patchseq_tau.json"), ROOT / "atlas_data", args.out,
            args.seeds or protocol["seeds"])
