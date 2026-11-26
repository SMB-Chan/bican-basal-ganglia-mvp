"""Measure functional collisions within fixed-constraint graph ensembles."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from atlas import ROOT, load_atlas
from model import connect, edge_matrix, population
from run import write_csv
from validation.run_validation import measure_pulse, measure_tonic, patchseq_population, pulse_metrics


HERE = Path(__file__).resolve().parent
REGIONS = ("GPe", "STN", "GPi")
SIGNS = ("early_excitation", "late_inhibition", "blocked_early_excitation", "late_attenuation")
WINDOWS = ("pulse_early_hz", "pulse_late_hz", "blocked_early_hz", "blocked_late_hz")
CONTINUOUS = tuple(f"{region}_{metric}" for region in REGIONS for metric in ("rate_hz", "isi_cv")) + WINDOWS


def functional_matches(left, right, tolerances):
    bounds = [tolerances["rate_hz"], tolerances["isi_cv"]] * len(REGIONS)
    bounds += [tolerances["early_hz"], tolerances["late_hz"]] * 2
    delta = np.array([abs(left[key] - right[key]) for key in CONTINUOUS])
    return bool(np.all(delta <= bounds))


def pair_stats(rows, edge_ids, protocol):
    edges_per_graph = len(next(iter(edge_ids.values())))
    results = []
    for placement in sorted({row["placement"] for row in rows}):
        for membrane in protocol["membrane_conditions"]:
            indices = [i for i, row in enumerate(rows) if row["placement"] == placement and row["membrane"] == membrane]
            for a, i in enumerate(indices):
                left = rows[i]
                for j in indices[a + 1:]:
                    right = rows[j]
                    same_family = left["hypothesis"] == right["hypothesis"]
                    # Within a family compare distinct draws. Between families pair the same Gumbel draw.
                    if not same_family and left["graph_seed"] != right["graph_seed"]:
                        continue
                    signs_equal = all(left[key] == right[key] for key in SIGNS)
                    close = functional_matches(left, right, protocol["pair_tolerances"])
                    adj_overlap = None
                    if close or same_family:
                        left_edges = edge_ids[placement, left["hypothesis"], left["graph_seed"]]
                        right_edges = edge_ids[placement, right["hypothesis"], right["graph_seed"]]
                        shared = np.intersect1d(left_edges, right_edges, assume_unique=True).size
                        adj_overlap = shared / (2 * edges_per_graph - shared)
                    results.append({"placement": placement, "membrane": membrane,
                                    "left_hypothesis": left["hypothesis"], "right_hypothesis": right["hypothesis"],
                                    "left_graph_seed": left["graph_seed"], "right_graph_seed": right["graph_seed"],
                                    "signs_equal": int(signs_equal), "continuous_close": int(close), "edge_jaccard": adj_overlap,
                                    "stn_to_gpe_fraction_gap": abs(left["stn_to_gpe_fraction"] - right["stn_to_gpe_fraction"])})
    return results


def execute(config, validation, patchseq, protocol, out, placements, transform=None, extra=None):
    """Evaluate the graph ensemble. `transform` changes cell dynamics only; graphs use the untransformed cells."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    atlas, info = load_atlas(ROOT / "atlas_data")
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, edges = [], [], {}
    sim = config["simulation"]
    tonic = validation["tonic"]
    truth = {region: np.mean(tonic["rate_hz"][region]) for region in REGIONS}
    pulse = {**validation["stn_pulse"], "repetitions": protocol["pulse_repetitions"]}
    for placement in placements:
        original = population(config, atlas, placement)
        dynamic = original if transform is None else transform(original)
        populations = {"E": dynamic, "F": patchseq_population(dynamic, patchseq)}
        indices = {name: original.names.index(name) for name in ("STN", "GPe", "GPi")}
        for draw in range(protocol["draws_per_placement"]):
            graph_seed = protocol["first_graph_seed"] + draw
            for name, stn_odds in protocol["hypotheses"].items():
                cfg = {**config, "matched_channel_gain": 1.0,
                       "projection_propensity": {**config["projection_propensity"], "STN": stn_odds}}
                graph = connect("D", original, cfg, graph_seed)
                source = np.repeat(graph.sources, graph.targets.shape[1])
                edges[placement, name, graph_seed] = source * len(original.cell_type) + graph.targets[graph.sources].ravel()
                matrix = edge_matrix(original, graph)
                fraction = float(matrix[indices["STN"], indices["GPe"]] / matrix[indices["STN"]].sum())
                for membrane in protocol["membrane_conditions"]:
                    pop = populations[membrane]
                    activity = measure_tonic(pop, graph, sim, tonic, placement)
                    tonic_mae = float(np.mean([abs(activity[name]["rate_hz"] - truth[name]) for name in REGIONS]))
                    response = measure_pulse(pop, graph, sim, pulse, placement)
                    scores = pulse_metrics(response, sim["dt_ms"], pulse)
                    coarse = tonic_mae <= protocol["external_rate_mae_hz"] and scores["pulse_features"] == 4
                    strict = coarse and abs(scores["blocked_late_hz"]) <= protocol["external_blocked_late_abs_hz"]
                    rows.append({"placement": placement, "graph_seed": graph_seed, "membrane": membrane,
                                 "hypothesis": name, "neurons": len(pop.cell_type), "edges": len(source),
                                 "stn_to_gpe_fraction": fraction,
                                 **{f"{region}_{key}": activity[region][key] for region in REGIONS for key in ("rate_hz", "isi_cv")},
                                 "tonic_rate_mae_hz": tonic_mae, **scores, "external_coarse": int(coarse), "external_strict": int(strict)})
                    curves.append(response)
        print(f"placement {placement}: {protocol['draws_per_placement']} draws per projection hypothesis", flush=True)
    write_csv(out / "evaluations.csv", rows)
    write_csv(out / "pairs.csv", pair_stats(rows, edges, protocol))
    np.savez_compressed(out / "response_arrays.npz", responses=np.array(curves))
    manifest = {**(extra or {}), "protocol": protocol, "placements": placements, "model_config": config, "validation": validation,
                "patchseq_summary": patchseq, "atlas": info, "python": platform.python_version(), "numpy": np.__version__,
                "raw_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir())}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--placements", type=int, nargs="+")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "validation/protocol.json").read_text(encoding="utf-8"))
    patchseq = json.loads((ROOT / "validation/patchseq_tau.json").read_text(encoding="utf-8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    execute(config, validation, patchseq, protocol, args.out, args.placements or protocol["placements"])
