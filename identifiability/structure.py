"""Which level of wiring do the stage-1 measurements constrain, and does discrimination survive animal-level noise?

1. Edge level. Compare the edge overlap of functionally close pairs with all pairs drawn under the same
   hypothesis (label-permutation test). Cells of one group are interchangeable in the model, so any
   population measurement is unchanged when targets are swapped within a group.
2. Block level. Regenerate each graph from the saved seeds and count edges per (source group, target group).
   Relate functional distance to block distance and to edge distance across pairs.
3. Measurement noise. Classify held-out graphs after adding noise to the test measurements: tonic rate and
   CV noise from the spread of the three reference monkeys, and an assumed sweep for stimulus windows.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from atlas import ROOT, load_atlas
from identifiability.run import CONTINUOUS, REGIONS, WINDOWS
from model import connect, edge_matrix, population
from run import write_csv


TONIC_RATES = tuple(f"{region}_rate_hz" for region in REGIONS)
TONIC = tuple(f"{region}_{key}" for region in REGIONS for key in ("rate_hz", "isi_cv"))
FEATURE_SETS = {"tonic_rates": TONIC_RATES, "tonic_rates_and_cv": TONIC, "continuous": CONTINUOUS}


def block_matrices(manifest):
    """Regenerate every stage-1 graph and return its group-pair edge counts, keyed like pairs.csv."""
    config, protocol = manifest["model_config"], manifest["protocol"]
    atlas, _ = load_atlas(ROOT / "atlas_data")
    blocks, fractions = {}, {}
    for placement in manifest["placements"]:
        pop = population(config, atlas, placement)
        stn, gpe = pop.names.index("STN"), pop.names.index("GPe")
        for draw in range(protocol["draws_per_placement"]):
            seed = protocol["first_graph_seed"] + draw
            for name, odds in protocol["hypotheses"].items():
                cfg = {**config, "matched_channel_gain": 1.0,
                       "projection_propensity": {**config["projection_propensity"], "STN": odds}}
                matrix = edge_matrix(pop, connect("D", pop, cfg, seed))
                blocks[placement, name, seed] = matrix
                fractions[placement, name, seed] = matrix[stn, gpe] / matrix[stn].sum()
    return blocks, fractions


def functional_scale(rows):
    """Per-membrane SD of each continuous measure across all evaluations, used to standardize distances."""
    scale = {}
    for membrane in {row["membrane"] for row in rows}:
        values = np.array([[float(row[key]) for key in CONTINUOUS] for row in rows if row["membrane"] == membrane])
        sd = values.std(axis=0)
        sd[sd < 1e-12] = 1.0
        scale[membrane] = sd
    return scale


def pair_structure(rows, pairs, blocks):
    index = {(int(r["placement"]), r["membrane"], r["hypothesis"], int(r["graph_seed"])): r for r in rows}
    scale = functional_scale(rows)
    edges = int(next(iter(blocks.values())).sum())
    table = []
    for p in pairs:
        placement, membrane = int(p["placement"]), p["membrane"]
        left = (placement, p["left_hypothesis"], int(p["left_graph_seed"]))
        right = (placement, p["right_hypothesis"], int(p["right_graph_seed"]))
        a, b = index[(placement, membrane) + left[1:]], index[(placement, membrane) + right[1:]]
        delta = np.array([float(a[key]) - float(b[key]) for key in CONTINUOUS]) / scale[membrane]
        table.append({"placement": placement, "membrane": membrane, "same_hypothesis": int(left[1] == right[1]),
                      "pair": f"{left[1]}|{right[1]}", "continuous_close": int(p["continuous_close"]),
                      "edge_jaccard": float(p["edge_jaccard"]) if p["edge_jaccard"] != "" else None,
                      # Fraction of edges that would have to move between group pairs to turn one graph into the other.
                      "block_distance": float(np.abs(blocks[left] - blocks[right]).sum() / (2 * edges)),
                      "functional_distance": float(np.sqrt(np.mean(delta ** 2)))})
    return table


def rank(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def edge_level_test(table, replicates, seed):
    """Is the edge Jaccard of close pairs different from pairs drawn under the same hypothesis at random?"""
    rng = np.random.default_rng(seed)
    results = []
    for membrane in sorted({row["membrane"] for row in table}):
        within = [row for row in table if row["membrane"] == membrane and row["same_hypothesis"]]
        jaccard = np.array([row["edge_jaccard"] for row in within])
        close = np.array([row["continuous_close"] for row in within], dtype=bool)
        strata = np.array([f"{row['placement']}|{row['pair']}" for row in within])
        observed = jaccard[close].mean() - jaccard[~close].mean()
        null = np.empty(replicates)
        for i in range(replicates):
            permuted = close.copy()
            for stratum in np.unique(strata):
                use = strata == stratum
                permuted[use] = rng.permutation(close[use])
            null[i] = jaccard[permuted].mean() - jaccard[~permuted].mean()
        results.append({"membrane": membrane, "within_hypothesis_pairs": len(within), "close_pairs": int(close.sum()),
                        "close_mean_jaccard": float(jaccard[close].mean()), "other_mean_jaccard": float(jaccard[~close].mean()),
                        "difference": float(observed), "null_q025": float(np.quantile(null, 0.025)),
                        "null_q975": float(np.quantile(null, 0.975)),
                        "two_sided_p": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (replicates + 1)),
                        "spearman_functional_vs_edge_distance": spearman([row["functional_distance"] for row in within], 1 - jaccard),
                        "spearman_functional_vs_block_distance_within": spearman([row["functional_distance"] for row in within],
                                                                                 [row["block_distance"] for row in within])})
    return results


def block_summary(table):
    rows = []
    for membrane in sorted({row["membrane"] for row in table}):
        for same, label in ((1, "same hypothesis"), (0, "different hypotheses")):
            sample = [row for row in table if row["membrane"] == membrane and row["same_hypothesis"] == same]
            block = np.array([row["block_distance"] for row in sample])
            function = np.array([row["functional_distance"] for row in sample])
            rows.append({"membrane": membrane, "comparison": label, "pairs": len(sample),
                         "block_distance_median": float(np.median(block)), "block_distance_max": float(block.max()),
                         "functional_distance_median": float(np.median(function)),
                         "spearman_functional_vs_block": spearman(function, block)})
        everything = [row for row in table if row["membrane"] == membrane]
        rows.append({"membrane": membrane, "comparison": "all pairs", "pairs": len(everything),
                     "block_distance_median": float(np.median([r["block_distance"] for r in everything])),
                     "block_distance_max": float(max(r["block_distance"] for r in everything)),
                     "functional_distance_median": float(np.median([r["functional_distance"] for r in everything])),
                     "spearman_functional_vs_block": spearman([r["functional_distance"] for r in everything],
                                                              [r["block_distance"] for r in everything])})
    return rows


def reference_noise(validation):
    tonic = validation["tonic"]
    sd = {}
    for region in REGIONS:
        sd[f"{region}_rate_hz"] = float(np.std(tonic["rate_hz"][region], ddof=1))
        sd[f"{region}_isi_cv"] = float(np.std(tonic["isi_cv"][region], ddof=1))
    return sd


def noisy_discrimination(rows, protocol, validation, settings):
    """Held-out nearest-centroid classification with noise added to test measurements only.

    Distances are scaled by sqrt(model spread² + noise SD²) per feature, the matched rule for independent
    Gaussian noise, so features with tiny model spread are not over-weighted once noise is present.
    """
    reference = reference_noise(validation)
    placement = np.array([int(r["placement"]) for r in rows])
    parity = np.array([int(r["graph_seed"]) % 2 for r in rows])
    membrane = np.array([r["membrane"] for r in rows])
    label = np.array([r["hypothesis"] for r in rows])
    rng = np.random.default_rng(settings["seed"])
    out = []
    for condition in protocol["membrane_conditions"]:
        for feature_name, keys in FEATURE_SETS.items():
            x = np.array([[float(r[key]) for key in keys] for r in rows])
            for tonic_scale in settings["tonic_noise_multipliers"]:
                window_levels = settings["window_noise_sd_hz"] if any(k in WINDOWS for k in keys) else [0.0]
                for window_sd in window_levels:
                    noise_sd = np.array([reference[k] * tonic_scale if k in reference else window_sd for k in keys])
                    accuracies = []
                    for p in sorted(set(placement)):
                        for split in (0, 1):
                            test = (placement == p) & (membrane == condition) & (parity == split)
                            train = (placement != p) & (membrane == condition) & (parity != split)
                            classes = np.unique(label[train])
                            means = np.array([x[train & (label == c)].mean(axis=0) for c in classes])
                            spread = np.sqrt(x[train].var(axis=0) + noise_sd ** 2)
                            spread[spread < 1e-12] = 1.0
                            for _ in range(settings["draws"]):
                                observed = x[test] + rng.standard_normal(x[test].shape) * noise_sd
                                d = (((observed[:, None] - means[None]) / spread) ** 2).sum(axis=2)
                                accuracies.append(np.mean(classes[d.argmin(axis=1)] == label[test]))
                    out.append({"membrane": condition, "features": feature_name, "tonic_noise_x_reference_sd": tonic_scale,
                                "window_noise_sd_hz": window_sd if any(k in WINDOWS for k in keys) else None,
                                "accuracy": float(np.mean(accuracies)), "chance": 1 / len(protocol["hypotheses"])})
    return out, reference


def make_figure(out, table, noise_rows):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), layout="constrained")
    fig.suptitle("What the stage-1 measurements constrain", fontsize=15, fontweight="bold")
    ax = axes[0]
    within = [r for r in table if r["membrane"] == "F" and r["same_hypothesis"]]
    for close, color, name in ((False, "#9aa6b2", "Other pairs"), (True, "#a34b31", "Functionally close pairs")):
        values = [r["edge_jaccard"] for r in within if bool(r["continuous_close"]) == close]
        ax.hist(values, bins=30, alpha=0.75, color=color, label=f"{name} (n={len(values)})", density=True)
    ax.set(title="Edge overlap, same hypothesis | F", xlabel="Edge Jaccard", ylabel="Density")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    sample = [r for r in table if r["membrane"] == "F"]
    colors = ["#207c74" if r["same_hypothesis"] else "#a34b31" for r in sample]
    ax.scatter([r["block_distance"] for r in sample], [r["functional_distance"] for r in sample], s=6, c=colors, alpha=0.5)
    ax.scatter([], [], c="#207c74", label="Same hypothesis")
    ax.scatter([], [], c="#a34b31", label="Different hypotheses")
    ax.set(title="Group-level wiring vs function | F", xlabel="Fraction of edges moved between group pairs",
           ylabel="Functional distance (SD units)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    for (features, window_sd), color, name in (((
            "tonic_rates", None), "#647b91", "Tonic rates"), (("tonic_rates_and_cv", None), "#437fb4", "Rates + CV"),
            (("continuous", 0.0), "#207c74", "Continuous, windows noiseless"),
            (("continuous", 5.0), "#a34b31", "Continuous, windows ±5 Hz")):
        sample = sorted((r for r in noise_rows if r["membrane"] == "F" and r["features"] == features
                         and r["window_noise_sd_hz"] == window_sd), key=lambda r: r["tonic_noise_x_reference_sd"])
        ax.plot([r["tonic_noise_x_reference_sd"] for r in sample], [r["accuracy"] for r in sample], "o-", color=color, label=name)
    ax.axhline(1 / 3, color="#778595", linestyle="--", lw=1, label="Chance")
    ax.set(title="Held-out hypothesis accuracy with noise | F", xlabel="Tonic noise (× SD across 3 reference monkeys)",
           ylabel="Accuracy", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, out / "structure_and_noise.png")


def analyze(results, settings):
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(results / "evaluations.csv")
    blocks, fractions = block_matrices(manifest)
    mismatch = max(abs(fractions[int(r["placement"]), r["hypothesis"], int(r["graph_seed"])] - float(r["stn_to_gpe_fraction"]))
                   for r in rows)
    if mismatch > 1e-12:
        raise ValueError(f"Regenerated graphs do not match the saved run (max gap {mismatch})")
    table = pair_structure(rows, read_csv(results / "pairs.csv"), blocks)
    edge = edge_level_test(table, settings["permutations"], settings["seed"])
    block = block_summary(table)
    noise, reference = noisy_discrimination(rows, manifest["protocol"], manifest["validation"], settings)
    write_csv(results / "structure_pairs.csv", table)
    write_csv(results / "edge_level_test.csv", edge)
    write_csv(results / "block_level_summary.csv", block)
    write_csv(results / "noise_discrimination.csv", noise)
    (results / "noise_reference.json").write_text(json.dumps({"settings": settings, "reference_sd": reference}, indent=2) + "\n",
                                                  encoding="utf-8")
    figure_style()
    make_figure(results, table, noise)
    return edge, block, noise


SETTINGS = {"seed": 410227, "permutations": 2000, "draws": 100,
            "tonic_noise_multipliers": [0.0, 0.25, 0.5, 0.75, 1.0],
            "window_noise_sd_hz": [0.0, 1.0, 2.0, 5.0, 10.0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    analyze(parser.parse_args().results, SETTINGS)
