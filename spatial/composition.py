"""At which spatial scale is the D1/D2 composition of macaque dorsal striatum organized?

Uses the same public MERSCOPE prefix and cell selection as `spatial.preflight`. Two label pairs are tested on
the same cells and positions:
- D1 vs D2 SPN (the question), and
- striosome vs matrix SPN (positive control: striosomes are known to form patches).
For each label pair: nearest-neighbour same-label fraction against shuffles, and a tile dispersion index
across 0.25–2 mm tiles against whole-plane label shuffles.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, save
from spatial.preflight import mixing, neighbors, select_spns


SETTINGS = {"seed": 520117, "permutations": 500, "neighbors": 8, "scales_mm": [0.25, 0.5, 1.0, 2.0],
            "min_cells_per_tile": 10, "grid_offsets": [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
            "min_cells_per_plane": 1000}


def compartment(group):
    if "Striosome" in group:
        return "striosome"
    if "Matrix" in group:
        return "matrix"
    return None


def dispersion_index(positions, positive, scale, offset, min_cells):
    """Pearson chi-square of tile label fractions divided by its degrees of freedom.

    Values near 1 match independent labels; larger values mean composition differs between tiles.
    """
    tiles = np.floor((positions - positions.min(axis=0)) / scale + np.asarray(offset)).astype(int)
    _, tile_id = np.unique(tiles, axis=0, return_inverse=True)
    counts = np.bincount(tile_id)
    hits = np.bincount(tile_id, weights=positive.astype(float))
    use = counts >= min_cells
    p = hits[use].sum() / counts[use].sum()
    chi_square = np.sum((hits[use] - counts[use] * p) ** 2 / (counts[use] * p * (1 - p)))
    return float(chi_square / max(use.sum() - 1, 1)), int(use.sum())


def multiscale(positions, positive, settings, rng):
    rows = []
    for scale in settings["scales_mm"]:
        observed, tiles = zip(*(dispersion_index(positions, positive, scale, offset, settings["min_cells_per_tile"])
                                for offset in settings["grid_offsets"]))
        null = []
        for _ in range(settings["permutations"]):
            shuffled = rng.permutation(positive)
            null.append(np.mean([dispersion_index(positions, shuffled, scale, offset, settings["min_cells_per_tile"])[0]
                                 for offset in settings["grid_offsets"]]))
        null = np.array(null)
        value = float(np.mean(observed))
        rows.append({"scale_mm": scale, "tiles": int(np.mean(tiles)), "dispersion_index": value,
                     "null_mean": float(null.mean()), "null_q975": float(np.quantile(null, 0.975)),
                     "one_sided_p": float((np.sum(null >= value) + 1) / (len(null) + 1))})
    return rows


def execute(directory, settings):
    _, _, _, selected = select_spns(directory)
    rng = np.random.default_rng(settings["seed"])
    planes = {}
    for plane in sorted({row[3] for row in selected}):
        group = [row for row in selected if row[3] == plane]
        if len(group) < settings["min_cells_per_plane"]:
            continue
        positions = np.array([row[0] for row in group])
        sections = [row[2] for row in group]
        kind = np.array([row[1] for row in group])
        compartments = np.array([compartment(row[4]) for row in group], dtype=object)
        labelled = compartments != None  # noqa: E711 - elementwise test on an object array
        planes[plane] = {
            "cells": len(group), "counts": dict(Counter(kind)),
            "compartment_counts": dict(Counter(c for c in compartments if c is not None)),
            "D1_vs_D2": {"neighbors": mixing(positions, kind, sections, settings["neighbors"], settings["permutations"]),
                         "multiscale": multiscale(positions, kind == "D1", settings, rng)},
            "striosome_vs_matrix": {
                "cells": int(labelled.sum()),
                "neighbors": mixing(positions[labelled], compartments[labelled].astype(str),
                                    [s for s, keep in zip(sections, labelled) if keep], settings["neighbors"], settings["permutations"]),
                "multiscale": multiscale(positions[labelled], compartments[labelled] == "striosome", settings, rng)},
            "_points": (positions, kind, compartments)}
    return planes


def make_figure(planes, path):
    plane, data = max(planes.items(), key=lambda item: item[1]["cells"])
    positions, kind, compartments = data["_points"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), layout="constrained")
    fig.suptitle("Spatial organization of dorsal-striatum SPNs in one macaque slab plane", fontsize=15, fontweight="bold")
    for ax, labels, palette, title in (
            (axes[0], kind, {"D1": "#2e6f9e", "D2": "#c0602e"}, "D1 vs D2 SPN"),
            (axes[1], compartments, {"matrix": "#b9c2cb", "striosome": "#7a2f7a"}, "Striosome vs matrix SPN")):
        for name, color in palette.items():
            use = labels == name
            ax.scatter(positions[use, 0], positions[use, 1], s=1.2, c=color, label=f"{name} ({use.sum()})", alpha=0.8)
        ax.set(title=title, xlabel="Slab x (mm)", ylabel="Slab y (mm)", aspect="equal")
        ax.legend(frameon=False, fontsize=8, markerscale=6, loc="upper right")
    ax = axes[2]
    for key, color, name in (("D1_vs_D2", "#2e6f9e", "D1 vs D2"), ("striosome_vs_matrix", "#7a2f7a", "Striosome vs matrix")):
        rows = data[key]["multiscale"]
        scales = [r["scale_mm"] for r in rows]
        ax.plot(scales, [r["dispersion_index"] for r in rows], "o-", color=color, label=f"{name}, observed")
        ax.plot(scales, [r["null_q975"] for r in rows], ":", color=color, label=f"{name}, shuffle 97.5%")
    ax.axhline(1, color="#929ca9", lw=0.8, linestyle="--")
    ax.set(xscale="log", yscale="log", title="Composition differences between tiles", xlabel="Tile size (mm)",
           ylabel="Dispersion index (χ²/df, 1 = random)")
    ax.set_xticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="Directory filled by `spatial.preflight --fetch`")
    parser.add_argument("--out", type=Path, required=True, help="Folder for composition.json and composition.png")
    args = parser.parse_args()
    planes = execute(args.cache, SETTINGS)
    args.out.mkdir(parents=True, exist_ok=True)
    figure_style()
    make_figure(planes, args.out / "composition.png")
    report = {"settings": SETTINGS, "source": "spatial/preflight.json (same inputs and cell selection)",
              "planes": {plane: {k: v for k, v in data.items() if k != "_points"} for plane, data in planes.items()}}
    (args.out / "composition.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
