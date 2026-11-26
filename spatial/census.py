"""Census of the full public macaque MERSCOPE release: composition, regional densities and D1/D2 mixing."""

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BASE = "https://allen-brain-cell-atlas.s3.amazonaws.com/metadata/HMBA-BG-MERSCOPE-Macaque/20261101/"
TABLES = {
    "cell_metadata.csv": ("cell_label", "brain_section_label", "qc_pass", "cell_volume"),
    "slab_plane_coordinates.csv": ("cell_label", "x_slab_mm", "y_slab_mm", "slab_plane_label"),
    "cell_anatomical_annotations.csv": ("cell_label", "parcellation_term_symbol"),
    "mmc_results.csv": ("cell_label", "Neighborhood", "Class", "Subclass", "Group",
                        "Subclass_bootstrapping_probability", "Group_bootstrapping_probability", "hierarchy_consistent"),
}
COMPLETE = ("specimen_metadata.csv", "donor.csv")
# Model regions and the MERSCOPE anatomical annotation symbol used for each.
REGION_SYMBOL = {"GPe": "GPe", "STN": "STH", "GPi": "GPi"}
# Hemisphere voxel counts of the bundled 160 µm MRI annotation (atlas.py, RESULTS.md).
MRI_VOXELS = {"STRd": 372123, "GPe": 42309, "STN": 3700, "GPi": 20363}
MRI_VOXEL_MM3 = 0.16 ** 3
NON_NEURON = {"Nonneuron", "Adjacent"}
INTERNEURON_CLASSES = {"CN MGE GABA", "CN CGE GABA"}
SPN_SUBCLASSES = {"STR D1 SPN", "STR D2 SPN"}


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in (*TABLES, *COMPLETE):
        path = directory / name
        with urllib.request.urlopen(urllib.request.Request(BASE + name, method="HEAD"), timeout=60) as response:
            expected = int(response.headers["Content-Length"])
        if path.exists() and path.stat().st_size == expected:
            continue
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(BASE + name, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1 << 22):
                handle.write(chunk)
        if partial.stat().st_size != expected:
            raise ValueError(f"{name}: received {partial.stat().st_size} bytes, expected {expected}")
        partial.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def read_columns(path, columns):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = [header.index(column) for column in columns]
        values = [[] for _ in columns]
        for row in reader:
            for store, i in zip(values, index):
                store.append(row[i])
    return dict(zip(columns, values))


def load(directory):
    """Join the four per-cell tables on cell_label and keep QC-passing cells on their specimen plane."""
    tables = {name: read_columns(directory / name, columns) for name, columns in TABLES.items()}
    labels = tables["cell_metadata.csv"]["cell_label"]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate cell labels in cell_metadata.csv")
    cells = {"cell_label": np.array(labels)}
    for name, table in tables.items():
        order = None
        if table["cell_label"] != labels:
            position = {label: i for i, label in enumerate(table["cell_label"])}
            order = np.array([position[label] for label in labels])
        for column, values in table.items():
            if column == "cell_label":
                continue
            values = np.array(values)
            cells[column] = values if order is None else values[order]
    specimens = {row["brain_section_label"]: row for row in csv.DictReader(
        (directory / "specimen_metadata.csv").open(encoding="utf-8-sig"))}
    specimen_plane = np.array([specimens[s]["slab_plane_label"] for s in cells["brain_section_label"]])
    keep = (cells["qc_pass"] == "True") & (specimen_plane == cells["slab_plane_label"])
    out = {key: value[keep] for key, value in cells.items()}
    out["cell_volume"] = np.array([float(v) if v else np.nan for v in out["cell_volume"]])
    out["x"] = out.pop("x_slab_mm").astype(float)
    out["y"] = out.pop("y_slab_mm").astype(float)
    out["kind"] = classify(out)
    info = {"joined_cells": len(labels), "qc_pass_and_matched_specimen_plane": int(keep.sum()),
            "slab_planes": len(set(out["slab_plane_label"])), "sections": len(set(out["brain_section_label"])),
            "hemispheres": sorted({row["hemisphere"] for row in specimens.values()}),
            "slabs": sorted({row["brain_slab_label"] for row in specimens.values()}),
            "donors": sorted({row["donor_label"] for row in specimens.values()})}
    return out, info


def classify(cells):
    group, subclass = cells["Group"], cells["Subclass"]
    kind = np.full(len(group), "other", dtype=object)
    neuron = ~np.isin(cells["Neighborhood"], list(NON_NEURON))
    kind[neuron] = "neuron"
    spn = np.isin(subclass, list(SPN_SUBCLASSES))
    kind[spn & np.char.startswith(group.astype(str), "STRd ")] = "STRd_SPN"
    kind[spn & np.char.startswith(group.astype(str), "STRv ")] = "STRv_SPN"
    kind[subclass == "STR Eccentric SPN"] = "eSPN"
    kind[np.isin(cells["Class"], list(INTERNEURON_CLASSES))] = "interneuron"
    return kind


def tile_keys(cells, use, tile_mm):
    return np.char.add(np.char.add(cells["brain_section_label"][use].astype(str), ":"),
                       np.char.add(np.floor(cells["x"][use] / tile_mm).astype(int).astype(str),
                                   np.char.add(",", np.floor(cells["y"][use] / tile_mm).astype(int).astype(str))))


def dorsal_striatum(cells, tile_mm, min_spn=5):
    """Mask of STR-annotated cells in tiles where dorsal SPN groups outnumber ventral ones."""
    in_str = cells["parcellation_term_symbol"] == "STR"
    keys = tile_keys(cells, in_str, tile_mm)
    kinds = cells["kind"][in_str]
    dorsal = Counter(keys[kinds == "STRd_SPN"])
    ventral = Counter(keys[kinds == "STRv_SPN"])
    accepted = {key for key, n in dorsal.items() if n >= min_spn and n > ventral.get(key, 0)}
    mask = np.zeros(len(in_str), dtype=bool)
    mask[np.flatnonzero(in_str)[np.isin(keys, list(accepted))]] = True
    return mask


def striatal_composition(cells, tile_mm):
    mask = dorsal_striatum(cells, tile_mm)
    kinds = cells["kind"][mask]
    counts = Counter(kinds)
    spn = mask & (cells["kind"] == "STRd_SPN")
    d1 = cells["Subclass"][spn] == "STR D1 SPN"
    striosome = np.char.find(cells["Group"][spn].astype(str), "Striosome") >= 0
    model_neurons = counts["STRd_SPN"] + counts["interneuron"]
    all_neurons = model_neurons + counts["eSPN"] + counts["STRv_SPN"]
    inter_groups = Counter(cells["Group"][mask & (cells["kind"] == "interneuron")])
    per_section = []
    sections = cells["brain_section_label"][spn]
    for section in sorted(set(sections)):
        use = sections == section
        if use.sum() >= 500:
            per_section.append({"section": section, "dorsal_spn": int(use.sum()), "d1_fraction": float(d1[use].mean())})
    return {
        "tile_mm": tile_mm, "dorsal_cells": int(mask.sum()), "counts": {k: int(v) for k, v in counts.items()},
        "d1_fraction_of_dorsal_spn": float(d1.mean()), "striosome_fraction_of_dorsal_spn": float(striosome.mean()),
        "interneuron_fraction_of_spn_plus_interneurons": counts["interneuron"] / model_neurons,
        "interneuron_fraction_of_all_striatal_neurons": counts["interneuron"] / all_neurons,
        "interneuron_groups_fraction_of_all_striatal_neurons": {g: n / all_neurons for g, n in inter_groups.most_common()},
        "sections_with_500_dorsal_spn": per_section,
    }


def regional_density(cells, tile_mm, dorsal_mask):
    """Neurons per mm² of tiled section area; relative 3D counts assume equal section thickness and detection."""
    neuron = cells["kind"] != "other"
    rows = {}
    masks = {"STRd": dorsal_mask, **{r: cells["parcellation_term_symbol"] == s for r, s in REGION_SYMBOL.items()}}
    for region, use in masks.items():
        area = len(set(tile_keys(cells, use, tile_mm))) * tile_mm ** 2
        n = int((use & neuron).sum())
        rows[region] = {"neurons": n, "cells": int(use.sum()), "area_mm2": area, "neurons_per_mm2": n / area,
                        "neuron_fraction": n / int(use.sum()),
                        "sections": len(set(cells["brain_section_label"][use])),
                        "mri_volume_mm3": MRI_VOXELS[region] * MRI_VOXEL_MM3,
                        "median_segmented_neuron_diameter_um": float(np.nanmedian(
                            (6 * cells["cell_volume"][use & neuron] / np.pi) ** (1 / 3)))}
    relative = {r: v["neurons_per_mm2"] * v["mri_volume_mm3"] for r, v in rows.items()}
    pallidal = {r: relative[r] for r in REGION_SYMBOL}
    total = sum(pallidal.values())
    for region in rows:
        rows[region]["relative_count_vs_GPe"] = relative[region] / relative["GPe"]
        if region in pallidal:
            rows[region]["share_of_GPe_STN_GPi"] = pallidal[region] / total
    return {"tile_mm": tile_mm, "regions": rows}


def mixing(cells, k=8, permutations=500, tile_mm=0.5, seed=175031):
    """Same-type fraction among the k nearest high-confidence dorsal SPNs, within each tissue section."""
    spn = cells["kind"] == "STRd_SPN"
    confident = (cells["hierarchy_consistent"] == "True") & (cells["parcellation_term_symbol"] == "STR")
    confident &= np.minimum(cells["Subclass_bootstrapping_probability"].astype(float),
                            cells["Group_bootstrapping_probability"].astype(float)) >= 0.8
    use = spn & confident & np.isfinite(cells["x"]) & np.isfinite(cells["y"])
    rng = np.random.default_rng(seed)
    rows = []
    for section in sorted(set(cells["brain_section_label"][use])):
        idx = np.flatnonzero(use & (cells["brain_section_label"] == section))
        if len(idx) < 1000:
            continue
        xy = np.column_stack([cells["x"][idx], cells["y"][idx]])
        neighbor = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
        d1 = cells["Subclass"][idx] == "STR D1 SPN"
        striosome = np.char.find(cells["Group"][idx].astype(str), "Striosome") >= 0
        tiles = np.floor(xy / tile_mm).astype(int)
        _, tile_id = np.unique(tiles, axis=0, return_inverse=True)
        row = {"section": section, "plane": cells["slab_plane_label"][idx[0]], "spn": len(idx),
               "d1_fraction": float(d1.mean()), "striosome_fraction": float(striosome.mean())}
        for name, label in (("d1d2", d1), ("compartment", striosome)):
            observed = float((label[:, None] == label[neighbor]).mean())
            null = {"section": [], "tile": []}
            by_tile = np.argsort(tile_id, kind="stable")
            for _ in range(permutations):
                shuffled = rng.permutation(label)
                null["section"].append(float((shuffled[:, None] == shuffled[neighbor]).mean()))
                # Sorting by tile with random tie-breaks permutes labels within each tile only.
                local = np.empty_like(label)
                local[by_tile] = label[np.lexsort((rng.random(len(label)), tile_id))]
                null["tile"].append(float((local[:, None] == local[neighbor]).mean()))
            row[f"{name}_observed"] = observed
            for scope, values in null.items():
                lo, hi = np.quantile(values, [0.025, 0.975])
                row[f"{name}_{scope}_null_mean"] = float(np.mean(values))
                row[f"{name}_{scope}_null_q025"] = float(lo)
                row[f"{name}_{scope}_null_q975"] = float(hi)
                row[f"{name}_{scope}_excess"] = observed - float(np.mean(values))
        rows.append(row)
    return rows


def model_counts(composition, density, striatal_total=500, other_total=500):
    """Even group counts that keep the model's striatal and extra-striatal totals fixed."""
    def even(x):
        return int(2 * round(x / 2))
    sin = even(striatal_total * composition["interneuron_fraction_of_spn_plus_interneurons"])
    d1 = even((striatal_total - sin) * composition["d1_fraction_of_dorsal_spn"])
    shares = {r: density["regions"][r]["share_of_GPe_STN_GPi"] for r in REGION_SYMBOL}
    gpe, stn = even(other_total * shares["GPe"]), even(other_total * shares["STN"])
    return {"D1": d1, "D2": striatal_total - sin - d1, "SIn": sin, "GPe": gpe, "STN": stn, "GPi": other_total - gpe - stn}


def execute(directory, out, permutations):
    cells, info = load(directory)
    composition = {tile: striatal_composition(cells, tile) for tile in (0.25, 0.5)}
    dorsal = dorsal_striatum(cells, 0.25)
    density = {tile: regional_density(cells, tile, dorsal) for tile in (0.05, 0.1, 0.2)}
    mixing_rows = mixing(cells, permutations=permutations)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "spn_mixing_by_section.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mixing_rows[0]))
        writer.writeheader()
        writer.writerows(mixing_rows)
    neurons_by_region = defaultdict(Counter)
    for region, subclass, kind in zip(cells["parcellation_term_symbol"], cells["Subclass"], cells["kind"]):
        if kind != "other":
            neurons_by_region[region][subclass] += 1
    summary = {
        "release": "HMBA-BG-MERSCOPE-Macaque 20261101", "license": "CC BY-NC 4.0", "url_prefix": BASE,
        "source_sha256": {name: sha256(directory / name) for name in (*TABLES, *COMPLETE)}, **info,
        "coordinate_system": "2D slab plane, millimetres; neighbours are computed within one tissue section",
        "neuron_subclasses_by_region": {r: dict(c.most_common(12)) for r, c in sorted(neurons_by_region.items())},
        "dorsal_striatum_composition": {str(k): v for k, v in composition.items()},
        "regional_density": {str(k): v for k, v in density.items()},
        "model_counts_from_measurement": model_counts(composition[0.25], density[0.1]),
        "mixing_permutations": permutations,
    }
    (out / "census.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()
    if args.fetch:
        fetch(args.cache)
    result = execute(args.cache, args.out, args.permutations)
    print(json.dumps(result["model_counts_from_measurement"]))
