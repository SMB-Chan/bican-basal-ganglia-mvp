"""Verify the public macaque slab join and measure cell-type mixing in one data prefix."""

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE = "https://allen-brain-cell-atlas.s3.amazonaws.com/metadata/HMBA-BG-MERSCOPE-Macaque/20261101/"
RANGED = ("cell_metadata.csv", "slab_plane_coordinates.csv", "cell_anatomical_annotations.csv", "mmc_results.csv")
COMPLETE = ("specimen_metadata.csv", "donor.csv")
PREFIX_BYTES = 8_388_608


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in RANGED + COMPLETE:
        path = directory / name
        if path.exists():
            continue
        headers = {"Range": f"bytes=0-{PREFIX_BYTES - 1}"} if name in RANGED else {}
        with urllib.request.urlopen(urllib.request.Request(BASE + name, headers=headers), timeout=60) as response:
            if name in RANGED and response.status != 206:
                raise ValueError(f"Expected bounded HTTP 206 response for {name}")
            data = response.read(PREFIX_BYTES + 1)
        if len(data) > PREFIX_BYTES or (name in RANGED and len(data) != PREFIX_BYTES):
            raise ValueError(f"Unexpected response size for {name}")
        path.write_bytes(data)


def read(directory, name):
    data = (directory / name).read_bytes()
    if name in RANGED:
        if len(data) != PREFIX_BYTES:
            raise ValueError(f"Expected a {PREFIX_BYTES}-byte sample for {name}")
        data = data[:data.rfind(b"\n") + 1]
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def neighbors(positions, k):
    neighbor_ids = np.empty((len(positions), k), dtype=int)
    for start in range(0, len(positions), 256):
        stop = min(start + 256, len(positions))
        i = np.arange(start, stop)
        dist_sq = (positions[i, 0, None] - positions[:, 0]) ** 2 + (positions[i, 1, None] - positions[:, 1]) ** 2
        dist_sq[np.arange(len(i)), i] = np.inf
        neighbor_ids[start:stop] = np.argpartition(dist_sq, k - 1, axis=1)[:, :k]
    return neighbor_ids


def mixing(positions, labels, sections, k=8, permutations=500):
    adjacency = neighbors(positions, k)
    observed = float((labels[:, None] == labels[adjacency]).mean())
    rng = np.random.default_rng(175031)
    stats = {}
    tiles = np.floor(positions / 0.5).astype(int)
    strata = {"by_section": sections, "by_section_and_0_5_mm_tile": list(zip(sections, tiles[:, 0], tiles[:, 1]))}
    for name, assignment in strata.items():
        bins = defaultdict(list)
        for i, key in enumerate(assignment):
            bins[key].append(i)
        null = []
        for _ in range(permutations):
            shuffled = labels.copy()
            for i in bins.values():
                shuffled[i] = rng.permutation(labels[i])
            null.append(float((shuffled[:, None] == shuffled[adjacency]).mean()))
        lo, hi = np.quantile(null, [0.025, 0.975]).tolist()
        stats[name] = {"strata": len(bins), "mean": float(np.mean(null)), "q025": lo, "q975": hi,
                       "observed_minus_mean": observed - float(np.mean(null))}
    return {"neighbors_per_cell": k, "permutations": permutations, "observed_same_type_fraction": observed, "nulls": stats}


def execute(directory):
    meta, coords, anatomy, types = [{row["cell_label"]: row for row in read(directory, name)} for name in RANGED]
    specimens = {row["brain_section_label"]: row for row in read(directory, "specimen_metadata.csv")}
    shared = sorted(set(meta) & set(coords) & set(anatomy) & set(types))
    validated = [c for c in shared if meta[c]["qc_pass"] == "True"
                 and specimens[meta[c]["brain_section_label"]]["slab_plane_label"] == coords[c]["slab_plane_label"]]
    selected = []
    for cell in validated:
        row = types[cell]
        if anatomy[cell]["STR"] != "True" or not row["Group"].startswith(("STRd D1 ", "STRd D2 ")):
            continue
        if row["Subclass"] not in ("STR D1 SPN", "STR D2 SPN") or row["hierarchy_consistent"] != "True":
            continue
        if min(float(row["Subclass_bootstrapping_probability"]), float(row["Group_bootstrapping_probability"])) < 0.8:
            continue
        coord = coords[cell]
        xyz = (float(coord["x_slab_mm"]), float(coord["y_slab_mm"]))
        if not np.isfinite(xyz).all():
            continue
        selected.append((xyz, row["Subclass"].split()[1], meta[cell]["brain_section_label"], coord["slab_plane_label"]))

    tested = {}
    for plane in sorted({row[3] for row in selected}):
        group = [row for row in selected if row[3] == plane]
        positions = np.array([row[0] for row in group])
        labels = np.array([row[1] for row in group])
        sections = [row[2] for row in group]
        tested[plane] = {"counts": dict(Counter(labels)), "sections": sorted(set(sections)),
                         "slab_xy_mm_min": positions.min(axis=0).tolist(), "slab_xy_mm_max": positions.max(axis=0).tolist(),
                         **mixing(positions, labels, sections)}
    return {"release": "HMBA-BG-MERSCOPE-Macaque 20261101", "license": "CC BY-NC 4.0", "url_prefix": BASE,
            "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in RANGED + COMPLETE},
            "ranged_first_bytes_per_table": PREFIX_BYTES, "donors": [row["donor_label"] for row in read(directory, "donor.csv")],
            "table_rows": dict(zip(RANGED, map(len, (meta, coords, anatomy, types)))), "shared_cell_ids": len(shared),
            "qc_pass_and_matched_specimen_plane": len(validated), "coordinate_system": "2D slab plane, millimetres",
            "planes": tested}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.cache)
    args.out.write_text(json.dumps(execute(args.cache), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
