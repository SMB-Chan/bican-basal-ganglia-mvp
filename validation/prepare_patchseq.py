"""Derive dorsal-striatal membrane-time-constant summaries from ABC Atlas Patch-seq."""

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np


BASE = "https://allen-brain-cell-atlas.s3.amazonaws.com/metadata/HMBA-Macaque-PatchSeq/20260228/"
ASSETS = {
    "cell_metadata.csv": "5cec3b3eac79fcb05717434cf95fabd6ebaa9828e4b5f90c88f50e94d24e7ede",
    "cross_sweep_long_square_features.csv": "6839ec7ae3462823d9b9590821be7bcdc8004720713344542d209975116df275",
    "mmc_results.csv": "aee9b60c16085f4e860f13f0451115c03324b05be48bb8a4cb8c480e36267e99",
    "electrophysiological_feature_lookup.xlsx": "c10ac74f0600f7407d70944b46c4f2324c4a3e8889912babb0c162a3d0bc3fee",
}


def get_files(directory, fetch):
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in ASSETS.items():
        path = directory / name
        if fetch and not path.exists():
            temp = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(BASE + name, timeout=30) as source, temp.open("wb") as target:
                shutil.copyfileobj(source, target)
            if hashlib.sha256(temp.read_bytes()).hexdigest() != digest:
                raise ValueError(f"File hash mismatch: {temp}")
            temp.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"File hash mismatch: {path}")


def index_by_cell(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    indexed = {row["cell_label"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate cell label in {path}")
    return indexed


def summarize(directory):
    mapping = index_by_cell(directory / "mmc_results.csv")
    metadata = index_by_cell(directory / "cell_metadata.csv")
    measurements = index_by_cell(directory / "cross_sweep_long_square_features.csv")
    groups = {}
    for kind in ("D1", "D2"):
        donors = set()
        values = []
        removed = 0
        for cell, row in mapping.items():
            if row["Subclass_name"] != f"STR {kind} MSN" or not row["Group_name"].startswith(f"STRd {kind} "):
                continue
            if float(row["Subclass_bootstrapping_probability"]) < 0.8 or float(row["Group_bootstrapping_probability"]) < 0.8:
                continue
            seconds = measurements.get(cell, {}).get("tau")
            if not seconds:
                continue
            tau = float(seconds) * 1000
            if not 1 <= tau <= 100:
                removed += 1
                continue
            values.append(tau)
            donors.add(metadata[cell]["donor_label"])
        if not values:
            raise ValueError(f"No qualifying Patch-seq cells for {kind}")
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75]).tolist()
        groups[kind] = {"cells": len(values), "donors": len(donors), "excluded_tau_range": removed,
                        "q1_tau_ms": q1, "median_tau_ms": median, "q3_tau_ms": q3}
    return {
        "dataset": "HMBA-Macaque-PatchSeq 20260228", "paper": "10.1016/j.cell.2026.10.028",
        "license": "CC BY-NC 4.0", "source_base_url": BASE, "source_sha256": ASSETS,
        "filter": "MapMyCells dorsal D1/D2 group and matching subclass, both bootstrap probabilities >=0.8; finite tau 1–100 ms",
        "groups": groups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fetch", action="store_true")
    options = parser.parse_args()
    get_files(options.cache, options.fetch)
    options.out.write_text(json.dumps(summarize(options.cache), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
