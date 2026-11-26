"""Read the pinned HMBA rhesus macaque atlas and extract one-sided region masks."""

import argparse
import ast
import csv
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BUCKET = "https://allen-atlas-assets.s3.us-west-2.amazonaws.com"
ANNOTATION = "annotation-sets/hmba-adult-rhesusmacaque-homba-annotation/2025"
TERMINOLOGY = "terminologies/hmba-mammalian-homba-terminology/2025"
ASSETS = {
    "rhesus_annotations_160.nii.gz": (
        f"{ANNOTATION}/annotations_compressed_160.nii.gz",
        "4fe297fa8d8e90b7713c759b3c071624775bf3211097f4f116f9cba94bba69c7",
    ),
    "terminology.csv": (
        f"{TERMINOLOGY}/terminology.csv",
        "f0131888ef69cdf81de6199101d36684eaffe056864761af3059513dd08fa43f",
    ),
    "volumes.csv": (
        f"{ANNOTATION}/parcellation_volumes.csv",
        "a77e9ac590282df90efa131e6869a0712d3b595e11506d931aa0538788de545f",
    ),
}
REGIONS = {"STRd": "HOMBA:AA30192", "GPe": "HOMBA:10343", "STN": "HOMBA:10466", "GPi": "HOMBA:10344"}


def verified_path(directory, name):
    path = directory / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != ASSETS[name][1]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return path


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (key, _) in ASSETS.items():
        if not (directory / name).exists():
            temp = directory / (name + ".partial")
            with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=60) as remote, temp.open("wb") as local:
                shutil.copyfileobj(remote, local)
            if hashlib.sha256(temp.read_bytes()).hexdigest() != ASSETS[name][1]:
                raise ValueError(f"SHA-256 mismatch for {temp}")
            temp.replace(directory / name)
        verified_path(directory, name)


def load_atlas(directory):
    import nibabel as nib

    image = nib.load(verified_path(directory, "rhesus_annotations_160.nii.gz"))
    labels = np.asanyarray(image.dataobj)
    if labels.ndim != 3 or image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Expected a 3D annotation volume in millimetres")
    with verified_path(directory, "terminology.csv").open(encoding="utf-8", newline="") as f:
        terms = {row["identifier"]: row for row in csv.DictReader(f)}
    with verified_path(directory, "volumes.csv").open(encoding="utf-8", newline="") as f:
        reference_counts = {row["identifier"]: int(row["voxel_count"]) for row in csv.DictReader(f)}

    # Along the left-right anatomical axis, keep the half at higher RAS x.
    # Image grid midpoint, in physical space, is a reproducible midline proxy.
    midline = nib.affines.apply_affine(image.affine, (np.array(labels.shape) - 1) / 2)[0]
    atlas = {}
    info = {
        "release": "HMBA Adult Rhesus Macaque Brain Atlas 2025",
        "license": "CC BY 4.0",
        "shape": list(labels.shape),
        "affine_ras_mm": image.affine.tolist(),
        "selection": "RAS x greater than image-grid physical midpoint",
        "midline_ras_x_mm": float(midline),
        "assets": {name: {"url": f"{BUCKET}/{key}", "sha256": digest} for name, (key, digest) in ASSETS.items()},
        "regions": {},
    }
    for name, identifier in REGIONS.items():
        values = ast.literal_eval(terms[identifier]["descendant_annotation_values"])
        voxel_indices = np.argwhere(np.isin(labels, values))
        if len(voxel_indices) != reference_counts[identifier]:
            raise ValueError(f"{name}: count disagrees with the official parcellation volume table")
        centers = nib.affines.apply_affine(image.affine, voxel_indices)
        atlas[name] = points = centers[centers[:, 0] > midline]
        if not len(points):
            raise ValueError(f"Empty mask for {name}")
        info["regions"][name] = {
            "homba_identifier": identifier,
            "atlas_abbreviation": terms[identifier]["abbreviation"],
            "nonzero_labels": sorted(np.unique(labels[tuple(voxel_indices.T)]).astype(int).tolist()),
            "voxels_full_volume": len(centers),
            "voxels_selected": len(points),
            "centroid_ras_mm": points.mean(axis=0).tolist(),
        }
    return atlas, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "atlas_data")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.data)
    _, info = load_atlas(args.data)
    print(json.dumps(info, indent=2))
