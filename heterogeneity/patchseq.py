"""Load paired cell properties, keeping raw Patch-seq records in a local cache."""

import csv
from dataclasses import dataclass

import numpy as np

from validation.prepare_patchseq import ASSETS, get_files, index_by_cell


PATCHSEQ_ASSETS = {
    **ASSETS,
    "ap_features.csv": "d80c13b8bba45560252ded4b42edc6ca4552ac4eaeabf6a527b4cb62f2c5a44d",
    "donor.csv": "e4d348b4d9489188ca88d669dd0ab7434a18954d8b61944172dd8b822390740f",
}
TYPES = ("D1", "D2")
RHESUS = "Macaca mulatta"


@dataclass(frozen=True)
class Cell:
    kind: str
    donor: str
    species: str
    tau_ms: float
    gap_mv: float
    resistance_mohm: float
    rheobase_pa: float | None

    def predictors(self):
        return self.tau_ms, self.gap_mv, self.resistance_mohm


def load_cells(cache, qc, fetch=False):
    get_files(cache, fetch, PATCHSEQ_ASSETS)
    metadata = index_by_cell(cache / "cell_metadata.csv")
    mapping = index_by_cell(cache / "mmc_results.csv")
    sweeps = index_by_cell(cache / "cross_sweep_long_square_features.csv")
    spikes = index_by_cell(cache / "ap_features.csv")
    with (cache / "donor.csv").open(encoding="utf-8", newline="") as f:
        species = {row["donor_label"]: row["species_scientific_name"] for row in csv.DictReader(f)}
    cells = []
    for label, row in sorted(mapping.items()):
        kinds = [kind for kind in TYPES
                 if row["Subclass_name"] == f"STR {kind} MSN" and row["Group_name"].startswith(f"STRd {kind} ")]
        if not kinds or min(float(row["Subclass_bootstrapping_probability"]),
                            float(row["Group_bootstrapping_probability"])) < qc["min_bootstrapping_probability"]:
            continue
        sweep = sweeps.get(label, {})
        fields = [sweep.get("tau"), sweep.get("v_baseline"), sweep.get("input_resistance_ss"),
                  spikes.get(label, {}).get("threshold_v_short_square")]
        if not all(fields):
            continue
        tau, rest, resistance, threshold = map(float, fields)
        tau *= 1000
        gap = threshold - rest
        values = {"tau_ms": tau, "v_rest_mv": rest, "gap_mv": gap, "resistance_mohm": resistance}
        if not all(qc[key][0] <= value <= qc[key][1] for key, value in values.items()):
            continue
        rheobase = float(sweep["rheobase_i"]) if sweep.get("rheobase_i") else None
        if rheobase is not None and (not np.isfinite(rheobase) or rheobase <= 0):
            rheobase = None
        donor = metadata[label]["donor_label"]
        cells.append(Cell(kinds[0], donor, species[donor], tau, gap, resistance, rheobase))
    return cells


def summarize(cells):
    rows = []
    for species in ("pooled", *sorted({cell.species for cell in cells})):
        for kind in TYPES:
            group = [cell for cell in cells if cell.kind == kind and (species == "pooled" or cell.species == species)]
            values = np.array([cell.predictors() for cell in group])
            quantiles = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
            correlation = np.corrcoef(values, rowvar=False)
            distribution = {}
            for name, column in zip(("tau_ms", "gap_mv", "resistance_mohm"), quantiles.T):
                distribution.update({f"{name}_{stat}": float(value) for stat, value in zip(("q1", "median", "q3"), column)})
            rows.append({"species": species, "kind": kind, "cells": len(group),
                         "donors": len({cell.donor for cell in group}),
                         **distribution,
                         "r_tau_gap": float(correlation[0, 1]), "r_tau_resistance": float(correlation[0, 2]),
                         "r_gap_resistance": float(correlation[1, 2])})
    return rows
