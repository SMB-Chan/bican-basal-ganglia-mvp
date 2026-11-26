"""Additional Stage 4B summaries: every pairwise model comparison, and tests 3-4 scored only when test 2 passed.

Tests 3 and 4 ask whether a blockade reduces 8-15 Hz oscillation. When no oscillation emerged (test 2 failed),
a pass reduces to a firing-rate change, so the conditional score credits tests 3-4 only on seeds that passed test 2.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from analyze import read_csv
from run import write_csv


MODELS = ("baseline_lif", "selected_biophysics", "selected_rebound")
SEED = 20261127
REPLICATES = 5000


def conditional_checks(row):
    oscillation = int(row["pd_beta_emergence_pass"])
    blockade = int(row["tachibana_stn_inactivation_pass"]) + int(row["tachibana_gpe_stn_block_pass"])
    return int(row["pd_checks_passed"]) - blockade * (1 - oscillation)


def summarize(rows):
    seeds = sorted({int(r["seed"]) for r in rows})
    index = {(r["model"], int(r["seed"])): r for r in rows}
    draws = np.random.default_rng(SEED).integers(0, len(seeds), size=(REPLICATES, len(seeds)))
    scores = {(m, kind): np.array([float(index[m, s]["pd_checks_passed"]) if kind == "all_six" else conditional_checks(index[m, s])
                                   for s in seeds]) for m in MODELS for kind in ("all_six", "tests_3_4_conditional")}
    out = []
    for kind in ("all_six", "tests_3_4_conditional"):
        for m in MODELS:
            v = scores[m, kind]
            low, high = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
            out.append({"scoring": kind, "comparison": m, "seeds": len(seeds), "mean": float(v.mean()), "ci_low": float(low), "ci_high": float(high)})
        for a, b in itertools.permutations(MODELS, 2):
            if MODELS.index(a) < MODELS.index(b):
                a, b = b, a
            d = scores[a, kind] - scores[b, kind]
            low, high = np.quantile(d[draws].mean(axis=1), [0.025, 0.975])
            row = {"scoring": kind, "comparison": f"{a} - {b}", "seeds": len(seeds), "mean": float(d.mean()),
                   "ci_low": float(low), "ci_high": float(high)}
            if row not in out:
                out.append(row)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    write_csv(out / "comparisons.csv", summarize(read_csv(out / "evaluations.csv")))
