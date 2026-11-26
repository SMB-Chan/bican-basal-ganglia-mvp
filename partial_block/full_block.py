"""Exploratory, post hoc: GPi rate under complete block of inhibitory input, compared with the normal state.

Uses only the existing rates.csv. With every inhibitory input to GPi removed, H2 (stronger GPe->GPi inhibition)
has nothing extra left to act through, while H1 (lower GPi bias) keeps its intrinsic shift. The comparison with the
normal state at the same full block therefore does not depend on how much inhibition each hypothesis had. This
analysis was chosen after the pre-registered results were seen and is not part of the frozen protocol.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODELS = ("baseline_lif", "selected_biophysics", "selected_rebound")
REPLICATES, SEED = 10000, 20261128
# SD of GPi rate across the three awake monkeys of Gatev & Wichmann (validation/protocol.json), for scale only.
ANIMAL_SD_SOURCE = "validation/protocol.json tonic.rate_hz.GPi"


def main(results, out):
    rates = {}
    with (results / "rates.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rates[r["model"], r["condition"], float(r["block"]), int(r["seed"])] = float(r["gpi_rate_hz"])
    seeds = sorted({k[3] for k in rates})
    draws = np.random.default_rng(SEED).integers(len(seeds), size=(REPLICATES, len(seeds)))
    gpi = json.loads((HERE.parent / "validation/protocol.json").read_text(encoding="utf-8"))["tonic"]["rate_hz"]["GPi"]
    animal_sd = float(np.std(gpi, ddof=1))
    rows = []
    for model in MODELS:
        level = lambda c, b: np.array([rates[model, c, b, s] for s in seeds])
        for name, diff in (("h1_minus_normal_full_block", level("h1", 1.0) - level("normal", 1.0)),
                           ("h2_minus_normal_full_block", level("h2", 1.0) - level("normal", 1.0)),
                           ("h2_minus_h1_full_block", level("h2", 1.0) - level("h1", 1.0)),
                           ("h1_minus_normal_no_block", level("h1", 0.0) - level("normal", 0.0)),
                           ("h2_minus_normal_no_block", level("h2", 0.0) - level("normal", 0.0))):
            lo, hi = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
            rows.append({"model": model, "measure": name, "seeds": len(seeds), "mean_hz": float(diff.mean()),
                         "low_hz": float(lo), "high_hz": float(hi), "seed_min_hz": float(diff.min()),
                         "seed_max_hz": float(diff.max()), "mean_over_animal_sd": float(diff.mean() / animal_sd)})
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"animal SD of GPi rate ({ANIMAL_SD_SOURCE}): {animal_sd:.2f} Hz")
    for r in rows:
        print(f"{r['model']:20s} {r['measure']:28s} {r['mean_hz']:+7.2f} [{r['low_hz']:+.2f}, {r['high_hz']:+.2f}]  /SD {r['mean_over_animal_sd']:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    main(args.results, args.results / "full_block_posthoc.csv")
