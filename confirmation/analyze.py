"""Score the Stage 4A factorial on fresh seeds against the pre-registered criteria in protocol.json."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import read_csv
from rebound.run import select_rebound_variant
from run import write_csv


HERE = Path(__file__).resolve().parent
OUTCOMES = ("joint_normal_pass", "kita_strict_pass", "m1_all_causal_pass", "nambu_stn_block_pass",
            "tachibana_gaba_block_pass", "isolated_rebound_pass", "tonic_rate_mae_hz", "tonic_cv_mae",
            "stn_block_late_drop_hz", "blocked_late_hz")


def by_variant(rows):
    seeds = sorted({int(r["seed"]) for r in rows})
    variants = sorted({r["variant"] for r in rows})
    index = {(r["variant"], int(r["seed"])): r for r in rows}
    return seeds, variants, {(v, k): np.array([float(index[v, s][k]) for s in seeds]) for v in variants for k in OUTCOMES}


def interval(values, draws):
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def score(rows, protocol):
    seeds, variants, data = by_variant(rows)
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    draws = rng.integers(0, len(seeds), size=(protocol["bootstrap_replicates"], len(seeds)))
    summary = []
    for v in variants:
        for k in OUTCOMES:
            mean, low, high = interval(data[v, k], draws)
            summary.append({"variant": v, "metric": k, "seeds": len(seeds), "mean": mean, "ci_low": low, "ci_high": high})
    frozen = protocol["frozen_variant"]
    rates = {v: data[v, "joint_normal_pass"].mean() for v in variants}
    ranking = sorted(variants, key=lambda v: -rates[v])
    paired = []
    for v in variants:
        if v == frozen:
            continue
        mean, low, high = interval(data[frozen, "joint_normal_pass"] - data[v, "joint_normal_pass"], draws)
        paired.append({"frozen": frozen, "alternative": v, "metric": "joint_normal_pass", "seeds": len(seeds),
                       "mean_difference": mean, "ci_low": low, "ci_high": high})
    best_alternative = max((v for v in variants if v != frozen), key=lambda v: rates[v])
    c2 = next(p for p in paired if p["alternative"] == best_alternative)
    reselected = select_rebound_variant(rows, {})["variant"]
    frozen_mean, frozen_low, frozen_high = interval(data[frozen, "joint_normal_pass"], draws)
    original = protocol["original_joint_normal_pass"][0] / protocol["original_joint_normal_pass"][1]
    verdict = {
        "seeds": len(seeds),
        "frozen_variant": frozen,
        "frozen_joint_pass": {"mean": frozen_mean, "ci_low": frozen_low, "ci_high": frozen_high},
        "frozen_rank": ranking.index(frozen) + 1,
        "ranking_by_joint_pass": [{"variant": v, "joint_pass": float(rates[v])} for v in ranking],
        "C1_selection_stable": {"reselected": reselected, "pass": reselected == frozen},
        "C2_superiority": {"best_alternative": best_alternative, **{k: c2[k] for k in ("mean_difference", "ci_low", "ci_high")},
                           "pass": c2["ci_low"] > 0},
        "C3_no_shrinkage": {"original_rate": original, "pass": frozen_low <= original <= frozen_high},
    }
    return summary, paired, verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Output folder of `python -m rebound.run --seeds ...`")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if sorted(manifest["seeds"]) != sorted(protocol["seeds"]):
        raise ValueError("The run does not use the pre-registered seeds")
    summary, paired, verdict = score(read_csv(args.run / "evaluations.csv"), protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "paired.csv", paired)
    (args.out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
