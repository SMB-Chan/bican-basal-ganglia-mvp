"""Score the frozen model predictions against the DANDI:000947 GPi units with the pre-registered rules."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import figure_style, read_csv, save
from run import write_csv


HERE = Path(__file__).resolve().parent
METRICS = ("rate_hz", "isi_cv", "beta_index")


def finite(rows, metric):
    return [r for r in rows if r.get(metric) not in (None, "") and np.isfinite(float(r[metric]))]


def monkey_effect(units, metric, rng, replicates):
    """Median after minus median before, with a session-cluster bootstrap within each MPTP state."""
    by_state = {}
    for state in ("pre", "post"):
        sessions = {}
        for r in finite([u for u in units if u["state"] == state], metric):
            sessions.setdefault(r["session"], []).append(float(r[metric]))
        by_state[state] = list(sessions.values())
    point = float(np.median(np.concatenate(by_state["post"])) - np.median(np.concatenate(by_state["pre"])))
    boots = np.empty(replicates)
    for b in range(replicates):
        med = {}
        for state, groups in by_state.items():
            pick = rng.integers(0, len(groups), size=len(groups))
            med[state] = np.median(np.concatenate([groups[i] for i in pick]))
        boots[b] = med["post"] - med["pre"]
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"change": point, "ci_low": float(low), "ci_high": float(high),
            "units_pre": sum(map(len, by_state["pre"])), "units_post": sum(map(len, by_state["post"])),
            "sessions_pre": len(by_state["pre"]), "sessions_post": len(by_state["post"]),
            "median_pre": float(np.median(np.concatenate(by_state["pre"]))), "median_post": float(np.median(np.concatenate(by_state["post"])))}


def data_direction(effects, margin=0.0):
    above = [e["ci_low"] > 0 and e["change"] >= margin for e in effects]
    below = [e["ci_high"] < 0 and e["change"] <= -margin for e in effects]
    if all(above):
        return "increase"
    if all(below):
        return "decrease"
    if any(above) and any(below):
        return "opposite"
    return "no reliable change"


def score(units, predictions, protocol, selection):
    stats = protocol["statistics"]
    rng = np.random.default_rng(stats["bootstrap_seed"])
    included = [u for u in units if u["included"] == "1"]
    if selection == "sensitivity":
        included = [u for u in included if "excellent" in u["quality"].lower() or "good" in u["quality"].lower()]
    effects, data = [], {}
    for metric in METRICS:
        per_monkey = []
        for subject in protocol["data"]["subjects"]:
            e = monkey_effect([u for u in included if u["subject"] == subject], metric, rng, stats["replicates"])
            effects.append({"selection": selection, "metric": metric, "subject": subject, **e})
            per_monkey.append(e)
        data[metric] = {"direction": data_direction(per_monkey, protocol["directions"]["minimum_effect"][metric]),
                        "range_low": min(e["ci_low"] for e in per_monkey), "range_high": max(e["ci_high"] for e in per_monkey)}
    scores = []
    for model in protocol["model_predictions"]["models"]:
        total = 0
        for metric in METRICS:
            p = next(r for r in predictions if r["model"] == model and r["metric"] == metric)
            agree = int(data[metric]["direction"] != "opposite" and p["direction"] == data[metric]["direction"])
            within = int(data[metric]["range_low"] <= float(p["change"]) <= data[metric]["range_high"])
            total += agree
            scores.append({"selection": selection, "model": model, "metric": metric, "model_direction": p["direction"],
                           "model_change": float(p["change"]), "data_direction": data[metric]["direction"],
                           "direction_agrees": agree, "magnitude_within_data_range": within})
        scores.append({"selection": selection, "model": model, "metric": "TOTAL", "direction_agrees": total})
    return effects, scores, data


def make_figure(out, effects, predictions):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), layout="constrained")
    fig.suptitle("GPi before vs after MPTP (DANDI:000947) and frozen model predictions", fontsize=14, fontweight="bold")
    labels = {"rate_hz": "Firing rate change (Hz)", "isi_cv": "ISI CV change", "beta_index": "8–15 Hz index change"}
    colors = {"Gaia": "#2e6f9e", "Isis": "#c0602e", "baseline_lif": "#66788a", "selected_biophysics": "#937638", "selected_rebound": "#207c74"}
    for ax, metric in zip(axes, METRICS):
        rows = [e for e in effects if e["metric"] == metric and e["selection"] == "primary"]
        items = [(r["subject"], r["change"], r["ci_low"], r["ci_high"]) for r in rows]
        items += [(p["model"], float(p["change"]), float(p["ci_low"]), float(p["ci_high"])) for p in predictions if p["metric"] == metric]
        for y, (name, mid, lo, hi) in enumerate(items):
            ax.errorbar(mid, y, xerr=[[mid - lo], [hi - mid]], fmt="o", color=colors[name], capsize=3)
        ax.axvline(0, color="#929ca9", lw=0.8, linestyle="--")
        ax.set_yticks(range(len(items)), [n.replace("selected_", "") for n, *_ in items])
        ax.invert_yaxis()
        ax.set(title=labels[metric], xlabel="After / depleted minus before / normal")
        ax.grid(axis="x", alpha=0.2)
    save(fig, out / "heldout_gpi.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    units, predictions = read_csv(args.units), read_csv(args.predictions)
    all_effects, all_scores, verdict = [], [], {}
    for selection in ("primary", "sensitivity"):
        effects, scores, data = score(units, predictions, protocol, selection)
        all_effects += effects
        all_scores += scores
        verdict[selection] = {"data": data, "totals": {s["model"]: s["direction_agrees"] for s in scores if s["metric"] == "TOTAL"}}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "data_effects.csv", all_effects)
    write_csv(args.out / "scores.csv", all_scores)
    (args.out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figure_style()
    make_figure(args.out, all_effects, predictions)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
