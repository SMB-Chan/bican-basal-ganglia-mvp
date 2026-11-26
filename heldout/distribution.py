"""Post-hoc description of the extracted GPi rates (not part of the pre-registered test).

Prints rate quartiles and the share of units below 20 Hz and above 60 Hz per monkey and MPTP state, session-median
rates by recording period, and Isis rates by sort-quality label.
"""

import collections
import csv
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    units = [r for r in csv.DictReader(open(HERE / "results/units.csv", encoding="utf-8")) if r["included"] == "1"]
    for subject in ("Gaia", "Isis"):
        for state in ("pre", "post"):
            rates = sorted(float(u["rate_hz"]) for u in units if u["subject"] == subject and u["state"] == state)
            n = len(rates)
            print(f"{subject} {state}: n={n}, quartiles {rates[n // 4]:.1f} / {rates[n // 2]:.1f} / {rates[3 * n // 4]:.1f} Hz, "
                  f"<20 Hz {100 * sum(r < 20 for r in rates) / n:.0f}%, >60 Hz {100 * sum(r > 60 for r in rates) / n:.0f}%")
    sessions = collections.defaultdict(list)
    for u in units:
        sessions[u["subject"], u["state"], u["session"]].append(float(u["rate_hz"]))
    for subject in ("Gaia", "Isis"):
        for state in ("pre", "post"):
            items = sorted((re.search(r"-(\d{6})-", k[2]).group(1), statistics.median(v))
                           for k, v in sessions.items() if k[:2] == (subject, state))
            third = max(1, len(items) // 3)
            parts = [items[:third], items[third:-third], items[-third:]]
            meds = [statistics.median(m for _, m in p) if p else float("nan") for p in parts]
            print(f"{subject} {state}: {len(items)} sessions {items[0][0]}..{items[-1][0]}, session-median rate by thirds "
                  + " / ".join(f"{m:.1f}" for m in meds))
    for state in ("pre", "post"):
        by_label = collections.defaultdict(list)
        for u in units:
            if u["subject"] == "Isis" and u["state"] == state:
                by_label[u["quality"] or "(none)"].append(float(u["rate_hz"]))
        print(f"Isis {state} by label:", {k: (len(v), round(statistics.median(v), 1)) for k, v in by_label.items() if len(v) >= 5})
