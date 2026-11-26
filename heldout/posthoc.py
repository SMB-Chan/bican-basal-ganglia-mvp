"""Post-hoc checks after scoring (not part of the pre-registered test).

For each GPi unit: the whole-span firing rate (spikes between the first and last spike divided by that span)
and the firing rate in the 1 s after movement onset, so the pre-cue decrease can be compared with the session
as a whole and with movement periods.
"""

import argparse
import csv
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import remfile

from heldout.extract import decode
from heldout.inventory import API, DANDISET, VERSION
from heldout.metrics import drop_duplicate_trains
from run import write_csv


HERE = Path(__file__).resolve().parent


def file_rows(job):
    path, asset_id, subject, state = job
    with h5py.File(remfile.File(f"{API}/assets/{asset_id}/download/"), "r") as f:
        units = f["units"]
        locations = decode(units["location"][:])
        ends = units["spike_times_index"][:]
        starts = np.concatenate(([0], ends[:-1]))
        gpi = [i for i, loc in enumerate(locations) if loc.strip().lower() == "gpi"]
        trains = [np.sort(units["spike_times"][starts[i]:ends[i]]) for i in gpi]
        move = f["intervals/trials/movement_start_time"][:] if "intervals/trials" in f and "movement_start_time" in f["intervals/trials"] else np.array([])
    keep = drop_duplicate_trains(trains)
    rows = []
    for k in keep:
        s = trains[k]
        if len(s) < 2:
            continue
        move_ok = move[np.isfinite(move) & (move >= s[0]) & (move + 1.0 <= s[-1])]
        move_rate = (float(np.mean([np.searchsorted(s, m + 1.0) - np.searchsorted(s, m) for m in move_ok])) if len(move_ok) >= 20 else float("nan"))
        rows.append({"subject": subject, "state": state, "session": path, "span_s": float(s[-1] - s[0]),
                     "whole_span_rate_hz": (len(s) - 1) / float(s[-1] - s[0]), "post_movement_rate_hz": move_rate,
                     "movement_trials": int(len(move_ok))})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inventory = list(csv.DictReader(open(HERE / "inventory.csv", encoding="utf-8")))
    url = f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=400"
    assets = {a["path"]: a["asset_id"] for a in json.load(urllib.request.urlopen(url, timeout=60))["results"]}
    jobs = [(r["path"], assets[r["path"]], r["subject"], r["state"]) for r in inventory
            if sum(int(r[c] or 0) for c in r if c.lower() == "units_gpi") > 0]
    with ThreadPoolExecutor(6) as pool:
        rows = [row for chunk in pool.map(file_rows, jobs) for row in chunk]
    write_csv(args.out, rows)
