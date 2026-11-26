"""Read GPi units from DANDI:000947 and compute pre-cue window metrics. Run only after the protocol is pushed.

Uses the top-level NWB Units table (the table the dataset's own demo notebook uses). The undocumented
processing/ecephys/units table present in some files is not used.
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

from heldout.inventory import API, DANDISET, VERSION
from heldout.metrics import drop_duplicate_trains, unit_metrics, windows_from_trials
from run import write_csv


HERE = Path(__file__).resolve().parent


def decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def file_rows(job):
    path, asset_id, subject, state, spec = job
    with h5py.File(remfile.File(f"{API}/assets/{asset_id}/download/"), "r") as f:
        units = f["units"]
        locations = decode(units["location"][:])
        quality = decode(units["unit_quality_post_sorting"][:]) if "unit_quality_post_sorting" in units else [""] * len(locations)
        names = decode(units["unit_name"][:]) if "unit_name" in units else [str(i) for i in range(len(locations))]
        ends = units["spike_times_index"][:]
        starts = np.concatenate(([0], ends[:-1]))
        gpi = [i for i, loc in enumerate(locations) if loc.strip().lower() == "gpi"]
        trains = [units["spike_times"][starts[i]:ends[i]] for i in gpi]
        if "intervals/trials" in f and "cue_onset_time" in f["intervals/trials"]:
            t = f["intervals/trials"]
            error = t["error_onset_time"][:] if "error_onset_time" in t else np.full(len(t["start_time"]), np.nan)
            windows = windows_from_trials(t["start_time"][:], t["cue_onset_time"][:], error, spec["window_s"])
        else:
            # No trial table: no pre-cue windows, so the file's units cannot meet the window criterion.
            windows = np.empty((0, 2))
    keep = drop_duplicate_trains(trains)
    rows = []
    for k, i in enumerate(gpi):
        row = {"subject": subject, "state": state, "session": path, "unit": names[i], "quality": quality[i],
               "duplicate_dropped": int(k not in keep), "trial_windows": len(windows)}
        m = unit_metrics(trains[k], windows, spec) if k in keep else None
        rows.append({**row, "included": int(m is not None), **(m or {})})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    inventory = list(csv.DictReader(open(HERE / "inventory.csv", encoding="utf-8")))
    url = f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=400"
    assets = {a["path"]: a["asset_id"] for a in json.load(urllib.request.urlopen(url, timeout=60))["results"]}
    jobs = [(r["path"], assets[r["path"]], r["subject"], r["state"], protocol["metrics"]) for r in inventory
            if sum(int(r[c] or 0) for c in r if c.lower() == "units_gpi") > 0]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [row for chunk in pool.map(file_rows, jobs) for row in chunk]
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.out, [{k: r.get(k) for k in columns} for r in rows])
    print(f"{len(jobs)} files, {len(rows)} GPi units, {sum(r['included'] for r in rows)} included")
