"""Inventory of DANDI:000947 without reading neural values.

For every NWB file: subject, pre/post-MPTP label, unit counts per recorded location and per sort-quality label,
trial count, and the distribution of gaps between consecutive trials (behavioral timestamps only). Spike times are
never read. Files are opened remotely with HTTP range requests (remfile), so nothing large is downloaded.
"""

import argparse
import json
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import remfile

from run import write_csv


DANDISET, VERSION = "000947", "0.240510.2211"
API = "https://api.dandiarchive.org/api"


def assets():
    url = f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=400"
    return json.load(urllib.request.urlopen(url, timeout=60))["results"]


def decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def describe(asset):
    path = asset["path"]
    row = {"path": path, "subject": path.split("/")[0].removeprefix("sub-"),
           "state": re.search(r"ses-(pre|post)-MPTP", path).group(1), "bytes": asset["size"]}
    with h5py.File(remfile.File(f"{API}/assets/{asset['asset_id']}/download/"), "r") as f:
        units = f["units"] if "units" in f else None
        locations = Counter(decode(units["location"][:])) if units is not None and "location" in units else Counter()
        quality = Counter(decode(units["unit_quality_post_sorting"][:])) if units is not None and "unit_quality_post_sorting" in units else Counter()
        row.update({f"units_{k}": v for k, v in sorted(locations.items())})
        row["unit_quality_labels"] = json.dumps(dict(quality), ensure_ascii=False)
        if "intervals/trials" in f:
            t = f["intervals/trials"]
            start, stop = t["start_time"][:], t["stop_time"][:]
            gaps = start[1:] - stop[:-1] if len(start) > 1 else np.array([])
            row.update({"trials": len(start), "recording_span_s": float(stop.max() - start.min()) if len(start) else 0.0,
                        "median_trial_s": float(np.median(stop - start)) if len(start) else None,
                        "median_gap_s": float(np.median(gaps)) if len(gaps) else None,
                        "total_gap_s": float(gaps[gaps > 0].sum()) if len(gaps) else 0.0})
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    items = assets()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(describe, items))
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.out, [{k: r.get(k, 0 if k.startswith("units_") else None) for k in columns} for r in rows])
