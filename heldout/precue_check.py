"""Pre-cue hold durations in DANDI:000947 from behavioral timestamps only (no spike times are read).

Samples 5 GPi-containing files per subject and MPTP state (random seed 11) and reports the distribution of
cue_onset_time - start_time per trial.
"""

import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import remfile

from heldout.inventory import API, DANDISET, VERSION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    rows = list(csv.DictReader(open(parser.parse_args().inventory, encoding="utf-8")))
    url = f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=400"
    assets = {a["path"]: a for a in json.load(urllib.request.urlopen(url, timeout=60))["results"]}
    rng = random.Random(11)
    for subject in ("Gaia", "Isis"):
        for state in ("pre", "post"):
            candidates = [r for r in rows if r["subject"] == subject and r["state"] == state
                          and any(int(r[c] or 0) > 0 for c in r if c.lower() == "units_gpi")]
            durations, trials = [], []
            for r in rng.sample(candidates, min(5, len(candidates))):
                with h5py.File(remfile.File(f"{API}/assets/{assets[r['path']]['asset_id']}/download/"), "r") as f:
                    t = f["intervals/trials"]
                    start, cue = t["start_time"][:], t["cue_onset_time"][:]
                    ok = np.isfinite(cue)
                    durations.extend((cue[ok] - start[ok]).tolist())
                    trials.append(int(ok.sum()))
            q = np.quantile(durations, [0.5, 0.1, 0.9])
            print(f"{subject} {state}: files {len(trials)}, trials/file {trials}, start->cue s median {q[0]:.2f}, 10% {q[1]:.2f}, 90% {q[2]:.2f}")
