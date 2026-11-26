"""Per-unit metrics in fixed windows, shared by the DANDI:000947 data and the model predictions.

Each unit contributes non-overlapping windows of equal length. From the spikes inside the windows we compute
the firing rate, the coefficient of variation of inter-spike intervals that fall inside a window, and the
8-15 Hz spectral index (mean periodogram power at 8-15 Hz divided by mean power at 55-150 Hz; Hann taper,
1 ms bins, averaged over windows). This is the unit-level index of Stage 4B computed on separate windows.
"""

import numpy as np


def windows_from_trials(start, cue, error, length_s):
    """Pre-cue windows [cue - length, cue) for trials with a long enough hold and no error before the cue."""
    start, cue, error = (np.asarray(x, dtype=float) for x in (start, cue, error))
    ok = np.isfinite(cue) & np.isfinite(start) & (cue - start >= length_s)
    ok &= ~(np.isfinite(error) & (error < cue))
    return np.column_stack((cue[ok] - length_s, cue[ok]))


def unit_metrics(spike_times, windows, spec, restrict_to_span=True):
    """Metrics for one unit. By default windows outside the unit's first-to-last spike span are dropped,
    because a sorted unit may be isolated for only part of a session. Simulated cells pass False."""
    spikes = np.sort(np.asarray(spike_times, dtype=float))
    if len(spikes) < 2 or len(windows) == 0:
        return None
    windows = np.asarray(windows, dtype=float)
    if restrict_to_span:
        windows = windows[(windows[:, 0] >= spikes[0]) & (windows[:, 1] <= spikes[-1])]
    if len(windows) < spec["min_windows"]:
        return None
    length = float(spec["window_s"])
    lo = np.searchsorted(spikes, windows[:, 0], side="left")
    hi = np.searchsorted(spikes, windows[:, 1], side="left")
    counts = hi - lo
    total = int(counts.sum())
    isis = np.concatenate([np.diff(spikes[a:b]) for a, b in zip(lo, hi) if b - a >= 2] or [np.array([])])
    cv = float(isis.std() / isis.mean()) if len(isis) >= spec["min_isis"] and isis.mean() > 0 else float("nan")

    bin_s = spec["bin_ms"] / 1000.0
    n_bins = int(round(length / bin_s))
    freqs = np.fft.rfftfreq(n_bins, d=bin_s)
    taper = np.hanning(n_bins)
    psd = np.zeros(len(freqs))
    for (w0, _), a, b in zip(windows, lo, hi):
        train = np.bincount(np.minimum(((spikes[a:b] - w0) / bin_s).astype(int), n_bins - 1), minlength=n_bins) / bin_s
        psd += np.abs(np.fft.rfft((train - train.mean()) * taper)) ** 2
    beta = (freqs >= spec["beta_band_hz"][0]) & (freqs <= spec["beta_band_hz"][1])
    ref = (freqs >= spec["reference_band_hz"][0]) & (freqs <= spec["reference_band_hz"][1])
    index = float(psd[beta].mean() / psd[ref].mean()) if total >= spec["min_spikes"] and psd[ref].mean() > 0 else float("nan")
    return {"windows": int(len(windows)), "spikes": total, "rate_hz": total / (len(windows) * length),
            "isi_cv": cv, "isis": int(len(isis)), "beta_index": index}


def drop_duplicate_trains(trains):
    """Indices of spike trains to keep: the first of any group of identical trains."""
    seen, keep = set(), []
    for i, train in enumerate(trains):
        key = np.asarray(train, dtype=float).tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep
