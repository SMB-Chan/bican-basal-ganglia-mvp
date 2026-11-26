import unittest

import numpy as np

from heldout.metrics import drop_duplicate_trains, unit_metrics, windows_from_trials
from heldout.score import data_direction


SPEC = {"window_s": 1.0, "bin_ms": 1.0, "beta_band_hz": [8.0, 15.0], "reference_band_hz": [55.0, 150.0],
        "min_windows": 20, "min_isis": 20, "min_spikes": 50}


class HeldoutTests(unittest.TestCase):
    def test_windows_require_hold_and_no_early_error(self):
        w = windows_from_trials([0, 10, 20, 30], [2.0, 10.5, 23.0, 33.0], [np.nan, np.nan, 22.0, 40.0], 1.0)
        np.testing.assert_allclose(w, [[1.0, 2.0], [32.0, 33.0]])

    def test_index_detects_a_12_hz_rhythm_and_not_a_poisson_train(self):
        rng = np.random.default_rng(0)
        windows = np.column_stack((np.arange(60) * 2.0, np.arange(60) * 2.0 + 1.0))
        t = np.arange(0, 120, 0.001)
        poisson = t[rng.random(len(t)) < 0.04]
        rhythmic = t[rng.random(len(t)) < 0.04 * (1 + 0.9 * np.sin(2 * np.pi * 12 * t))]
        flat = unit_metrics(poisson, windows, SPEC)
        osc = unit_metrics(rhythmic, windows, SPEC)
        self.assertAlmostEqual(flat["beta_index"], 1.0, delta=0.2)
        self.assertGreater(osc["beta_index"], 1.5)
        self.assertAlmostEqual(flat["rate_hz"], 40.0, delta=2.0)
        self.assertAlmostEqual(flat["isi_cv"], 1.0, delta=0.1)

    def test_span_restriction_and_duplicates(self):
        windows = np.column_stack((np.arange(30.0), np.arange(30.0) + 1.0))
        spikes = np.linspace(10.05, 29.95, 400)
        self.assertIsNone(unit_metrics(spikes, windows, SPEC))
        self.assertEqual(unit_metrics(spikes, windows, {**SPEC, "min_windows": 10})["windows"], 18)
        self.assertEqual(drop_duplicate_trains([np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([3.0])]), [0, 2])

    def test_data_direction_rules(self):
        e = lambda lo, hi: {"ci_low": lo, "ci_high": hi, "change": (lo + hi) / 2}
        self.assertEqual(data_direction([e(1, 2), e(0.5, 3)]), "increase")
        self.assertEqual(data_direction([e(-2, -1), e(-3, -0.1)]), "decrease")
        self.assertEqual(data_direction([e(1, 2), e(-3, -1)]), "opposite")
        self.assertEqual(data_direction([e(1, 2), e(-1, 3)]), "no reliable change")
        self.assertEqual(data_direction([e(0.01, 0.03), e(0.01, 0.05)], margin=0.1), "no reliable change")
