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


class DepletionTests(unittest.TestCase):
    def test_conditions_switch_only_the_named_components(self):
        import json
        from atlas import ROOT
        from depletion.run import build_condition
        from model import connect, population
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        depletion = json.loads((ROOT / "parkinsonian/protocol.json").read_text(encoding="utf-8"))["dopamine_depletion"]
        pop = population(config, None, 11)
        graph = connect("D", pop, {**config, "matched_channel_gain": 1.0}, 11)
        base_w = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
        normal, w0 = build_condition(pop, graph, depletion, {"d1": False, "d2": False, "gpe_stn": False})
        np.testing.assert_array_equal(normal.bias, pop.bias)
        np.testing.assert_array_equal(w0, base_w)
        d1 = pop.cell_type == pop.names.index("D1")
        gpi = pop.cell_type == pop.names.index("GPi")
        only, w1 = build_condition(pop, graph, depletion, {"d1": True, "d2": False, "gpe_stn": False, "gpi_bias_mv": -3.0})
        np.testing.assert_allclose(only.bias[d1], pop.bias[d1] + depletion["d1_bias_shift_mv"])
        np.testing.assert_allclose(only.bias[gpi], pop.bias[gpi] - 3.0)
        np.testing.assert_array_equal(w1, base_w)
        _, w2 = build_condition(pop, graph, depletion, {"d1": False, "d2": False, "gpe_stn": False, "gpi_input_gain": {"STN": 0.5}})
        post = np.where(graph.targets >= 0, pop.cell_type[np.maximum(graph.targets, 0)], -1)
        edges = (pop.cell_type[:, None] == pop.names.index("STN")) & (post == pop.names.index("GPi"))
        np.testing.assert_allclose(w2[edges], 0.5 * base_w[edges])
        np.testing.assert_array_equal(w2[~edges], base_w[~edges])
