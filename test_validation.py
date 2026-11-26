import json
import unittest

import numpy as np

from atlas import ROOT
from model import connect, population
from validation.run_validation import block_stn_to_gpe, condition_config, patchseq_population, pulse_metrics


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.protocol = json.loads((ROOT / "validation" / "protocol.json").read_text(encoding="utf-8"))
        cls.pop = population(cls.config, None, 101)

    def test_projection_mask_weights_and_channel_are_separable(self):
        specs = self.protocol["conditions"]
        masked = condition_config(self.config, specs["D_mask"])
        self.assertEqual(set(v for row in masked["projection_propensity"].values() for v in row.values()), {1.0})
        self.assertEqual(masked["matched_channel_gain"], 1.0)
        odds = condition_config(self.config, specs["E_odds"])
        self.assertEqual(odds["projection_propensity"], self.config["projection_propensity"])
        self.assertEqual(odds["matched_channel_gain"], 1.0)
        default = connect("D", self.pop, self.config, 101)
        old_d = connect("D", self.pop, condition_config(self.config, specs["E_channel"]), 101)
        np.testing.assert_array_equal(default.targets, old_d.targets)

    def test_patchseq_changes_only_two_time_constants(self):
        patched = patchseq_population(self.pop, {"groups": {"D1": {"median_tau_ms": 10.5}, "D2": {"median_tau_ms": 11.1}}})
        for attr in ("xyz", "channel", "bias", "jump", "sign"):
            self.assertIs(getattr(self.pop, attr), getattr(patched, attr))
        np.testing.assert_array_equal(patched.tau[:236], 10.5)
        np.testing.assert_array_equal(patched.tau[236:472], 11.1)
        np.testing.assert_array_equal(patched.tau[472:], self.pop.tau[472:])
        np.testing.assert_array_equal(self.pop.tau[:472], 18)

    def test_blockade_zeroes_only_stn_to_gpe(self):
        graph = connect("D", self.pop, self.config, 101)
        weights = block_stn_to_gpe(self.pop, graph)
        before = np.broadcast_to(graph.weights[:, None], weights.shape)
        expected = (self.pop.cell_type == self.pop.names.index("STN"))[:, None] & (
            self.pop.cell_type[graph.targets] == self.pop.names.index("GPe"))
        self.assertGreater(expected.sum(), 0)
        np.testing.assert_array_equal(weights[~expected], before[~expected])
        np.testing.assert_array_equal(weights[expected], 0)

    def test_response_windows_use_the_four_specified_features(self):
        response = np.zeros((2, 70))
        response[:, 3:13] = 3
        response[0, 14:54] = -2
        res = pulse_metrics(response, 1, self.protocol["stn_pulse"])
        self.assertEqual(res["pulse_features"], 4)
        self.assertEqual(res["late_attenuation_hz"], 2)
        response[1, 14:54] = -3
        self.assertEqual(pulse_metrics(response, 1, self.protocol["stn_pulse"])["pulse_features"], 3)
