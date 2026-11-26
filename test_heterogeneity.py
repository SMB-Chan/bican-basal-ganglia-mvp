import json
import unittest
from dataclasses import replace

import numpy as np

from atlas import ROOT
from heterogeneity.patchseq import Cell
from heterogeneity.run import donor_validation, measured_parameters, passive_rheobase
from model import Graph, Population, population, simulate


class HeterogeneityTests(unittest.TestCase):
    def test_shuffle_preserves_marginals_and_only_modifies_spns(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        pop = population(config, None, 307)
        cells = [Cell(kind, f"d{i}", "Macaca mulatta", 5 + i, 20 + i, 40 + 3 * i, 150)
                 for kind in ("D1", "D2") for i in range(10)]
        joint = measured_parameters(pop, cells, config["simulation"], "joint", 307)
        shuffled = measured_parameters(pop, cells, config["simulation"], "shuffled", 307)
        np.testing.assert_array_equal(joint.tau, shuffled.tau)
        for i in (0, 1):
            mask = pop.cell_type == i
            for name in ("threshold_mv", "cue_gain"):
                paired = getattr(joint, name)[mask]
                permuted = getattr(shuffled, name)[mask]
                np.testing.assert_array_equal(np.sort(paired), np.sort(permuted))
                self.assertFalse(np.array_equal(paired, permuted))
        spn = np.isin(pop.cell_type, (0, 1))
        np.testing.assert_array_equal(joint.tau[~spn], pop.tau[~spn])
        np.testing.assert_array_equal(joint.bias[~spn], pop.bias[~spn])
        np.testing.assert_array_equal(joint.sign, pop.sign)
        self.assertIs(joint.xyz, pop.xyz)

    def test_held_out_response_does_not_fit_its_own_prediction(self):
        cells = [Cell("D1", "a", "Macaca mulatta", 8, 30, 100, 300),
                 Cell("D1", "b", "Macaca mulatta", 8, 30, 100, 150)]
        self.assertEqual(passive_rheobase(30, 100), 300)
        first = next(row for row in donor_validation(cells) if row["fold"] == 1 and row["model"] == "calibrated_cell")
        cells[0] = replace(cells[0], rheobase_pa=400)
        second = next(row for row in donor_validation(cells) if row["fold"] == 1 and row["model"] == "calibrated_cell")
        self.assertEqual(first["signed_error_pa"], -150)
        self.assertEqual(second["signed_error_pa"], -250)

    def test_cell_threshold_and_cue_gain_affect_simulation(self):
        pop = Population(("group",), np.zeros(2, dtype=int), np.zeros(2, dtype=int), np.zeros((2, 3)),
                         np.ones(2), np.ones(2), np.zeros(2), np.ones(2))
        graph = Graph(np.array([], dtype=int), np.full((2, 1), -1), np.zeros(2), np.zeros((2, 1), dtype=int),
                      np.empty((0, 1)))
        sim = dict(dt_ms=1, baseline_ms=1, cue_ms=2, refractory_ms=1, rest_mv=-2, reset_mv=-2, threshold_mv=0)
        initial = np.full(2, -2.)
        self.assertFalse(simulate(pop, graph, sim, np.zeros((3, 2)), initial, np.ones(2)).any())
        variable = replace(pop, threshold_mv=np.array([-1.5, 0.]), cue_gain=np.array([1., 3.]))
        spikes = simulate(variable, graph, sim, np.zeros((3, 2)), initial, np.ones(2))
        np.testing.assert_array_equal(np.argwhere(spikes), [[1, 0], [2, 1]])
