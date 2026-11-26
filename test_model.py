import json
import unittest

import numpy as np

from atlas import ROOT, load_atlas
from model import Graph, Population, connect, edge_matrix, graph_statistics, pathway_matrix, population, simulate
from run import cue_input, rates, score_trial, trial_plan, trial_stimuli


class WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.pop = population(cls.config, None, 11)

    def test_counts_out_degree_sign_and_geometry_based_delays_are_shared(self):
        for condition in "ABCD":
            with self.subTest(condition=condition):
                graph = connect(condition, self.pop, self.config, 11)
                src = graph.sources
                targets = graph.targets[src]
                n = len(self.pop.cell_type)
                self.assertEqual(n, 1000)
                self.assertEqual(targets.shape, (820, 24))
                self.assertTrue((targets != src[:, None]).all())
                self.assertTrue((np.diff(targets, axis=1) > 0).all())
                self.assertTrue((graph.targets[self.pop.cell_type == self.pop.names.index("GPi")] == -1).all())
                np.testing.assert_array_equal(np.sign(graph.weights), self.pop.sign)
                lengths = np.linalg.norm(self.pop.xyz[src, None] - self.pop.xyz[targets], axis=2)
                np.testing.assert_allclose(graph.lengths, lengths)
                expected = np.ceil(1.0 + lengths / 0.9)
                np.testing.assert_array_equal(graph.delay[src], expected)
                stats = graph_statistics(self.pop, graph, self.config)
                self.assertEqual(stats["excitatory_edges"], 2880)
                self.assertEqual(stats["inhibitory_edges"], 16800)
                self.assertEqual(edge_matrix(self.pop, graph).sum(), 19680)

    def test_d_is_directed_and_obeys_projection_mask(self):
        graph = connect("D", self.pop, self.config, 11)
        np.testing.assert_array_equal(edge_matrix(self.pop, graph)[pathway_matrix(self.pop, self.config) == 0], 0)
        gpe = self.pop.names.index("GPe")
        gpi = self.pop.names.index("GPi")
        matrix = edge_matrix(self.pop, graph)
        self.assertGreater(matrix[gpe, gpi], 0)
        self.assertEqual(matrix[gpi, gpe], 0)

    def test_graph_and_trials_are_deterministic(self):
        first = connect("D", self.pop, self.config, 11)
        second = connect("D", population(self.config, None, 11), self.config, 11)
        np.testing.assert_array_equal(first.targets, second.targets)
        np.testing.assert_array_equal(first.delay, second.delay)
        self.assertFalse(np.array_equal(first.targets, connect("D", self.pop, self.config, 23).targets))
        sim = self.config["simulation"]
        plan = trial_plan(sim, 11)
        self.assertEqual(plan.count((0.0, None)), 4)
        for delta in sim["cue_deltas_mv"]:
            self.assertEqual(plan.count((delta, 0)), plan.count((delta, 1)))
        self.assertEqual(plan, trial_plan(sim, 11))
        np.testing.assert_array_equal(trial_stimuli(self.pop, sim, 11, 4)[0], trial_stimuli(self.pop, sim, 11, 4)[0])
        difference = cue_input(self.pop, sim, 5.0, 0) - cue_input(self.pop, sim, 0.0, None)
        expected = 5.0 * (np.isin(self.pop.cell_type, [0, 1]) & (self.pop.channel == 0))
        np.testing.assert_array_equal(difference, expected)

    @unittest.skipUnless((ROOT / "atlas_data" / "rhesus_annotations_160.nii.gz").exists(), "Atlas files absent")
    def test_official_volume_counts_and_sampled_positions(self):
        atlas, info = load_atlas(ROOT / "atlas_data")
        self.assertEqual({name: len(points) for name, points in atlas.items()},
                         {"STRd": 372123, "GPe": 42309, "STN": 3700, "GPi": 20363})
        pop = population(self.config, atlas, 11)
        np.testing.assert_array_equal(pop.channel, self.pop.channel)
        for region, points in atlas.items():
            table = {tuple(point) for point in points}
            types = [i for i, spec in enumerate(self.config["groups"].values()) if spec["region"] == region]
            selected = pop.xyz[np.isin(pop.cell_type, types)]
            self.assertTrue(all(tuple(point) in table for point in selected))
            self.assertTrue((selected[:, 0] > info["midline_ras_x_mm"]).all())


class DynamicsTests(unittest.TestCase):
    def test_direction_sign_and_two_tick_delivery(self):
        pop = Population(("pre", "post"), np.array([0, 1]), np.array([0, 0]), np.zeros((2, 3)),
                         np.array([1, -1]), np.ones(2), np.zeros(2), np.ones(2))
        graph_args = dict(sources=np.array([0]), targets=np.array([[1], [-1]]), delay=np.array([[2], [0]]), lengths=np.ones((1, 1)))
        sim = dict(dt_ms=1, baseline_ms=2, cue_ms=4, refractory_ms=2, rest_mv=-1, reset_mv=-1, threshold_mv=0)
        noise = np.zeros((6, 2))
        noise[0, 0] = 2
        initial = np.array([-1., -1.])
        excitatory = Graph(**graph_args, weights=np.array([2., 0.]))
        inhibitory = Graph(**graph_args, weights=np.array([-2., 0.]))
        spikes = simulate(pop, excitatory, sim, noise, initial, np.zeros(2))
        np.testing.assert_array_equal(np.argwhere(spikes), [[0, 0], [2, 1]])
        noise[2, 1] = 1.2
        spikes = simulate(pop, inhibitory, sim, noise, initial, np.zeros(2))
        np.testing.assert_array_equal(np.argwhere(spikes), [[0, 0]])

    def test_refractory_and_rate_window(self):
        pop = Population(("cell",), np.array([0]), np.array([0]), np.zeros((1, 3)),
                         np.array([1]), np.ones(1), np.zeros(1), np.ones(1))
        graph = Graph(np.array([], dtype=int), np.array([[-1]]), np.array([0.]), np.array([[0]]), np.empty((0, 1)))
        sim = dict(dt_ms=1, baseline_ms=4, cue_ms=3, refractory_ms=2, rest_mv=-1, reset_mv=-1, threshold_mv=0,
                   baseline_readout_ms=2, cue_readout_ms=2)
        spikes = simulate(pop, graph, sim, np.full((7, 1), 2.0), np.array([-1.]), np.zeros(1))
        np.testing.assert_array_equal(np.flatnonzero(spikes[:, 0]), [0, 3, 6])
        for phase in rates(spikes, sim):
            np.testing.assert_array_equal(phase, [500.0])

    def test_task_scores_ties_and_equal_cues(self):
        self.assertEqual(score_trial([12, 19], 0), (0, 1.0, 7))
        self.assertEqual(score_trial([12, 19], 1), (0, 0.0, -7))
        self.assertEqual(score_trial([19, 19], 1), (None, 0.5, 0))
        self.assertEqual(score_trial([19, 12], None), (1, None, None))


if __name__ == "__main__":
    unittest.main()
