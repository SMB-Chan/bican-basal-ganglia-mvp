import unittest

import numpy as np

from identifiability.analyze import predict_nearest_centroid
from identifiability.run import CONTINUOUS, SIGNS, pair_stats
from spatial.preflight import neighbors


class IdentifiabilityTests(unittest.TestCase):
    def test_close_outputs_can_have_nonmatching_edges(self):
        fields = {**dict.fromkeys(CONTINUOUS, 0.0), **dict.fromkeys(SIGNS, 1),
                  "placement": 701, "membrane": "E", "hypothesis": "same", "stn_to_gpe_fraction": 0.5}
        rows = [{**fields, "graph_seed": 1}, {**fields, "graph_seed": 2}]
        edges = {(701, "same", 1): np.array([1, 2, 3]), (701, "same", 2): np.array([1, 4, 5])}
        protocol = {"membrane_conditions": ["E"], "pair_tolerances": {"rate_hz": 2, "isi_cv": 0.10, "early_hz": 1, "late_hz": 0.5}}
        pair, = pair_stats(rows, edges, protocol)
        self.assertEqual(pair["signs_equal"], 1)
        self.assertEqual(pair["continuous_close"], 1)
        self.assertAlmostEqual(pair["edge_jaccard"], 0.2)
        rows[1]["blocked_late_hz"] = -0.6
        pair, = pair_stats(rows, edges, protocol)
        self.assertEqual(pair["signs_equal"], 1)
        self.assertEqual(pair["continuous_close"], 0)

    def test_classification_ties_have_no_label_preference(self):
        classes, credit = predict_nearest_centroid(np.ones((3, 1)), np.array(["b", "c", "a"]), np.ones((1, 1)))
        np.testing.assert_array_equal(classes, ["a", "b", "c"])
        np.testing.assert_allclose(credit, [[1 / 3, 1 / 3, 1 / 3]])

    def test_spatial_neighbors_exclude_each_cell_itself(self):
        locations = np.array([[0., 0.], [1., 0.], [4., 0.]])
        np.testing.assert_array_equal(neighbors(locations, 1).ravel(), [1, 0, 1])
