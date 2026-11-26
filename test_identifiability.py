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


class StructureAndCompositionTests(unittest.TestCase):
    def test_block_distance_counts_edges_moved_between_group_pairs(self):
        from identifiability.structure import pair_structure
        row = {**dict.fromkeys(CONTINUOUS, 0.0), "placement": "1", "membrane": "E"}
        rows = [{**row, "hypothesis": "h", "graph_seed": "1"}, {**row, "hypothesis": "h", "graph_seed": "2", "GPe_rate_hz": 2.0}]
        blocks = {(1, "h", 1): np.array([[3, 1], [0, 0]]), (1, "h", 2): np.array([[2, 2], [0, 0]])}
        pairs = [{"placement": "1", "membrane": "E", "left_hypothesis": "h", "right_hypothesis": "h",
                  "left_graph_seed": "1", "right_graph_seed": "2", "continuous_close": "1", "edge_jaccard": "0.5"}]
        result, = pair_structure(rows, pairs, blocks)
        self.assertAlmostEqual(result["block_distance"], 0.25)
        self.assertGreater(result["functional_distance"], 0)

    def test_dispersion_index_separates_patches_from_random_labels(self):
        from spatial.composition import compartment, dispersion_index, permute_within_strata
        rng = np.random.default_rng(3)
        positions = rng.uniform(0, 4, size=(4000, 2))
        random_labels = rng.random(4000) < 0.5
        patches = positions[:, 0] < 2
        self.assertLess(dispersion_index(positions, random_labels, 0.5, (0, 0), 10)[0], 1.5)
        self.assertGreater(dispersion_index(positions, patches, 0.5, (0, 0), 10)[0], 20)
        strata = np.where(patches, "striosome", "matrix")
        shuffled = permute_within_strata(random_labels, strata, rng)
        for name in ("striosome", "matrix"):
            self.assertEqual(int(random_labels[strata == name].sum()), int(shuffled[strata == name].sum()))
        self.assertEqual(compartment("STRd D1 Striosome SPN"), "striosome")
        self.assertEqual(compartment("STRd D2 Matrix SPN"), "matrix")
        self.assertIsNone(compartment("STR D1D2 eSPN"))

    def test_rank_averages_ties_and_edge_test_removes_stratum_confounding(self):
        from identifiability.structure import edge_level_test, rank
        np.testing.assert_allclose(rank([10, 20, 20, 30]), [0.0, 1.5, 1.5, 3.0])
        rng = np.random.default_rng(19)
        table = []
        for pair_name, base_j, close_count in (("a|a", 0.04, 90), ("b|b", 0.08, 10)):
            jaccards = base_j + rng.normal(0.0, 0.0005, size=100)
            for k, val in enumerate(jaccards):
                table.append({"membrane": "E", "same_hypothesis": 1, "placement": 701, "pair": pair_name,
                              "continuous_close": int(k < close_count), "edge_jaccard": float(val),
                              "functional_distance": float(rng.random()), "block_distance": float(rng.integers(1, 5))})
        res, = edge_level_test(table, 500, 7)
        self.assertLess(res["raw_difference"], -0.02)
        self.assertAlmostEqual(res["null_mean"], 0.0, places=4)
        self.assertAlmostEqual(res["difference"], 0.0, delta=0.001)
        self.assertGreater(res["two_sided_p"], 0.1)
