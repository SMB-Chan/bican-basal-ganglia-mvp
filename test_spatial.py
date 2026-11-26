import json
import unittest

import numpy as np

from atlas import ROOT
from model import Graph, Population, population
from spatial.census import classify, dorsal_striatum, mixing, model_counts
from spatial.composition import composition_counts, drive_matched_weights, pair_in_degree, with_counts


def synthetic_cells(n=2400, clustered=False, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 4, n), rng.uniform(0, 4, n)
    d1 = rng.random(n) < 0.5
    # Striosome patches: one 1 mm disc per quadrant when clustered, random labels otherwise.
    patch = np.zeros(n, dtype=bool)
    for cx, cy in ((1, 1), (3, 1), (1, 3), (3, 3)):
        patch |= (x - cx) ** 2 + (y - cy) ** 2 < 0.25
    striosome = patch if clustered else rng.random(n) < patch.mean()
    group = np.where(d1, "STRd D1 ", "STRd D2 ").astype(object) + np.where(striosome, "Striosome SPN", "Matrix SPN")
    return {
        "brain_section_label": np.full(n, "S1"), "slab_plane_label": np.full(n, "P1"), "x": x, "y": y,
        "Neighborhood": np.full(n, "Subpallium GABA"), "Class": np.full(n, "CN LGE GABA"),
        "Subclass": np.where(d1, "STR D1 SPN", "STR D2 SPN"), "Group": group.astype(str),
        "Subclass_bootstrapping_probability": np.full(n, "1.0"), "Group_bootstrapping_probability": np.full(n, "1.0"),
        "hierarchy_consistent": np.full(n, "True"), "parcellation_term_symbol": np.full(n, "STR"),
    }


class SpatialCensusTests(unittest.TestCase):
    def test_classification_separates_dorsal_ventral_and_interneurons(self):
        cells = {"Neighborhood": np.array(["Subpallium GABA"] * 4 + ["Nonneuron"]),
                 "Class": np.array(["CN LGE GABA", "CN LGE GABA", "CN MGE GABA", "CN LGE GABA", "OPC-Oligo"]),
                 "Subclass": np.array(["STR D1 SPN", "STR D2 SPN", "CN ST18 GABA", "STR Eccentric SPN", "Oligo"]),
                 "Group": np.array(["STRd D1 Matrix SPN", "STRv D2 SPN", "STR FS PTHLH-PVALB GABA",
                                    "STR D1D2 eSPN", "Oligo OPALIN"])}
        np.testing.assert_array_equal(classify(cells), ["STRd_SPN", "STRv_SPN", "interneuron", "eSPN", "other"])

    def test_dorsal_tiles_need_dorsal_majority(self):
        cells = synthetic_cells(2400)
        cells["kind"] = classify(cells)
        self.assertTrue(dorsal_striatum(cells, 0.5).all())
        cells["kind"][:] = "STRv_SPN"
        self.assertFalse(dorsal_striatum(cells, 0.5).any())

    def test_mixing_detects_patches_but_not_random_labels(self):
        for clustered in (False, True):
            cells = synthetic_cells(clustered=clustered)
            cells["kind"] = classify(cells)
            row, = mixing(cells, permutations=40)
            self.assertLess(abs(row["d1d2_section_excess"]), 0.02)
            if clustered:
                self.assertGreater(row["compartment_section_excess"], 0.1)
                self.assertGreater(row["compartment_observed"], row["compartment_tile_null_q975"])
            else:
                self.assertLess(abs(row["compartment_section_excess"]), 0.02)

    def test_model_counts_keep_totals_and_even_sizes(self):
        composition = {"interneuron_fraction_of_spn_plus_interneurons": 0.098, "d1_fraction_of_dorsal_spn": 0.4955}
        density = {"regions": {"GPe": {"share_of_GPe_STN_GPi": 0.619}, "STN": {"share_of_GPe_STN_GPi": 0.161},
                               "GPi": {"share_of_GPe_STN_GPi": 0.220}}}
        counts = model_counts(composition, density)
        self.assertEqual(counts, {"D1": 224, "D2": 228, "SIn": 48, "GPe": 310, "STN": 80, "GPi": 110})


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.protocol = json.loads((ROOT / "spatial" / "composition_protocol.json").read_text(encoding="utf-8"))

    def test_compositions_keep_total_and_assumed_first(self):
        census = {"model_counts_from_measurement": {"D1": 224, "D2": 228, "SIn": 48, "GPe": 310, "STN": 80, "GPi": 110}}
        counts = composition_counts(self.config, census, self.protocol)
        self.assertEqual(next(iter(counts)), "assumed")
        self.assertTrue(all(sum(c.values()) == 1000 for c in counts.values()))
        self.assertEqual(counts["striatal_literature"]["SIn"], 70)
        self.assertEqual(counts["pallidal_measured"]["D1"], 236)
        self.assertEqual(population(with_counts(self.config, counts["both_measured"]), None, 3).cell_type.size, 1000)

    def test_drive_matching_restores_reference_input_per_group_pair(self):
        cell_type = np.array([0, 0, 0, 0, 1, 1])
        ones = np.ones(6)
        pop = Population(("A", "B"), cell_type, np.zeros(6, int), np.zeros((6, 3)), ones, ones, ones, ones)
        targets = np.array([[4, 5], [4, 5], [4, 1], [4, 5], [0, 1], [0, 2]])
        graph = Graph(np.arange(6), targets, np.array([1.0, 1, 1, 1, -2, -2]), np.ones((6, 2), int), np.zeros((6, 2)))
        realized = pair_in_degree(pop, graph)
        np.testing.assert_allclose(realized, [[0.25, 3.5], [1.0, 0]])
        reference = np.array([[0.5, 7.0], [2.0, 0]])
        weights, max_log_ratio, unmatched = drive_matched_weights(pop, graph, reference)
        np.testing.assert_allclose(weights[0], [2.0, 2.0])
        np.testing.assert_allclose(weights[4], [-4.0, -4.0])
        self.assertAlmostEqual(max_log_ratio, np.log(2))
        self.assertEqual(unmatched, 0)
        same, ratio, _ = drive_matched_weights(pop, graph, realized)
        np.testing.assert_allclose(same, np.broadcast_to(graph.weights[:, None], targets.shape))
        self.assertEqual(ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
