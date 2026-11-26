import json
import unittest

import numpy as np

from atlas import ROOT
from confirmation.analyze import score
from model import connect, population
from oscillation.run import loop_weights


CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


class FollowUpTests(unittest.TestCase):
    def test_loop_weights_scale_only_the_two_loop_projections(self):
        pop = population(CONFIG, None, 11)
        graph = connect("D", pop, {**CONFIG, "matched_channel_gain": 1.0}, 11)
        base = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
        scaled = loop_weights(pop, graph, 3.0, 2.0)
        gpe, stn = pop.names.index("GPe"), pop.names.index("STN")
        valid = graph.targets >= 0
        post = np.where(valid, pop.cell_type[np.maximum(graph.targets, 0)], -1)
        es = (pop.cell_type[:, None] == gpe) & (post == stn)
        se = (pop.cell_type[:, None] == stn) & (post == gpe)
        self.assertTrue(es.any() and se.any())
        np.testing.assert_allclose(scaled[es], 3.0 * base[es])
        np.testing.assert_allclose(scaled[se], 2.0 * base[se])
        np.testing.assert_array_equal(scaled[~(es | se)], base[~(es | se)])

    def test_confirmation_criteria(self):
        rows = []
        for seed in range(10):
            for variant, passed in (("a|x|y", 1), ("b|x|y", int(seed < 5))):
                pathway, synapse, rebound = variant.split("|")
                rows.append({"seed": seed, "variant": variant, "pathway": pathway, "synapse": synapse, "rebound": rebound,
                             "joint_normal_pass": passed, "kita_strict_pass": passed, "m1_all_causal_pass": passed,
                             "nambu_stn_block_pass": 1, "tachibana_gaba_block_pass": 1, "isolated_rebound_pass": 1,
                             "tonic_rate_mae_hz": 3.0, "tonic_cv_mae": 0.4, "stn_block_late_drop_hz": 1.0,
                             "blocked_late_hz": 0.0, "triphasic_latency_mae_ms": 2.0})
        protocol = {"frozen_variant": "a|x|y", "original_joint_normal_pass": [8, 12],
                    "bootstrap_replicates": 500, "bootstrap_seed": 1}
        _, paired, verdict = score(rows, protocol)
        self.assertTrue(verdict["C1_selection_stable"]["pass"])
        self.assertEqual(verdict["C2_superiority"]["best_alternative"], "b|x|y")
        self.assertAlmostEqual(verdict["C2_superiority"]["mean_difference"], 0.5)
        self.assertFalse(verdict["C3_no_shrinkage"]["pass"])


class ReadoutTests(unittest.TestCase):
    def test_rhythm_moves_only_target_cells_with_the_requested_equilibrium_amplitude(self):
        from readout.run import rhythm
        pop = population(CONFIG, None, 11)
        drive = rhythm(pop, "STN", 2.0, 12.0, 500, 1.0)
        stn = pop.cell_type == pop.names.index("STN")
        self.assertTrue(np.all(drive[:, ~stn] == 0))
        # Summing the per-step input through the membrane filter gives an equilibrium shift of `amplitude`.
        tau = pop.tau[stn][0]
        self.assertAlmostEqual(np.abs(drive[:, stn][:, 0]).max() / (1 - np.exp(-1 / tau)), 2.0, places=2)
        np.testing.assert_array_equal(rhythm(pop, "GPi", 0.0, 12.0, 10, 1.0), 0)
