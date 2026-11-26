import json
import unittest
from dataclasses import replace

import numpy as np

from atlas import ROOT
from cortical.run import apply_model_spec, block_edges, cortical_drive, onset_latencies, summarize_seed
from model import Graph, Population, connect, population, simulate
from run import trial_stimuli


CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CORTICAL_PROTOCOL = json.loads((ROOT / "cortical/protocol.json").read_text(encoding="utf-8"))


class BiophysicsAndCorticalTests(unittest.TestCase):
    def test_synaptic_decay_conserves_charge_and_approaches_delta_limit(self):
        pop = Population(("exc", "inh", "post"), np.array([0, 1, 2]), np.zeros(3, dtype=int), np.zeros((3, 3)),
                         np.array([1.0, -1.0, -1.0]), np.array([2.0, 2.0, 1.0]), np.zeros(3), np.full(3, 1000.0))
        graph = Graph(np.array([0, 1]), np.array([[2], [2], [-1]], dtype=np.int32),
                      np.array([2.0, -2.0, -1.0]), np.array([[1], [1], [0]], dtype=np.int32), np.zeros((3, 1)))
        sim_delta = dict(dt_ms=1.0, baseline_ms=40, cue_ms=0, refractory_ms=2.0,
                         rest_mv=-65.0, reset_mv=-68.0, threshold_mv=100.0)
        noise = np.zeros((40, 3))
        noise[0, 0] = 200.0
        initial = np.full(3, -65.0)
        # Near-zero synaptic tau matches exact delta-jump spikes on normal thresholds
        sim_active = {**sim_delta, "threshold_mv": -64.0}
        spikes_delta = simulate(pop, graph, sim_active, noise, initial, np.zeros(3))
        spikes_limit = simulate(pop, graph, {**sim_active, "exc_tau_ms": 1e-4, "inh_tau_ms": 1e-4}, noise, initial, np.zeros(3))
        np.testing.assert_array_equal(spikes_delta, spikes_limit)
        with self.assertRaises(ValueError):
            simulate(pop, graph, {**sim_delta, "inh_tau_ms": -1.0}, noise, initial, np.zeros(3))

    def test_slow_ou_noise_is_paired_and_raises_cv_above_one(self):
        n = 400
        ones = np.ones(n)
        pop = Population(("GPe",), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros((n, 3)),
                         -ones, 0.65 * ones, 11.0 * ones, 14.0 * ones, noise_scale=2.5 * ones, slow_noise_mv=25.0 * ones)
        sim = {**CONFIG["simulation"], "baseline_ms": 1600, "cue_ms": 0, "slow_noise_tau_ms": 160.0}
        n1, i1 = trial_stimuli(pop, sim, 77, 0)
        n2, i2 = trial_stimuli(pop, sim, 77, 0)
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(i1, i2)
        graph = Graph(np.array([], dtype=int), np.full((n, 1), -1), np.zeros(n), np.zeros((n, 1), dtype=int), np.empty((0, 1)))
        spikes = simulate(pop, graph, sim, n1, i1, np.zeros(n))[200:]
        cvs = [np.diff(np.flatnonzero(col)).std() / np.diff(np.flatnonzero(col)).mean()
               for col in spikes.T if col.sum() >= 6]
        self.assertGreater(float(np.mean(cvs)), 1.2)

    def test_cortical_drive_conserves_integrated_amplitude_and_blocks_selected_edges(self):
        pop = population(CONFIG, None, 11)
        graph = connect("D", pop, {**CONFIG, "matched_channel_gain": 1.0}, 11)
        stim = {**CORTICAL_PROTOCOL["m1_stimulus"], "after_ms": 120}
        drive = cortical_drive(pop, 1.0, stim, 11, include_stn=True, include_striatum=False)
        stn = pop.cell_type == pop.names.index("STN")
        self.assertAlmostEqual(float(drive[:, ~stn].sum()), 0.0)
        driven_sums = drive[:, stn].sum(axis=0)
        self.assertEqual(int(np.sum(driven_sums > 0)), round( int(stn.sum()) * 0.5 ))
        for val in driven_sums[driven_sums > 0]:
            self.assertAlmostEqual(float(val), stim["stn"]["amplitude_mv"], delta=0.05)

        blocked_gaba = block_edges(pop, graph, ("D1", "D2", "SIn", "GPe"), ("GPi",))
        gpi = pop.cell_type == pop.names.index("GPi")
        stn_mask = pop.cell_type == pop.names.index("STN")
        to_gpi = (graph.targets >= 0) & gpi[graph.targets]
        self.assertTrue(np.all(blocked_gaba[(~stn_mask)[:, None] & to_gpi] == 0.0))
        self.assertTrue(np.all(blocked_gaba[stn_mask[:, None] & to_gpi] > 0.0))

    def test_cortical_summary_rejects_inhibition_dominated_by_pacemaker_trough(self):
        t = np.arange(80, dtype=float)
        triphasic = np.zeros(80)
        triphasic[8:15] = 8.0
        triphasic[17:26] = -4.0
        triphasic[30:45] = 3.0
        lat = onset_latencies(triphasic, 1.0, CORTICAL_PROTOCOL["windows"])
        self.assertIsNotNone(lat["early_onset_ms"])
        self.assertIsNotNone(lat["inhibition_onset_ms"])
        self.assertIsNotNone(lat["late_onset_ms"])

        base = {c: {"GPi": triphasic.copy()} for c in CORTICAL_PROTOCOL["interventions"]}
        base["direct_only"]["GPi"][:16] = 0.0
        base["stn_blocked"]["GPi"][:] = -2.0
        base["gpi_gaba_blocked"]["GPi"][16:28] = 1.0
        # Case 1: artifact (hyperdirect_gpe_blocked) is -3.0 Hz (> 50% of -4.0 Hz intact) -> rejected
        base["hyperdirect_gpe_blocked"]["GPi"][16:28] = -3.0
        row_bad, _ = summarize_seed(907, "m", base, 1.0, CORTICAL_PROTOCOL)
        self.assertEqual(row_bad["intact_triphasic_signs"], 1)
        self.assertEqual(row_bad["pathway_mediated_inhibition"], 0)
        self.assertEqual(row_bad["genuine_triphasic"], 0)
        # Case 2: artifact is -0.4 Hz (<= 50% of -4.0 Hz intact) -> accepted
        base["hyperdirect_gpe_blocked"]["GPi"][16:28] = -0.4
        row_good, _ = summarize_seed(907, "m", base, 1.0, CORTICAL_PROTOCOL)
        self.assertEqual(row_good["pathway_mediated_inhibition"], 1)
        self.assertEqual(row_good["genuine_triphasic"], 1)
        self.assertEqual(row_good["all_causal_checks"], 1)

        pop2, _, sim2 = apply_model_spec(population(CONFIG, None, 11), CONFIG,
                                         {"groups": {"GPi": {"bias_mv": 15.0, "slow_noise_mv": 8.0}},
                                          "simulation": {"inh_tau_ms": 14.0}})
        self.assertEqual(sim2["inh_tau_ms"], 14.0)
        self.assertIsNotNone(pop2.slow_noise_mv)

    def test_biophysics_selection_prefers_highest_strict_fraction_then_lowest_cv_error(self):
        from biophysics.run import select_condition
        rows = [
            {"kind": "factorial", "pathway": "baseline", "synapse": "delta", "fluctuation": "pacemaker",
             "strict_candidate": 0, "coarse_candidate": 1, "tonic_rate_mae_hz": 3.0, "tonic_cv_mae": 0.99},
            {"kind": "factorial", "pathway": "indirect_balanced", "synapse": "delta", "fluctuation": "moderate_ou",
             "strict_candidate": 1, "coarse_candidate": 1, "tonic_rate_mae_hz": 3.2, "tonic_cv_mae": 0.40},
            {"kind": "factorial", "pathway": "indirect_balanced", "synapse": "exponential", "fluctuation": "moderate_ou",
             "strict_candidate": 1, "coarse_candidate": 1, "tonic_rate_mae_hz": 3.2, "tonic_cv_mae": 0.35},
            {"kind": "decomposition", "pathway": "baseline", "synapse": "delta", "fluctuation": "stn_only_matched",
             "strict_candidate": 1, "coarse_candidate": 1, "tonic_rate_mae_hz": 0.1, "tonic_cv_mae": 0.01},
        ]
        chosen = select_condition(rows, {"max_tonic_rate_mae_hz": 5.0, "max_blocked_late_abs_hz": 0.5})
        self.assertEqual((chosen["pathway"], chosen["synapse"], chosen["fluctuation"]),
                         ("indirect_balanced", "exponential", "moderate_ou"))
