import json
import unittest

import numpy as np

from atlas import ROOT
from model import Graph, Population, connect, population, simulate
from parkinsonian.run import configure_pd_condition, oscillation_metrics, welch_psd
from rebound.run import measure_isolated_rebound, select_rebound_variant


CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
REBOUND_PROTOCOL = json.loads((ROOT / "rebound/protocol.json").read_text(encoding="utf-8"))
PD_PROTOCOL = json.loads((ROOT / "parkinsonian/protocol.json").read_text(encoding="utf-8"))


class ReboundAndParkinsonianTests(unittest.TestCase):
    def test_ifb_rebound_requires_prior_hyperpolarization_and_null_matches_lif(self):
        pop_lif = population(CONFIG, None, 11)
        graph = connect("D", pop_lif, {**CONFIG, "matched_channel_gain": 1.0}, 11)
        rng = np.random.default_rng(19)
        noise = rng.standard_normal((120, len(pop_lif.cell_type))) * 1.1
        initial = rng.uniform(-68.0, -50.0, len(pop_lif.cell_type))
        sim_delta = {**CONFIG["simulation"], "baseline_ms": 120, "cue_ms": 0}
        spikes_none = simulate(pop_lif, graph, sim_delta, noise, initial, np.zeros(len(pop_lif.cell_type)))
        pop_zero = Population(*[getattr(pop_lif, k) for k in pop_lif.__dataclass_fields__ if k != "rebound_mv"],
                              rebound_mv=np.zeros(len(pop_lif.cell_type)))
        spikes_zero = simulate(pop_zero, graph, sim_delta, noise, initial, np.zeros(len(pop_lif.cell_type)))
        np.testing.assert_array_equal(spikes_none, spikes_zero)

        reb_passive, _ = measure_isolated_rebound(CONFIG, REBOUND_PROTOCOL, 0.0, 1009)
        reb_ifb, _ = measure_isolated_rebound(CONFIG, REBOUND_PROTOCOL, 65.0, 1009)
        self.assertLess(reb_passive, 0.0)
        self.assertGreater(reb_ifb, 10.0)

        # Without hyperpolarization below vh (-66.5 mV), a brief depolarizing pulse does not trigger late rebound
        n = 150
        ones = np.ones(n)
        pop_ifb = Population(("STN",), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros((n, 3)),
                             ones, ones, 15.0 * ones, 15.0 * ones, rebound_mv=65.0 * ones)
        empty = Graph(np.array([], dtype=int), np.full((n, 1), -1), np.zeros(n),
                      np.zeros((n, 1), dtype=int), np.empty((0, 1)))
        sim = {**CONFIG["simulation"], "baseline_ms": 300, "cue_ms": 0, **REBOUND_PROTOCOL["rebound_kinetics"]}
        n_draw = np.random.default_rng(23).standard_normal((300, n)) * 1.5
        init = np.full(n, -60.0)
        sham = simulate(pop_ifb, empty, sim, n_draw, init, np.zeros(n))
        depol = n_draw.copy()
        depol[150:160, :] += 2.0
        evoked = simulate(pop_ifb, empty, sim, depol, init, np.zeros(n))
        late_diff_hz = float((evoked[175:210].mean() - sham[175:210].mean()) * 1000.0)
        self.assertLess(late_diff_hz, 3.0)

        with self.assertRaises(ValueError):
            simulate(pop_ifb, empty, {**sim, "rebound_tau_plus_ms": 0.0}, n_draw, init, np.zeros(n))

    def test_welch_oscillation_index_is_rate_invariant_for_poisson_and_detects_low_beta(self):
        rng = np.random.default_rng(31)
        steps = 6000
        n_cells = 160
        mask = np.ones(n_cells, dtype=bool)
        sp_spec = PD_PROTOCOL["spontaneous"]
        sp_low = rng.random((steps, n_cells)) < (15.0 / 1000.0)
        sp_high = rng.random((steps, n_cells)) < (45.0 / 1000.0)
        m_low, _, _ = oscillation_metrics(sp_low, mask, 1.0, sp_spec)
        m_high, _, _ = oscillation_metrics(sp_high, mask, 1.0, sp_spec)
        self.assertAlmostEqual(m_low["pop_beta_index"], 1.0, delta=0.45)
        self.assertAlmostEqual(m_high["pop_beta_index"], 1.0, delta=0.45)
        self.assertAlmostEqual(m_high["unit_beta_index"], 1.0, delta=0.20)

        t_s = np.arange(steps)[:, None] / 1000.0
        rate_mod = 25.0 * (1.0 + 0.75 * np.sin(2.0 * np.pi * 12.0 * t_s))
        sp_beta = rng.random((steps, n_cells)) < (rate_mod / 1000.0)
        m_beta, freqs, _ = oscillation_metrics(sp_beta, mask, 1.0, sp_spec)
        self.assertEqual(m_beta["peak_freq_hz"], 12.0)
        self.assertGreater(m_beta["pop_beta_index"], 8.0)
        self.assertGreater(m_beta["unit_beta_index"], 1.3)
        self.assertAlmostEqual(float(freqs[1] - freqs[0]), 1.0)

    def test_pd_condition_preserves_extrastriatal_intrinsic_params_and_applies_blockades(self):
        pop = population(CONFIG, None, 11)
        graph = connect("D", pop, {**CONFIG, "matched_channel_gain": 1.0}, 11)
        pd_spec = PD_PROTOCOL["dopamine_depletion"]

        p_norm, w_norm, g_norm = configure_pd_condition(pop, graph, pd_spec, "normal_intact")
        self.assertIsNone(w_norm)
        np.testing.assert_array_equal(p_norm.bias, pop.bias)
        self.assertEqual(g_norm["D1"], 1.0)

        p_pd, w_pd, g_pd = configure_pd_condition(pop, graph, pd_spec, "pd_intact")
        for grp in ("SIn", "GPe", "STN", "GPi"):
            m = pop.cell_type == pop.names.index(grp)
            np.testing.assert_array_equal(p_pd.bias[m], pop.bias[m])
        d1 = pop.cell_type == pop.names.index("D1")
        d2 = pop.cell_type == pop.names.index("D2")
        np.testing.assert_allclose(p_pd.bias[d1], pop.bias[d1] + pd_spec["d1_bias_shift_mv"])
        np.testing.assert_allclose(p_pd.bias[d2], pop.bias[d2] + pd_spec["d2_bias_shift_mv"])
        self.assertLess(g_pd["D1"], 1.0)
        self.assertGreater(g_pd["D2"], 1.0)

        gpe_idx = pop.names.index("GPe")
        stn_idx = pop.names.index("STN")
        gpe_stn = (pop.cell_type[:, None] == gpe_idx) & (graph.targets >= 0) & (pop.cell_type[graph.targets] == stn_idx)
        base_w = np.broadcast_to(graph.weights[:, None], graph.targets.shape)
        np.testing.assert_allclose(w_pd[gpe_stn], base_w[gpe_stn] * pd_spec["gpe_to_stn_weight_scale"])

        _, w_str, _ = configure_pd_condition(pop, graph, pd_spec, "pd_striatal_only")
        np.testing.assert_allclose(w_str[gpe_stn], base_w[gpe_stn])

        _, w_stn_blk, _ = configure_pd_condition(pop, graph, pd_spec, "pd_stn_blocked")
        self.assertTrue(np.all(w_stn_blk[pop.cell_type == stn_idx] == 0.0))

        _, w_gpe_blk, _ = configure_pd_condition(pop, graph, pd_spec, "pd_gpe_to_stn_blocked")
        self.assertTrue(np.all(w_gpe_blk[gpe_stn] == 0.0))

    def test_rebound_selection_rule_maximizes_joint_normal_pass_then_lowest_errors(self):
        rows = [
            {"pathway": "gpe_gpi_heavy", "synapse": "delta", "rebound": "passive",
             "joint_normal_pass": 0, "kita_strict_pass": 1, "m1_all_causal_pass": 0,
             "tonic_rate_mae_hz": 3.0, "tonic_cv_mae": 0.35, "triphasic_latency_mae_ms": 1.8},
            {"pathway": "stn_loop_dominant", "synapse": "delta", "rebound": "moderate_ifb",
             "joint_normal_pass": 1, "kita_strict_pass": 1, "m1_all_causal_pass": 1,
             "tonic_rate_mae_hz": 3.2, "tonic_cv_mae": 0.36, "triphasic_latency_mae_ms": 2.1},
            {"pathway": "stn_loop_dominant", "synapse": "physiological_37c", "rebound": "moderate_ifb",
             "joint_normal_pass": 1, "kita_strict_pass": 1, "m1_all_causal_pass": 1,
             "tonic_rate_mae_hz": 3.1, "tonic_cv_mae": 0.33, "triphasic_latency_mae_ms": 1.7},
        ]
        chosen = select_rebound_variant(rows, REBOUND_PROTOCOL["selection"])
        self.assertEqual(chosen["variant"], "stn_loop_dominant|physiological_37c|moderate_ifb")

    def test_batched_pulse_and_m1_match_sequential_exactly(self):
        from cortical.run import evaluate_seed_model
        from rebound.run import fast_m1_evaluate, fast_pulse_groups
        from validation.run_validation import measure_pulse_groups

        vprot = json.loads((ROOT / "validation/protocol.json").read_text(encoding="utf-8"))
        cprot = json.loads((ROOT / "cortical/protocol.json").read_text(encoding="utf-8"))
        pspec = {**vprot["stn_pulse"], "repetitions": 3}
        cspec = {**cprot, "m1_stimulus": {**cprot["m1_stimulus"], "repetitions": 3}}
        pop = population(CONFIG, None, 11)
        pop = Population(*[getattr(pop, k) for k in pop.__dataclass_fields__ if k != "rebound_mv"],
                         rebound_mv=np.where(pop.cell_type == pop.names.index("STN"), 75.0, 0.0))
        graph = connect("D", pop, {**CONFIG, "matched_channel_gain": 1.0}, 11)
        sim = {**CONFIG["simulation"], "exc_tau_ms": 2.0, "inh_tau_ms": 4.0, **REBOUND_PROTOCOL["rebound_kinetics"]}

        seq_p = measure_pulse_groups(pop, graph, sim, pspec, 11, ("GPi", "STN"))
        bat_p = fast_pulse_groups(pop, graph, sim, pspec, 11, ("GPi", "STN"))
        for g in ("GPi", "STN"):
            np.testing.assert_allclose(seq_p[g], bat_p[g], rtol=0, atol=1e-12)

        seq_m = evaluate_seed_model(pop, graph, sim, cspec, 11)
        bat_m = fast_m1_evaluate(pop, graph, sim, cspec, 11)
        for cond in cspec["interventions"]:
            for g in ("GPi", "GPe", "STN"):
                np.testing.assert_allclose(seq_m[cond][g], bat_m[cond][g], rtol=0, atol=1e-12)
