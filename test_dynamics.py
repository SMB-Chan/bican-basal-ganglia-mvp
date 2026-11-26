import json
import unittest
from dataclasses import replace

import numpy as np

from atlas import ROOT
from dynamics.run import probe_response, regime_population, select_regime, binned_min
from model import population
from run import trial_stimuli
from validation.run_validation import measure_pulse, measure_pulse_groups
from model import connect


CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SPEC = {"cells": 2000, "seed": 5101, "duration_ms": 1100, "burn_in_ms": 200, "min_intervals": 5, "probe_at_ms": 1000,
        "probe_jump_mv": 4.0, "probe_window_ms": 70, "trough_window_ms": [3, 28]}


class DynamicsTests(unittest.TestCase):
    def test_noise_scale_rescales_the_same_draws(self):
        pop = population(CONFIG, None, 11)
        sim = CONFIG["simulation"]
        plain, initial = trial_stimuli(pop, sim, 11, 0)
        scale = np.where(pop.cell_type == pop.names.index("GPi"), 3.0, 1.0)
        scaled, initial_scaled = trial_stimuli(replace(pop, noise_scale=scale), sim, 11, 0)
        np.testing.assert_array_equal(initial, initial_scaled)
        np.testing.assert_allclose(scaled, plain * scale)

    def test_group_pulse_matches_original_gpi_pulse(self):
        pop = population(CONFIG, None, 23)
        graph = connect("D", pop, {**CONFIG, "matched_channel_gain": 1.0}, 23)
        spec = json.loads((ROOT / "validation/protocol.json").read_text(encoding="utf-8"))["stn_pulse"]
        spec = {**spec, "repetitions": 1}
        groups = measure_pulse_groups(pop, graph, CONFIG["simulation"], spec, 23, ("GPi", "STN"))
        np.testing.assert_array_equal(measure_pulse(pop, graph, CONFIG["simulation"], spec, 23), groups["GPi"])

    def test_regime_changes_only_listed_groups(self):
        pop = population(CONFIG, None, 11)
        changed = regime_population(pop, {"GPi": {"bias_mv": 5.0, "noise_multiplier": 4.0}})
        gpi = pop.cell_type == pop.names.index("GPi")
        np.testing.assert_array_equal(changed.bias[gpi], 5.0)
        np.testing.assert_array_equal(changed.noise_scale[gpi], 4.0)
        np.testing.assert_array_equal(changed.bias[~gpi], pop.bias[~gpi])
        np.testing.assert_array_equal(changed.noise_scale[~gpi], 1.0)
        np.testing.assert_array_equal(changed.xyz, pop.xyz)

    def test_selection_ignores_pulse_and_blockade(self):
        base = [{"membrane": "E", "changed_groups": "all", "noise_multiplier": m, "tonic_rate_mae_hz": r,
                 "tonic_cv_mae": cv, "blocked_late_hz": 0.0} for m, r, cv in ((1, 3, 0.9), (2, 4, 0.5), (3, 6, 0.1))]
        rule = {"membrane": "E", "max_tonic_rate_mae_hz": 5.0, "minimize": "tonic_cv_mae"}
        first = select_regime(base, rule, "all")
        changed = [{**row, "blocked_late_hz": -9.0 * row["noise_multiplier"]} for row in base]
        self.assertEqual(first, select_regime(changed, rule, "all"))
        self.assertEqual(first["noise_multiplier"], 2.0)
        self.assertFalse(first["grid_edge"])

    def test_regular_cells_dip_after_a_kick_and_irregular_cells_do_not(self):
        dt = CONFIG["simulation"]["dt_ms"]
        regular = probe_response(CONFIG, "GPi", CONFIG["groups"]["GPi"]["bias_mv"], 1.0, SPEC)
        irregular = probe_response(CONFIG, "GPi", 2.46, 10.0, SPEC)
        self.assertLess(binned_min(regular, SPEC["trough_window_ms"], 5, dt), -5.0)
        self.assertGreater(binned_min(irregular, SPEC["trough_window_ms"], 5, dt), -3.0)
