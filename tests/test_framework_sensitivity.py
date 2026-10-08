"""Check one-factor formulas, coverage, strict Cap limits and ranking comparisons."""

import unittest

import numpy as np
import pandas as pd
import torch

from src.analysis.analyze_framework_sensitivity import (
    SPECIFIED_MODELS, Variation, add_panel_omissions, cap_violations, compare_variations,
    primary_variations, rank_variations, select_highlights, training_scales,
    variation_coverage, variation_value,
)
from src.evaluation.forecast_evaluation import DEFAULT_EVALUATION_CONFIG, compute_components
from src.analysis.inventory_framework import COMPONENTS
from src.analysis.model_style import model_family, pretty_model_name


class FrameworkSensitivityTests(unittest.TestCase):
    def setUp(self):
        history = np.column_stack([np.arange(24), np.tile([1, 3], 12), np.full(24, 4), np.zeros(24)])
        self.history = torch.tensor(history, dtype=torch.float64).unsqueeze(0)
        self.target = torch.tensor([[[0., 2., 2., 0.], [1., 3., 0., 0.], [2., 2., 4., 0.]]], dtype=torch.float64)
        self.pred = torch.tensor([[[2., 0., 6., 3.], [3., 6., -2., 0.], [1., 3., 5., -1.]]], dtype=torch.float64)
        self.scales = training_scales(self.history)
        self.originals = compute_components(self.pred, self.target, DEFAULT_EVALUATION_CONFIG, self.history)

    def value(self, component, setting):
        variation = Variation(component, "test", "test", setting)
        return variation_value(self.pred, self.target, self.scales, variation, self.originals)

    def test_every_requested_setting_and_original_is_present_once(self):
        variants = primary_variations()
        self.assertEqual(len(variants), 23)
        self.assertEqual(len({(v.component, v.variation_id) for v in variants}), 23)
        self.assertEqual({v.component for v in variants if v.is_original}, set(COMPONENTS))
        self.assertEqual([v.setting for v in variants if v.component == "cap"], ["2", "1.25", "1.5", "3", "5"])
        for v in variants:
            if v.is_original:
                self.assertEqual(variation_value(self.pred, self.target, self.scales, v, self.originals),
                                 self.originals[v.component])

    def test_asym_half_tau_is_half_mae_and_extreme_taus_penalize_underprediction(self):
        self.assertAlmostEqual(self.value("asym", "0.5"), self.originals["item"] / 2)
        for tau in (.5, .6, .7, .8, .9):
            variation = Variation("asym", "test", "tau", str(tau))
            under = variation_value(self.target - 1, self.target, self.scales, variation, self.originals)
            over = variation_value(self.target + 1, self.target, self.scales, variation, self.originals)
            self.assertAlmostEqual(under, tau)
            self.assertAlmostEqual(over, 1 - tau)

    def test_scales_use_only_training_pairs_and_recompute_eligibility(self):
        np.testing.assert_array_equal(self.scales["lag1"].numpy().ravel(), [1, 2, 0, 0])
        np.testing.assert_array_equal(self.scales["lag12"].numpy().ravel(), [12, 0, 0, 0])
        np.testing.assert_array_equal(self.scales["mean"].numpy().ravel(), [11.5, 2, 4, 0])
        errors = (self.pred - self.target).abs().numpy()[0]
        self.assertAlmostEqual(self.value("scaled", "lag12"), errors[:, 0].mean() / 12)
        self.assertAlmostEqual(self.value("scaled", "mean"), (errors[:, :3] / [11.5, 2, 4]).mean())
        for setting, eligible in (("lag1", 2), ("lag12", 1), ("mean", 3)):
            coverage = variation_coverage(self.target, self.scales, Variation("scaled", "x", "scale", setting))
            self.assertEqual(coverage["eligible_skus"], eligible)
            self.assertEqual(coverage["n_observations"], 3 * eligible)
        with self.assertRaises(ValueError):
            training_scales(self.history[:, :12])

    def test_zero_keeps_signed_forecasts_and_same_eligible_observations_even_without_normalization(self):
        self.assertAlmostEqual(self.value("zero", "max"), (2/23 - 2/4) / 2)
        self.assertEqual(self.value("zero", "none"), 0)
        for setting in ("mean", "max", "none"):
            coverage = variation_coverage(self.target, self.scales, Variation("zero", "x", "normalization", setting))
            self.assertEqual(coverage["eligible_skus"], 3)
            self.assertEqual(coverage["excluded_skus"], 1)
            self.assertEqual(coverage["n_observations"], 2)
        changed = self.pred.clone()
        changed[:, :, 3] = 1e6  # The original no-demand-history exclusion remains in every Zero variant.
        value = variation_value(changed, self.target, self.scales, Variation("zero", "x", "normalization", "none"), self.originals)
        self.assertEqual(value, 0)

    def test_sum_preserves_relative_denominator_and_delta_is_unanchored(self):
        pred, target = self.pred.numpy()[0], self.target.numpy()[0]
        expected_sum = np.abs((pred.sum(1) - target.sum(1)) / target.sum(1)).mean()
        expected_delta = np.abs(np.diff(pred - target, axis=0)).mean()
        self.assertAlmostEqual(self.value("sum", "absolute"), expected_sum)
        self.assertAlmostEqual(self.value("delta", "absolute"), expected_delta)
        # A constant level shift affects Sum but contributes nothing to either Delta version.
        value = variation_value(self.target + 100, self.target, self.scales,
                                Variation("delta", "x", "error", "absolute"), self.originals)
        self.assertEqual(value, 0)

    def test_cap_counts_strict_violations_and_never_relaxes_zero_history_caps(self):
        maximum = torch.tensor([[[10., 0.]]], dtype=torch.float64)
        pred = torch.tensor([[[20., 0.], [21., 1.], [50., -1.]]], dtype=torch.float64)
        at_two = cap_violations(pred, maximum, 2)
        self.assertEqual(at_two["n_violations"], 3)
        self.assertEqual(at_two["n_predictions"], 6)
        self.assertEqual(at_two["zero_history_violations"], 1)
        self.assertEqual(at_two["n_violating_skus"], 2)
        self.assertEqual(cap_violations(pred, maximum, 5)["n_violations"], 1)
        for kappa in (1.25, 1.5, 2, 3, 5):
            self.assertEqual(cap_violations(pred, maximum, kappa)["zero_history_violations"], 1)
        value = variation_value(pred, torch.zeros_like(pred), {"max": maximum},
                                Variation("cap", "x", "kappa", "2"), {})
        self.assertAlmostEqual(value, 902 / 6)

    def test_undefined_scaled_or_zero_is_nan_and_cannot_be_ranked(self):
        scales = training_scales(torch.zeros_like(self.history))
        for component, setting in (("scaled", "lag12"), ("zero", "none")):
            value = variation_value(self.pred, self.target, scales, Variation(component, "x", "test", setting), {})
            self.assertTrue(np.isnan(value))
        values = self.rank_fixture()
        values.loc[0, "value"] = np.nan
        with self.assertRaises(ValueError):
            rank_variations(values)

    def rank_fixture(self):
        rows = []
        for component in COMPONENTS:
            meta = Variation(component, "original", "configuration", "original", is_original=True).metadata()
            for panel, component_values in (("A", [1, 1, 3, 4, 5, 5]), ("B", [6, 5, 4, 3, 2, 1])):
                for horizon in (3, 6):
                    for model, value in zip("ABCDEF", component_values):
                        rows.append({**meta, "dataset": f"{panel}_h{horizon}", "panel": panel,
                                     "model": model, "value": value, "matched_rows": horizon,
                                     "horizon": horizon, "n_series": 1, "n_train": 24})
        return pd.DataFrame(rows)

    def test_rank_before_averaging_and_remove_both_horizons_for_robustness(self):
        values = add_panel_omissions(self.rank_fixture())
        tasks, means = rank_variations(values)
        original = means.loc[(means.component == "item") & means.is_original].set_index("model")
        np.testing.assert_allclose(original.loc[list("ABCDEF"), "mean_rank"], [3.75, 3.25, 3.5, 3.5, 3.75, 3.25])
        omitted = means.loc[(means.component == "item") & (means.excluded_panel == "B")].set_index("model")
        np.testing.assert_allclose(omitted.loc[list("ABCDEF"), "mean_rank"], [1.5, 1.5, 3, 4, 5.5, 5.5])
        self.assertTrue((omitted.n_tasks == 2).all())
        self.assertEqual(set(tasks.loc[tasks.excluded_panel == "B", "dataset"]), {"A_h3", "A_h6"})
        summary = compare_variations(means)
        original_summary = summary.loc[(summary.component == "item") & summary.is_original].iloc[0]
        self.assertTrue(original_summary.original_top5_boundary_tie)
        self.assertEqual(original_summary.top5_tie_inclusive_overlap, 6)
        self.assertEqual(original_summary.top5_overlap, 5)
        self.assertEqual(original_summary.n_leaders, 2)
        omission = summary.loc[(summary.component == "item") & (summary.excluded_panel == "A")].iloc[0]
        self.assertTrue(omission.leader_changed)
        self.assertEqual(omission.top5_overlap, 4)
        self.assertEqual(omission.n_tasks, 2)
        with self.assertRaises(ValueError):
            rank_variations(values.iloc[1:])

    def test_constant_mean_rank_vectors_have_undefined_spearman_and_all_models_lead(self):
        _, means = rank_variations(self.rank_fixture().assign(value=1))
        summary = compare_variations(means)
        self.assertTrue(summary.spearman.isna().all())
        self.assertTrue((summary.n_leaders == 6).all())
        self.assertFalse(summary.leader_changed.any())

    def test_highlights_choose_other_family_best_separately_for_each_component(self):
        models = [*SPECIFIED_MODELS, "darts_global_naive_aggregate", "darts_global_naive_seasonal",
                  "darts_arima", "darts_prophet", "statsforecast_tsb", "darts_randomforest"]
        rows = []
        for component in ("asym", "cap"):
            for model in models:
                rank = 1
                if (component == "asym" and model == "darts_prophet") or (component == "cap" and model == "darts_arima"):
                    rank = 2
                rows.append({"component": component, "model": model, "mean_rank": rank,
                             "is_original": True, "family": model_family(model), "pretty_model": pretty_model_name(model)})
        highlights = select_highlights(pd.DataFrame(rows))
        self.assertEqual(len(highlights), 18)
        self.assertIn("darts_arima", set(highlights.loc[highlights.component == "asym", "model"]))
        self.assertIn("darts_prophet", set(highlights.loc[highlights.component == "cap", "model"]))
        self.assertFalse((highlights.model == "darts_global_naive_seasonal").any())


if __name__ == "__main__":
    unittest.main()
