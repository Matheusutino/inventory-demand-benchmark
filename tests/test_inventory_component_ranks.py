"""Check task-wise ranking, ties, coverage and component independence."""

import unittest

import numpy as np
import pandas as pd

from src.analysis.plot_inventory_component_rank_heatmap import compute_component_ranks


class ComponentRankTests(unittest.TestCase):
    def test_rank_before_average_and_keep_components_independent(self):
        rows = []
        for task, component_values in (("a", (0, 1)), ("b", (0, 1)), ("c", (1000, 0))):
            rows.extend([
                {"dataset": task, "model": "A", "item": component_values[0], "sum": component_values[1]},
                {"dataset": task, "model": "B", "item": component_values[1], "sum": component_values[0]},
            ])
        values = pd.DataFrame(rows)
        ranks, means = compute_component_ranks(values, ["item", "sum"])
        means = means.set_index("model")
        self.assertAlmostEqual(means.loc["A", "item"], 4 / 3)
        self.assertAlmostEqual(means.loc["B", "item"], 5 / 3)
        self.assertAlmostEqual(means.loc["A", "sum"], 5 / 3)
        self.assertAlmostEqual(means.loc["B", "sum"], 4 / 3)
        self.assertTrue((means.n_tasks == 3).all())
        self.assertEqual(len(ranks), 12)
        self.assertGreater(values.loc[values.model == "A", "item"].mean(), values.loc[values.model == "B", "item"].mean())
        self.assertNotIn("Overall", means)
        self.assertNotIn("avg_rank", means)

    def test_ties_share_rank_positions_including_constant_components(self):
        values = pd.DataFrame({
            "dataset": ["task"] * 3, "model": ["A", "B", "C"],
            "item": [1, 1, 3], "cap": [0, 0, 0],
        })
        _, means = compute_component_ranks(values, ["item", "cap"])
        means = means.set_index("model")
        np.testing.assert_allclose(means.loc[["A", "B", "C"], "item"], [1.5, 1.5, 3])
        np.testing.assert_allclose(means.cap, 2)

    def test_task_specific_scales_do_not_affect_ranks(self):
        values = pd.DataFrame({
            "dataset": ["a", "a", "b", "b"], "model": ["A", "B", "A", "B"],
            "item": [1, 2, 4, 3],
        })
        _, original = compute_component_ranks(values, ["item"])
        transformed = values.copy()
        transformed["item"] = transformed.item * transformed.dataset.map({"a": 10000, "b": 0.01}) + 7
        _, rescaled = compute_component_ranks(transformed, ["item"])
        pd.testing.assert_frame_equal(original, rescaled)

    def test_missing_duplicate_or_nonfinite_results_are_rejected(self):
        values = pd.DataFrame({
            "dataset": ["a", "a", "b", "b"], "model": ["A", "B", "A", "B"],
            "item": [1.0, 2.0, 4.0, 3.0],
        })
        with self.assertRaisesRegex(ValueError, "desbalanceada"):
            compute_component_ranks(values.iloc[:-1], ["item"])
        with self.assertRaisesRegex(ValueError, "exatamente uma vez"):
            compute_component_ranks(pd.concat([values, values.iloc[:1]]), ["item"])
        for invalid in (np.nan, np.inf):
            with self.subTest(value=invalid):
                broken = values.copy()
                broken.loc[0, "item"] = invalid
                with self.assertRaisesRegex(ValueError, "não finitos"):
                    compute_component_ranks(broken, ["item"])

    def test_inconsistent_sku_coverage_is_rejected(self):
        values = pd.DataFrame({
            "dataset": ["a", "a"], "model": ["A", "B"], "item": [1.0, 2.0],
            "matched_rows": [6, 3], "n_series": [2, 1], "horizon": [3, 3],
        })
        with self.assertRaisesRegex(ValueError, "Cobertura diferente"):
            compute_component_ranks(values, ["item"])
        values["matched_rows"] = 5
        values["n_series"] = 2
        with self.assertRaisesRegex(ValueError, "Cobertura incompleta"):
            compute_component_ranks(values, ["item"])

    def test_rows_group_families_without_ordering_by_overall_performance(self):
        values = pd.DataFrame({
            "dataset": ["a"] * 5,
            "model": ["chronos-2", "darts_randomforest", "darts_arima", "darts_global_naive_drift", "statsforecast_tsb"],
            "item": [1.0, 2.0, 5.0, 4.0, 3.0],
        })
        _, means = compute_component_ranks(values, ["item"])
        self.assertEqual(means.family.tolist(), ["naive", "statistical", "intermittent", "machine_learning", "foundation"])
        self.assertEqual(means.pretty_model.tolist(), ["Naive Drift", "ARIMA", "TSB", "Random Forest", "Chronos-2"])
        values["total"] = 1
        with self.assertRaisesRegex(ValueError, "fora da comparação principal"):
            compute_component_ranks(values, ["total"])
        values["sparse"] = 1
        with self.assertRaisesRegex(ValueError, "fora da comparação principal"):
            compute_component_ranks(values, ["sparse"])


if __name__ == "__main__":
    unittest.main()
