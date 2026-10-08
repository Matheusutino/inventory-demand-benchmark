import unittest

import numpy as np
import pandas as pd

from src.analysis.analyze_component_concordance import (
    compute_concordance, correlation_matrix, prepare_ranks,
)
from src.analysis.plot_inventory_component_rank_heatmap import COMPONENTS


class ComponentConcordanceTests(unittest.TestCase):
    def inputs(self):
        component_values = []
        for task in ("panel_h3", "panel_h6"):
            for index, model in enumerate(("A", "B", "C")):
                row = {"dataset": task, "model": model, "matched_rows": 6,
                       "n_series": 2, "horizon": 3}
                row.update({component: float(index + 1) for component in COMPONENTS})
                row["sum"] = 3 - index if task == "panel_h3" else index + 1
                component_values.append(row)
        return pd.DataFrame(component_values)

    def test_rank_within_tasks_and_preserve_distinct_averaging_orders(self):
        tasks, means = prepare_ranks(self.inputs())
        matrices, by_task, pairs = compute_concordance(tasks, means)
        self.assertEqual(list(matrices["spearman"].columns), list(COMPONENTS))
        self.assertNotIn("mae", tasks.columns)
        self.assertNotIn("mae", means.columns)
        self.assertTrue((means["sum"] == 2).all())
        self.assertTrue(np.isnan(matrices["spearman"].loc["item", "sum"]))
        pair = pairs.loc[(pairs.component_a == "item") & (pairs.component_b == "sum")].iloc[0]
        self.assertAlmostEqual(pair.spearman_task_mean, 0)
        self.assertAlmostEqual(pair.spearman_task_std, np.sqrt(2))
        self.assertEqual(pair.spearman_valid_tasks, 2)
        self.assertEqual(pair.spearman_task_min, -1)
        self.assertEqual(pair.spearman_task_max, 1)
        self.assertEqual(len(by_task), 2 * (len(COMPONENTS) * (len(COMPONENTS) - 1) // 2))
        self.assertNotIn("sparse", tasks.columns)

    def test_undefined_tasks_are_counted_and_do_not_artificially_reduce_dispersion(self):
        component_values = self.inputs()
        component_values.loc[component_values.dataset == "panel_h3", "sum"] = 5
        tasks, means = prepare_ranks(component_values)
        _, _, pairs = compute_concordance(tasks, means)
        pair = pairs.loc[(pairs.component_a == "item") & (pairs.component_b == "sum")].iloc[0]
        self.assertEqual(pair.spearman_valid_tasks, 1)
        self.assertEqual(pair.n_tasks, 2)
        self.assertAlmostEqual(pair.spearman_task_mean, 1)
        self.assertTrue(np.isnan(pair.spearman_task_std))

    def test_ties_and_reversed_rankings(self):
        ranks = pd.DataFrame({"x": [1, 2.5, 2.5, 4], "y": [4, 2.5, 2.5, 1],
                              "z": [1, 2.5, 2.5, 4]})
        for method in ("spearman", "kendall"):
            matrix = correlation_matrix(ranks, method)
            self.assertAlmostEqual(matrix.loc["x", "y"], -1)
            self.assertAlmostEqual(matrix.loc["x", "z"], 1)
            pd.testing.assert_frame_equal(matrix, matrix.T)

    def test_constant_rankings_are_undefined_including_diagonal(self):
        ranks = pd.DataFrame({"constant": [2, 2, 2], "x": [1, 2, 3]})
        for method in ("spearman", "kendall"):
            matrix = correlation_matrix(ranks, method)
            self.assertTrue(matrix.loc["constant"].isna().all())
            self.assertEqual(matrix.loc["x", "x"], 1)

    def test_missing_duplicate_invalid_components_and_mismatched_coverage_fail(self):
        component_values = self.inputs()
        broken_inputs = [component_values.iloc[:-1], pd.concat([component_values, component_values.iloc[:1]])]
        for column, value in (("item", np.nan), ("item", np.inf),
                              ("item", -1), ("matched_rows", 5), ("n_series", 1)):
            broken = component_values.copy()
            broken.loc[0, column] = value
            broken_inputs.append(broken)
        for broken in broken_inputs:
            with self.subTest(broken=broken.head().to_dict()), self.assertRaises(ValueError):
                prepare_ranks(broken)

    def test_item_rank_uses_each_task_not_raw_mean_values(self):
        component_values = self.inputs()
        component_values.loc[component_values.dataset == "panel_h3", "item"] = [1, 2, 1000]
        component_values.loc[component_values.dataset == "panel_h6", "item"] = [100, 2, 1]
        tasks, means = prepare_ranks(component_values)
        self.assertTrue((means.item == 2).all())
        self.assertEqual(tasks.item.tolist(), [1, 2, 3, 3, 2, 1])

    def test_nonfinite_rank_input_is_not_pairwise_deleted(self):
        with self.assertRaises(ValueError):
            correlation_matrix(pd.DataFrame({"x": [1, 2, np.nan]}), "spearman")


if __name__ == "__main__":
    unittest.main()
