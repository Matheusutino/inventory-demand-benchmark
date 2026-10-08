"""Verify revised formulas, training-only scales, eligibility and downstream scope."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.forecast_evaluation import (
    DEFAULT_EVALUATION_CONFIG, aligned_tensors, compute_components, compute_winners,
    evaluate_models, history_statistics,
)
from src.analysis.evaluate_inventory_framework import save_history_outputs
from src.analysis.cd_diagram import build_mae_cd_performance
from src.analysis.inventory_framework import COMPONENTS, EXCLUDED_COMPONENTS
from src.analysis.plot_inventory_component_rank_heatmap import compute_component_ranks
from src.evaluation.inventory import InventoryAwareEvaluator


class InventoryFrameworkTests(unittest.TestCase):
    def setUp(self):
        # One variable SKU, one positive constant SKU and one all-zero history.
        self.history = torch.tensor([[[0., 4., 0.], [2., 4., 0.], [0., 4., 0.]]], dtype=torch.float64)
        self.y = torch.tensor([[[0., 0., 0.], [2., 4., 0.]]], dtype=torch.float64)
        self.pred = torch.tensor([[[2., 8., 3.], [4., 2., 0.]]], dtype=torch.float64)

    def terms(self, pred=None, y=None, history=None):
        return compute_components(
            self.pred if pred is None else pred, self.y if y is None else y,
            DEFAULT_EVALUATION_CONFIG, self.history if history is None else history,
        )

    def test_raw_evaluation_exports_nine_components_and_exact_new_formulas(self):
        terms = self.terms()
        self.assertEqual(tuple(terms), COMPONENTS)
        self.assertEqual(len(terms), 9)
        self.assertFalse(EXCLUDED_COMPONENTS.intersection(terms))
        self.assertAlmostEqual(terms['item'], 17 / 6)
        self.assertAlmostEqual(terms['scaled'], 1.)
        self.assertAlmostEqual(terms['zero'], 2.5)
        self.assertAlmostEqual(terms['cap'], 1.5)
        # Keep the existing allocation, sum, JSD, pinball and unanchored delta formulas.
        residual = self.pred - self.y
        self.assertAlmostEqual(terms['alloc'], float(((residual - residual.mean(-1, keepdim=True)) ** 2).mean()))
        self.assertAlmostEqual(terms['asym'], float(torch.maximum(-.7 * residual, .3 * residual).mean()))
        self.assertAlmostEqual(terms['delta'], float(((self.pred.diff(dim=1) - self.y.diff(dim=1)) ** 2).mean()))
        self.assertTrue(0 <= terms['share'] <= np.log(2))
        totals = self.y.sum(-1)
        expected_sum = (((self.pred.sum(-1) - totals) / totals.abs().clamp(min=1e-6)) ** 2).mean()
        self.assertAlmostEqual(terms['sum'], float(expected_sum))

    def test_perfect_forecasts_and_zero_demand(self):
        perfect = self.terms(pred=self.y)
        for component, value in perfect.items():
            with self.subTest(component=component):
                self.assertAlmostEqual(value, 0., places=12)
        zeros = torch.zeros_like(self.y)
        terms = self.terms(pred=zeros, y=zeros)
        self.assertTrue(all(value == 0 for value in terms.values()))
        # An empty eligible set is undefined, not a fabricated perfect score.
        undefined = self.terms(pred=zeros, y=zeros, history=torch.zeros_like(self.history))
        self.assertTrue(np.isnan(undefined['scaled']))
        self.assertTrue(np.isnan(undefined['zero']))
        self.assertEqual(undefined['cap'], 0)
        self.assertTrue(np.isnan(self.terms(y=torch.ones_like(self.y))['zero']))

    def test_asym_penalizes_underprediction_more_and_delta_has_no_history_anchor(self):
        y = torch.full_like(self.y, 2.)
        under = self.terms(pred=y - 1, y=y)
        over = self.terms(pred=y + 1, y=y)
        self.assertAlmostEqual(under['asym'], .7)
        self.assertAlmostEqual(over['asym'], .3)
        self.assertEqual(under['delta'], 0)
        self.assertEqual(over['delta'], 0)
        self.assertEqual(under['item'], over['item'])
        self.assertEqual(under['scaled'], over['scaled'])

    def test_mase_averages_only_variable_skus_and_zero_is_observation_weighted(self):
        # Three eligible zero observations, rather than a mean of two per-SKU means.
        history = torch.tensor([[[0., 0.], [2., 4.], [4., 8.]]], dtype=torch.float64)
        y = torch.tensor([[[0., 0.], [0., 4.]]], dtype=torch.float64)
        pred = torch.tensor([[[2., 8.], [6., 4.]]], dtype=torch.float64)
        terms = self.terms(pred=pred, y=y, history=history)
        self.assertAlmostEqual(terms['zero'], (1 + 2 + 3) / 3)
        self.assertAlmostEqual(terms['scaled'], (1 + 2 + 3 + 0) / 4)
        scaled = self.terms(pred=pred * 10, y=y * 10, history=history * 10)
        self.assertAlmostEqual(scaled['scaled'], terms['scaled'])
        self.assertAlmostEqual(scaled['zero'], terms['zero'])

    def test_signed_zero_formula_is_not_clipped_and_can_be_ranked(self):
        terms = self.terms(pred=-torch.ones_like(self.y))
        self.assertAlmostEqual(terms['zero'], (-1.5 - .25) / 2)
        values = pd.DataFrame({'dataset': ['a', 'a'], 'model': ['A', 'B'], 'zero': [-1., 0.]})
        _, means = compute_component_ranks(values, ['zero'])
        self.assertEqual(means.set_index('model').loc['A', 'zero'], 1.)
        with self.assertRaises(ValueError):
            compute_component_ranks(values.rename(columns={'zero': 'item'}), ['item'])

    def test_history_is_required_and_evaluator_returns_independent_components(self):
        with self.assertRaisesRegex(ValueError, 'history'):
            compute_components(self.pred, self.y, DEFAULT_EVALUATION_CONFIG, None)
        evaluator = InventoryAwareEvaluator(**DEFAULT_EVALUATION_CONFIG)
        values = evaluator.evaluate(self.pred.requires_grad_(), self.y, self.history)
        terms = self.terms()
        self.assertEqual(tuple(values), COMPONENTS)
        for component in COMPONENTS:
            self.assertFalse(values[component].requires_grad)
            self.assertAlmostEqual(float(values[component]), terms[component])

    def frames(self):
        ids = ['variable', 'constant', 'inactive']
        train = pd.DataFrame(self.history[0].numpy(), columns=ids)
        train.insert(0, 'timestamp', pd.date_range('2020-01-01', periods=3, freq='MS'))
        truth = pd.DataFrame(self.y[0].numpy(), columns=ids)
        truth.insert(0, 'timestamp', pd.date_range('2020-04-01', periods=2, freq='MS'))
        return (frame.melt(id_vars='timestamp', var_name='id', value_name='target') for frame in (train, truth))

    def test_eligibility_and_training_scales_are_reported_per_sku_and_task(self):
        train, truth = self.frames()
        _, sku, summary = history_statistics(train, truth)
        sku = sku.set_index('id')
        self.assertEqual(sku.loc['variable', 'mase_scale'], 2.)
        self.assertEqual(sku.loc['constant', 'training_mean'], 4.)
        self.assertEqual(sku.loc['variable', 'cap_limit'], 4.)
        self.assertEqual(summary['mase_excluded_constant_skus'], 2)
        self.assertEqual(summary['zero_excluded_no_demand_skus'], 1)
        self.assertEqual(summary['mase_observations'], 2)
        self.assertEqual(summary['zero_observations'], 2)
        self.assertEqual(summary['zero_excluded_observations'], 2)
        with tempfile.TemporaryDirectory() as tmp:
            save_history_outputs({'a': truth}, {'a': train}, Path(tmp))
            self.assertEqual(len(pd.read_csv(Path(tmp) / 'training_scales_by_sku.csv')), 3)
            self.assertEqual(len(pd.read_csv(Path(tmp) / 'component_coverage.csv')), 1)

    def test_test_values_never_change_mase_scale_zero_denominator_or_cap(self):
        train, truth = self.frames()
        with tempfile.TemporaryDirectory() as tmp:
            prediction = truth.rename(columns={'target': 'q50'})
            prediction['q50'] = 10.
            prediction['dataset'] = 'a'
            path = Path(tmp) / 'A_all_datasets_predictions.parquet'
            prediction.to_parquet(path, index=False)
            original = evaluate_models({'a': truth}, [path], DEFAULT_EVALUATION_CONFIG, {'a': train})
            changed = truth.copy()
            changed.loc[changed.target > 0, 'target'] = 10000.
            after = evaluate_models({'a': changed}, [path], DEFAULT_EVALUATION_CONFIG, {'a': train})
            for field in ('cap', 'zero', 'mase_eligible_skus', 'mase_excluded_constant_skus', 'zero_observations'):
                self.assertEqual(original.loc[0, field], after.loc[0, field])
            _, scales_before, _ = history_statistics(train, truth)
            _, scales_after, _ = history_statistics(train, changed)
            pd.testing.assert_frame_equal(scales_before, scales_after)
            # SKU ordering cannot change metric values.
            shuffled = evaluate_models({'a': truth.sample(frac=1, random_state=1)}, [path], DEFAULT_EVALUATION_CONFIG,
                                       {'a': train.sample(frac=1, random_state=2)})
            pd.testing.assert_frame_equal(original, shuffled)

    def test_cd_diagram_exports_mae_with_its_metric_name_and_original_values(self):
        train, truth = self.frames()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for split, frame in (("train", train), ("test", truth)):
                wide = frame.pivot(index="timestamp", columns="id", values="target")
                wide.reset_index().rename(columns={"timestamp": "date"}).to_parquet(
                    output / f"panel_{split}.parquet", index=False,
                )
            for model, error in (("A", 2.), ("B", 0.)):
                prediction = truth.rename(columns={"target": "q50"}).copy()
                prediction["q50"] += error
                prediction["dataset"] = "panel"
                prediction.to_parquet(output / f"{model}_all_datasets_predictions.parquet", index=False)
            performance, series, scores, ranks = build_mae_cd_performance(tmp, tmp, None)
            for table in (performance, series, scores):
                self.assertIn("mae", table)
                self.assertNotIn("normalized_score", table)
            self.assertEqual(scores.set_index("model").loc["A", "mae"], 2.)
            self.assertEqual(scores.set_index("model").loc["B", "mae"], 0.)
            self.assertEqual(ranks.iloc[0].classifier_name, "B")

    def test_missing_duplicate_nonfinite_or_leaking_history_is_rejected(self):
        train, truth = self.frames()
        for bad in (train.iloc[:-1], pd.concat([train, train.iloc[:1]])):
            with self.assertRaises(ValueError):
                history_statistics(bad, truth)
        leaked = train.copy()
        leaked['timestamp'] += pd.offsets.MonthBegin(4)
        with self.assertRaisesRegex(ValueError, 'teste'):
            history_statistics(leaked, truth)
        pred = truth.rename(columns={'target': 'q50'})
        for bad in (pred.iloc[:-1], pd.concat([pred, pred.iloc[:1]])):
            with self.assertRaises(ValueError):
                aligned_tensors(truth, bad)
        pred.loc[0, 'q50'] = np.inf
        with self.assertRaises(ValueError):
            aligned_tensors(truth, pred)

    def test_timestamp_precision_does_not_change_history_or_prediction_alignment(self):
        train, truth = self.frames()
        history_ns, scales_ns, _ = history_statistics(train, truth)
        train['timestamp'] = train.timestamp.astype('datetime64[us]')
        history_us, scales_us, _ = history_statistics(train, truth)
        self.assertTrue(torch.equal(history_ns, history_us))
        pd.testing.assert_series_equal(scales_ns.mase_scale, scales_us.mase_scale)
        pred = truth.rename(columns={'target': 'q50'})
        pred['timestamp'] = pred.timestamp.astype('datetime64[us]')
        y_hat, y, rows, _, _ = aligned_tensors(truth, pred)
        self.assertTrue(torch.equal(y_hat, y))
        self.assertEqual(rows, len(truth))

    def test_stale_excluded_columns_cannot_reenter_ranks_or_wins(self):
        rows = []
        for model, value in (('A', 1), ('B', 2)):
            rows.append({'dataset': 'task', 'model': model,
                         **{c: value for c in COMPONENTS}, **{c: 3 - value for c in EXCLUDED_COMPONENTS}})
        values = pd.DataFrame(rows)
        ranks, means = compute_component_ranks(values)
        winners, counts, summary = compute_winners(values)
        self.assertEqual(set(ranks.component), set(COMPONENTS))
        self.assertFalse(EXCLUDED_COMPONENTS.intersection(means.columns))
        self.assertEqual(set(winners.component), set(COMPONENTS))
        self.assertEqual(set(counts.component), set(COMPONENTS))
        self.assertFalse(EXCLUDED_COMPONENTS.intersection(summary.columns))
        self.assertNotIn('total_wins', summary)
        self.assertTrue((winners.model == 'A').all())


if __name__ == '__main__':
    unittest.main()
