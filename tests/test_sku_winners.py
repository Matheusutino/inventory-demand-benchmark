"""Check SKU alignment, tied-family credit and the fixed-model-per-SKU oracle."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analysis.analyze_sku_winners import compute_sku_winners, load_sku_mae


class SkuWinnersTests(unittest.TestCase):
    def metrics(self, errors, dataset="Filtros_h3"):
        return pd.DataFrame([
            {"dataset": dataset, "id": str(sku), "model": model, "mae": error}
            for model, values in errors.items() for sku, error in enumerate(values)
        ])

    def test_best_mean_model_can_win_no_skus(self):
        tables = compute_sku_winners(self.metrics({
            "chronos-2": [0, 9], "statsforecast_tsb": [9, 0], "darts_linear": [4, 4],
        }))
        oracle = tables["oracle_by_panel"].iloc[0]
        self.assertEqual(oracle.best_model, "darts_linear")
        self.assertEqual(oracle.best_single_mae, 4)
        self.assertEqual(oracle.best_model_win_share, 0)
        self.assertEqual(oracle.oracle_mae, 0)
        self.assertEqual(oracle.oracle_gain_pct, 100)
        by_category = tables["model_wins_by_category"]
        np.testing.assert_allclose(by_category.groupby("model").fractional_wins.sum(),
                                   tables["model_win_shares"].set_index("model").fractional_wins)

    def test_ties_split_models_and_families_and_do_not_round_near_ties(self):
        metrics = self.metrics({
            "chronos-2": [0, 1], "TiRex": [0, np.nextafter(1., 2.)],
            "statsforecast_tsb": [0, 2], "darts_global_naive_drift": [1, 3],
        })
        tables = compute_sku_winners(metrics)
        model_credits = tables["sku_winners"].query('id == "0"').model_win_credit
        np.testing.assert_allclose(model_credits, [1/3] * 3)
        family_credits = tables["sku_family_winners"].query('id == "0"').family_win_credit
        np.testing.assert_allclose(family_credits, [.5, .5])
        second = tables["sku_winners"].query('id == "1"')
        self.assertEqual(second.model.tolist(), ["chronos-2"])
        for name in ("family_win_shares", "model_win_shares"):
            self.assertAlmostEqual(tables[name].win_share.sum(), 1)
        families = tables["family_win_shares"].set_index("family")
        self.assertEqual(families.loc["foundation", "win_share"], .75)
        self.assertEqual(families.loc["intermittent", "win_share"], .25)

    def test_global_mae_weights_panels_equally_and_keeps_ids_separate(self):
        a = self.metrics({"chronos-2": [0], "statsforecast_tsb": [10]})
        b = self.metrics({"chronos-2": [6, 6, 6], "statsforecast_tsb": [0, 0, 0]}, "Mecanismos_h3")
        h6 = a.copy()
        h6["dataset"] = "Filtros_h6"
        tables = compute_sku_winners(pd.concat([a, b, h6], ignore_index=True))
        overall = tables["oracle_overall"].set_index("horizon")
        self.assertEqual(overall.loc[3, "best_model"], "chronos-2")
        self.assertEqual(overall.loc[3, "best_single_mae"], 3)
        self.assertEqual(overall.loc[3, "oracle_mae"], 0)
        self.assertEqual(overall.loc[3, "best_model_win_share"], .25)
        self.assertEqual(overall.loc[6, "n_skus"], 1)
        self.assertEqual(len(tables["sku_summary"]), 5)

    def test_perfect_forecasts_have_zero_absolute_gain_and_undefined_percentage(self):
        tables = compute_sku_winners(self.metrics({"chronos-2": [0], "statsforecast_tsb": [0]}))
        for key in ("oracle_by_panel", "oracle_overall"):
            self.assertEqual(tables[key].iloc[0].oracle_gain_absolute, 0)
            self.assertTrue(np.isnan(tables[key].iloc[0].oracle_gain_pct))

    def test_volume_uses_realized_demand_and_splits_ties_without_double_counting(self):
        metrics = self.metrics({
            "chronos-2": [0, 2, 0], "TiRex": [0, 3, 0], "darts_global_naive_drift": [1, 0, 0],
        })
        metrics["test_total_demand"] = metrics.id.map({"0": 80, "1": 20, "2": 0})
        tables = compute_sku_winners(metrics)
        families = tables["family_win_shares"].set_index("family")
        self.assertEqual(families.loc["foundation", "win_share"], .5)
        self.assertEqual(families.loc["naive", "win_share"], .5)
        self.assertEqual(families.loc["foundation", "volume_win_share"], .8)
        self.assertEqual(families.loc["naive", "volume_win_share"], .2)
        models = tables["model_win_shares"].set_index("model")
        self.assertEqual(models.loc["chronos-2", "won_test_volume"], 40)
        self.assertEqual(models.loc["TiRex", "won_test_volume"], 40)
        self.assertEqual(models.loc["darts_global_naive_drift", "won_test_volume"], 20)
        for key in ("model", "family"):
            winners = tables[f"sku_{'winners' if key == 'model' else 'family_winners'}"]
            np.testing.assert_allclose(winners.groupby("id")[f"{key}_volume_credit"].sum(), [80, 20, 0])
            self.assertAlmostEqual(tables[f"{key}_win_shares"].volume_win_share.sum(), 1)
        original = compute_sku_winners(metrics.drop(columns="test_total_demand"))
        pd.testing.assert_series_equal(tables["sku_winners"].model_win_credit,
                                       original["sku_winners"].model_win_credit)
        pd.testing.assert_frame_equal(tables["oracle_overall"], original["oracle_overall"])

    def test_global_volume_pools_before_normalizing_and_separates_horizons(self):
        a = self.metrics({"chronos-2": [0], "darts_global_naive_drift": [1]}).assign(test_total_demand=1)
        b = self.metrics({"chronos-2": [1], "darts_global_naive_drift": [0]}, "Mecanismos_h3").assign(test_total_demand=99)
        h6 = a.assign(dataset="Filtros_h6", test_total_demand=7)
        tables = compute_sku_winners(pd.concat([a, b, h6], ignore_index=True))
        overall = tables["family_win_shares_overall"].set_index(["horizon", "family"])
        self.assertEqual(overall.loc[(3, "foundation"), "win_share"], .5)
        self.assertEqual(overall.loc[(3, "foundation"), "volume_win_share"], .01)
        self.assertEqual(overall.loc[(3, "foundation"), "equal_panel_volume_win_share"], .5)
        self.assertEqual(overall.loc[(6, "foundation"), "volume_win_share"], 1)
        self.assertEqual(overall.loc[(6, "foundation"), "total_test_volume"], 7)

    def test_zero_volume_is_undefined_and_excluded_from_equal_panel_volume_average(self):
        zero = self.metrics({"chronos-2": [0], "darts_global_naive_drift": [1]}).assign(test_total_demand=0)
        tables = compute_sku_winners(zero)
        for key in ("model", "family"):
            self.assertTrue(tables[f"{key}_win_shares"].volume_win_share.isna().all())
            self.assertTrue(tables[f"{key}_win_shares_overall"].volume_win_share.isna().all())
            self.assertTrue((tables[f"{key}_win_shares_overall"].n_panels_with_volume == 0).all())
        positive = zero.assign(dataset="Mecanismos_h3", test_total_demand=10)
        overall = compute_sku_winners(pd.concat([zero, positive], ignore_index=True))["family_win_shares_overall"]
        foundation = overall.loc[overall.family == "foundation"].iloc[0]
        self.assertEqual(foundation.equal_panel_volume_win_share, 1)
        self.assertEqual(foundation.n_panels_with_volume, 1)
        self.assertEqual(foundation.n_panels, 2)

    def test_rejects_invalid_or_model_dependent_volumes(self):
        metrics = self.metrics({"chronos-2": [0], "statsforecast_tsb": [1]})
        for volume in (np.nan, np.inf, -1, [1, 2]):
            with self.subTest(volume=volume):
                with self.assertRaises(ValueError):
                    compute_sku_winners(metrics.assign(test_total_demand=volume))

    def test_rejects_invalid_or_unbalanced_metrics(self):
        valid = self.metrics({"chronos-2": [1, 2], "statsforecast_tsb": [2, 1]})
        invalid = [valid.iloc[:-1], pd.concat([valid, valid.iloc[:1]]),
                   valid.assign(mae=np.nan), valid.assign(mae=-1),
                   valid.assign(n_points=2), valid.assign(dataset="Filtros")]
        for frame in invalid:
            with self.subTest(frame=frame.to_dict("records")):
                with self.assertRaises(ValueError):
                    compute_sku_winners(frame)

    def test_saved_forecasts_align_by_timestamp_and_choose_a_fixed_model(self):
        truth = pd.DataFrame({
            "dataset": ["Filtros_h3"] * 3, "id": ["0"] * 3,
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="MS"), "target": [0., 0., 0.],
        })
        with tempfile.TemporaryDirectory() as tmp, patch(
                "src.analysis.analyze_sku_winners.load_ground_truth", return_value=truth):
            root = Path(tmp)
            for model, forecasts in {"chronos-2": [-3., 0., 3.], "statsforecast_tsb": [0., 4., 0.]}.items():
                pred = truth.drop(columns="target").assign(q50=forecasts).iloc[::-1]
                pred.to_parquet(root / f"{model}_all_datasets_predictions.parquet", index=False)
            tables = compute_sku_winners(load_sku_mae("unused", tmp))
            row = tables["oracle_by_panel"].iloc[0]
            self.assertEqual(row.best_model, "statsforecast_tsb")
            self.assertAlmostEqual(row.oracle_mae, 4/3)
            self.assertEqual(row.oracle_gain_pct, 0)
            self.assertEqual(row.zero_test_skus, 1)
            path = root / "chronos-2_all_datasets_predictions.parquet"
            valid_pred = pd.read_parquet(path)
            invalid = [valid_pred.iloc[:-1], pd.concat([valid_pred, valid_pred.iloc[:1]]),
                       valid_pred.assign(q50=np.inf), valid_pred.assign(id="missing")]
            for pred in invalid:
                pred.to_parquet(path, index=False)
                with self.assertRaises(ValueError):
                    load_sku_mae("unused", tmp)


if __name__ == "__main__":
    unittest.main()
