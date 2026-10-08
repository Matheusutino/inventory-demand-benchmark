"""Run with .venvs/darts/bin/python -m unittest discover -s tests -v."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast.models import CrostonClassic, CrostonSBA, TSB

from src.analysis.analyze_predictions import evaluate_prediction_file, load_ground_truth
from src.models.intermittent_baselines import (
    forecast_values,
    parameter_candidates,
    run_model,
    select_parameters,
    to_monthly_panel,
)


ROOT = Path(__file__).resolve().parents[1]


def training_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24, freq="MS").as_unit("us"),
        "SKU_A": [0, 0, 8, 0, 0, 0, 12, 0, 0, 7, 0, 0] * 2,
        "SKU_B": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] * 2,
        "ZERO": np.zeros(24),
    })


class IntermittentBaselinesTests(unittest.TestCase):
    def test_zero_and_single_event_skus_are_retained(self):
        panel = pd.DataFrame({"zero": np.zeros(18), "single": [0] * 17 + [9]})
        for name in ("croston", "sba", "tsb"):
            with self.subTest(model=name):
                values = forecast_values(name, parameter_candidates(name)[0], panel, 6)
                self.assertEqual(values.shape, (6, 2))
                np.testing.assert_array_equal(values[:, 0], 0)
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values[:, 1] > 0).all())

    def test_sba_corrects_croston_and_tsb_reacts_to_observed_zeros(self):
        panel = pd.DataFrame({"sku": [0, 3, 0, 8, 0, 0, 2, 0, 4]})
        croston = forecast_values("croston", {"alpha": 0.1}, panel, 3)
        sba = forecast_values("sba", {"alpha": 0.1}, panel, 3)
        np.testing.assert_allclose(sba, 0.95 * croston)
        extended = pd.concat([panel, pd.DataFrame({"sku": np.zeros(6)})], ignore_index=True)
        params = {"alpha_d": 0.1, "alpha_p": 0.2}
        before = forecast_values("tsb", params, panel, 3)
        after = forecast_values("tsb", params, extended, 3)
        np.testing.assert_allclose(after, before * 0.8 ** 6)
        np.testing.assert_allclose(forecast_values("croston", {"alpha": 0.1}, extended, 3), croston)

    def test_selection_uses_only_training_prefix_and_one_shared_configuration(self):
        wide = training_frame().set_index("date")
        horizon = 6
        history, truth = wide.iloc[:-horizon], wide.iloc[-horizon:].to_numpy()
        expected = []
        for params in parameter_candidates("tsb"):
            predictions = np.column_stack([
                TSB(**params).forecast(y=history[sku].to_numpy(dtype=float), h=horizon)["mean"]
                for sku in history.columns
            ])
            expected.append(np.mean(np.abs(truth - predictions)))
        with contextlib.redirect_stdout(io.StringIO()):
            params, score = select_parameters("tsb", wide, horizon, "mae")
        self.assertEqual(params, parameter_candidates("tsb")[int(np.argmin(expected))])
        self.assertAlmostEqual(score, min(expected))

    def test_incomplete_invalid_or_nonmonthly_history_is_rejected(self):
        valid = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "id": "A", "target": [0.0, 2.0, 0.0],
        })
        for bad_value in (np.nan, np.inf, -1):
            with self.subTest(value=bad_value):
                bad = valid.copy()
                bad.loc[1, "target"] = bad_value
                with self.assertRaises(ValueError):
                    to_monthly_panel(bad)
        with self.assertRaises(ValueError):
            to_monthly_panel(valid.drop(index=1))
        with self.assertRaises(ValueError):
            to_monthly_panel(pd.concat([valid, valid.iloc[:1]]))

    def test_parquet_timestamp_precision_does_not_change_monthly_alignment(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="MS").as_unit("us"),
            "id": "A", "target": [0.0, 2.0, 0.0],
        })
        panel = to_monthly_panel(frame)
        self.assertTrue(panel.index.equals(pd.date_range("2020-01-01", periods=3, freq="MS")))

    def test_runner_outputs_h3_h6_refits_and_matches_existing_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "datasets"
            output_dir = Path(tmp) / "predictions"
            data_dir.mkdir()
            training = training_frame()
            for horizon in (3, 6):
                name = f"Synthetic_h{horizon}"
                training.to_parquet(data_dir / f"{name}_train.parquet", index=False)
                pd.DataFrame({
                    "date": pd.date_range("2022-01-01", periods=horizon, freq="MS"),
                    "SKU_A": np.zeros(horizon), "SKU_B": np.full(horizon, 4),
                    "ZERO": np.zeros(horizon),
                }).to_parquet(data_dir / f"{name}_test.parquet", index=False)

            result = subprocess.run([
                sys.executable, "scripts/run_models.py", "--models", "croston", "sba", "tsb",
                "--data-dir", str(data_dir), "--output-dir", str(output_dir), "--stop-on-error",
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            truth = load_ground_truth(str(data_dir))
            for name, budget in (("croston", 1), ("sba", 1), ("tsb", 9)):
                with self.subTest(model=name):
                    path = output_dir / f"statsforecast_{name}_all_datasets_predictions.parquet"
                    predictions = pd.read_parquet(path)
                    self.assertEqual(list(predictions.columns), ["dataset", "id", "timestamp", "q50"])
                    self.assertEqual(len(predictions), 27)
                    self.assertFalse(predictions.duplicated(["dataset", "id", "timestamp"]).any())
                    np.testing.assert_array_equal(predictions.loc[predictions.id == "ZERO", "q50"], 0)
                    parameters = json.loads((output_dir / f"statsforecast_{name}_best_hyperparameters.json").read_text())
                    for selection in parameters:
                        self.assertEqual(selection["num_candidates"], budget)
                        self.assertEqual(selection["validation_metric"], "mae")
                        horizon = int(selection["dataset"][-1])
                        for sku in ("SKU_A", "SKU_B"):
                            library_model = {
                                "croston": CrostonClassic, "sba": CrostonSBA,
                                "tsb": lambda: TSB(selection["alpha_d"], selection["alpha_p"]),
                            }[name]()
                            expected = library_model.forecast(y=training[sku].to_numpy(dtype=float), h=horizon)["mean"]
                            actual = predictions.loc[(predictions.dataset == selection["dataset"]) & (predictions.id == sku), "q50"]
                            np.testing.assert_allclose(actual, expected)
                    metrics, aggregate = evaluate_prediction_file(path, truth)
                    self.assertEqual(metrics.n_points.sum(), 27)
                    self.assertTrue(np.isfinite(aggregate.mae_mean).all())

            # Test demand must never affect parameter selection or forecasts.
            before = pd.read_parquet(output_dir / "statsforecast_tsb_all_datasets_predictions.parquet")
            before_params = (output_dir / "statsforecast_tsb_best_hyperparameters.json").read_text()
            for horizon in (3, 6):
                path = data_dir / f"Synthetic_h{horizon}_test.parquet"
                test = pd.read_parquet(path)
                test["SKU_A"] = 9999
                test.to_parquet(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_model("tsb", str(data_dir), str(output_dir), None, "mae"), 0)
            pd.testing.assert_frame_equal(before, pd.read_parquet(output_dir / "statsforecast_tsb_all_datasets_predictions.parquet"))
            self.assertEqual(before_params, (output_dir / "statsforecast_tsb_best_hyperparameters.json").read_text())

    def test_unknown_dataset_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Datasets não encontrados"):
                run_model("croston", tmp, tmp, ["missing"], "mae")


if __name__ == "__main__":
    unittest.main()
