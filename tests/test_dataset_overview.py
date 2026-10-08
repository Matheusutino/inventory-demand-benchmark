"""Check descriptive statistics and reconstruction of nested horizon files."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.plot_dataset_overview import (
    Panel,
    aggregate_statistics,
    intermittency_statistics,
    load_panels,
)


class DatasetOverviewTests(unittest.TestCase):
    def test_positive_size_variance_and_unclassifiable_skus_use_train_only(self):
        dates = pd.date_range("2020-01-01", periods=9, freq="MS")
        full = pd.DataFrame({
            "intermittent": [0, 2, 0, 4, 0, 6, 9999, 9999, 9999],
            "smooth": [3] * 9,
            "zero": [0] * 6 + [9999] * 3,
            "single": [0, 0, 0, 0, 0, 3, 9999, 9999, 9999],
        }, index=dates)
        stats = intermittency_statistics(Panel("Synthetic", full, {3: dates[6]})).set_index("id")
        self.assertEqual(stats.loc["intermittent", "adi"], 2)
        self.assertEqual(stats.loc["intermittent", "cv2_positive"], 0.25)
        self.assertEqual(stats.loc["intermittent", "category"], "intermittent")
        self.assertEqual(stats.loc["smooth", "category"], "smooth")
        self.assertEqual(stats.loc["zero", "category"], "no_demand")
        self.assertEqual(stats.loc["single", "category"], "insufficient_history")
        self.assertTrue(np.isnan(stats.loc["single", "cv2_positive"]))

    def test_normalization_uses_the_common_training_window(self):
        dates = pd.date_range("2020-01-01", periods=9, freq="MS")
        full = pd.DataFrame({"sku": [1, 2, 3, 100, 100, 100, 100, 100, 100]}, index=dates)
        panel = Panel("Synthetic", full, {3: dates[6], 6: dates[3]})
        stats = aggregate_statistics(panel, "index")
        np.testing.assert_allclose(stats.demand[:3], [50, 100, 150])
        np.testing.assert_allclose(stats.demand[3:], 5000)
        self.assertEqual(stats.is_test_h6.sum(), 6)
        self.assertEqual(stats.is_test_h3.sum(), 3)

    def test_horizon_files_are_deduplicated_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=9, freq="MS"),
                "sku": range(9),
            })
            for horizon in (3, 6):
                train = full.iloc[:-horizon].copy()
                if horizon == 3:
                    train["date"] = train.date.dt.as_unit("us")
                train.to_parquet(root / f"Synthetic_h{horizon}_train.parquet", index=False)
                full.iloc[-horizon:].to_parquet(root / f"Synthetic_h{horizon}_test.parquet", index=False)
            panels = load_panels(tmp)
            self.assertEqual(len(panels), 1)
            self.assertEqual(len(panels[0].full), 9)
            self.assertEqual(len(panels[0].common_train), 3)
            train_path = root / "Synthetic_h3_train.parquet"
            conflicting = pd.read_parquet(train_path)
            conflicting.loc[0, "sku"] = 999
            conflicting.to_parquet(train_path, index=False)
            with self.assertRaisesRegex(ValueError, "divergentes"):
                load_panels(tmp)


if __name__ == "__main__":
    unittest.main()
