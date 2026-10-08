"""Validate identifier privacy, exact demand preservation and benchmark splits."""

import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.export_anonymized_dataset import checksum, export_dataset, load_tasks


class AnonymizedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "release"
        self.mapping = self.root / "private" / "mapping.json"
        self.originals = {}
        for i, panel in enumerate(("private_panel_A", "private_panel_B")):
            full = pd.DataFrame({
                "date": pd.date_range("2022-01-01", periods=20, freq="MS"),
                "original_shared_sku": np.arange(20, dtype=float) + i / 4,
                f"original_inactive_{i}": np.r_[np.zeros(14), np.arange(6, dtype=float)],
                f"original_constant_{i}": np.full(20, 5.25),
            })
            full.attrs["private_source"] = "raw_secret_metadata"
            for horizon in (3, 6):
                for split, frame in (("train", full.iloc[:-horizon]), ("test", full.iloc[-horizon:])):
                    name = f"{panel}_h{horizon}_{split}"
                    frame.to_parquet(self.source / f"{name}.parquet", index=False)
                    self.originals[name] = frame.copy()
        (self.source / "confidential.xlsx").write_text("Never include source spreadsheets")

    def export(self, output=None):
        return export_dataset(self.source, output or self.output, self.mapping)

    def test_roundtrip_preserves_every_value_date_and_shared_id(self):
        result = self.export()
        self.assertEqual(result["n_skus"], 5)  # Shared SKU has one global anonymous ID.
        self.assertEqual(result["n_unique_observations"], 120)
        aliases = json.loads(self.mapping.read_text())["sku_ids"]
        tasks = pd.read_csv(self.output / "tasks.csv")
        for row in tasks.itertuples(index=False):
            panel = ("private_panel_A", "private_panel_B")[int(row.panel[-2:]) - 1]
            for split in ("train", "test"):
                original = self.originals[f"{panel}_h{row.horizon}_{split}"]
                expected = original.set_index("date").rename(columns=aliases).sort_index(axis=1)
                expected.attrs = {}
                restored = pd.read_parquet(self.output / "datasets" / f"{row.dataset}_{split}.parquet").set_index("date")
                pd.testing.assert_frame_equal(expected, restored)
        series = pd.read_csv(self.output / "monthly_demand.csv", parse_dates=["date"], float_precision="round_trip")
        self.assertFalse(series.duplicated(["panel", "id", "date"]).any())
        self.assertEqual(len(series), 120)
        for panel in series.panel.unique():
            pivoted = series.loc[series.panel == panel].pivot(index="date", columns="id", values="demand")
            train = pd.read_parquet(self.output / "datasets" / f"{panel}_h6_train.parquet").set_index("date")
            test = pd.read_parquet(self.output / "datasets" / f"{panel}_h6_test.parquet").set_index("date")
            pivoted.columns.name = None
            pd.testing.assert_frame_equal(pd.concat([train, test]), pivoted)

    def test_archive_contains_only_public_files_without_raw_identifiers_or_metadata(self):
        result = self.export()
        manifest = json.loads((self.output / "manifest.json").read_text())
        with zipfile.ZipFile(result["archive"]) as archive:
            self.assertEqual(archive.testzip(), None)
            self.assertEqual(set(archive.namelist()), {f"release/{name}" for name in [*manifest["files"], "manifest.json"]})
            for name in archive.namelist():
                content = archive.read(name)
                for raw in (b"original_", b"private_panel_", b"raw_secret_metadata", b"confidential.xlsx"):
                    self.assertNotIn(raw, content)
        for name, meta in manifest["files"].items():
            path = self.output / name
            self.assertEqual(meta["sha256"], checksum(path))
            self.assertEqual(meta["bytes"], path.stat().st_size)
            if path.suffix == ".parquet":
                self.assertIsNone(pq.read_schema(path).metadata)
        self.assertEqual(checksum(Path(result["archive"])), result["sha256"])
        self.assertEqual(stat.S_IMODE(self.mapping.stat().st_mode), 0o600)

    def test_reexport_reuses_mapping_and_identical_public_file_contents(self):
        self.export()
        private_before = self.mapping.read_bytes()
        output2 = self.root / "release_again"
        self.export(output2)
        self.assertEqual(self.mapping.read_bytes(), private_before)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (output2 / path.relative_to(self.output)).read_bytes())
        with self.assertRaises(FileExistsError):
            self.export()

    def test_export_is_compatible_with_benchmark_loader(self):
        self.export()
        datasets = discover_datasets(str(self.output / "datasets"))
        self.assertEqual(len(datasets), 4)
        for name, paths in datasets.items():
            loader = DataLoader()
            train, test, horizon = loader.load_dataset(paths["train"], paths["test"])
            self.assertEqual(horizon, int(name[-1]))
            self.assertEqual(set(train.id), set(test.id))
            self.assertEqual(len(test), 3 * horizon)
            self.assertTrue(np.isfinite(test.target).all())

    def test_private_mapping_cannot_be_packaged_and_source_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "outside the public release"):
            export_dataset(self.source, self.output, self.output / "mapping.json")
        with self.assertRaisesRegex(ValueError, "separate from source"):
            export_dataset(self.source, self.source / "release", self.mapping)
        self.assertFalse(self.mapping.exists())
        self.assertFalse(self.output.exists())

    def test_missing_pairs_and_divergent_horizons_are_rejected_before_export(self):
        path = self.source / "private_panel_A_h6_test.parquet"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing train/test pair"):
            self.export()
        path.write_bytes(original)
        frame = pd.read_parquet(path)
        frame.iloc[0, 1] += 1
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "H3/H6 histories differ"):
            self.export()
        self.assertFalse(self.mapping.exists())
        self.assertFalse(self.output.exists())

    def test_nonfinite_negative_and_irregular_history_are_rejected(self):
        path = self.source / "private_panel_A_h3_train.parquet"
        original = pd.read_parquet(path)
        for value in (np.nan, np.inf, -1):
            changed = original.copy()
            changed.iloc[0, 1] = value
            changed.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                load_tasks(self.source)
        changed = original.copy()
        changed.iloc[0, 0] += pd.Timedelta(days=1)
        changed.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "consecutive month-start"):
            load_tasks(self.source)


if __name__ == "__main__":
    unittest.main()
