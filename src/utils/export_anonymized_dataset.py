"""Prepare a dataset release with persistent, randomly assigned SKU aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PANEL_LABELS = {
    "Compressores Herméticos": "Hermetic Compressors",
    "Filtros": "Filters",
    "Mecanismos": "Mechanisms",
    "Motores Ventiladores": "Fan Motors",
    "Placas de Controle": "Control Boards",
}
ALIAS_PATTERN = re.compile(r"SKU_[0-9a-f]{16}\Z")


@dataclass
class Task:
    panel: str
    horizon: int
    train: pd.DataFrame
    test: pd.DataFrame


def read_panel(path: Path) -> pd.DataFrame:
    """Validate monthly, nonnegative demand without retaining source metadata."""
    raw = pd.read_parquet(path)
    if "date" not in raw or len(raw.columns) < 2:
        raise ValueError(f"Expected date and SKU columns: {path.name}")
    frame = raw.set_index("date")
    frame.columns = frame.columns.astype(str)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).as_unit("ns")
    if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"Empty panel or duplicate dates/SKUs: {path.name}")
    if frame.index.hasnans or frame.index.tz is not None:
        raise ValueError(f"Invalid or timezone-aware dates: {path.name}")
    frame = frame.sort_index().reindex(sorted(frame.columns), axis=1)
    expected = pd.date_range(frame.index[0], periods=len(frame), freq="MS")
    if not frame.index.equals(expected):
        raise ValueError(f"Expected consecutive month-start dates: {path.name}")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"Demand must be numeric: {path.name}")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Demand must be finite and nonnegative: {path.name}")
    # Construct fresh frames so attrs, index metadata and original names cannot leak.
    return pd.DataFrame(values.copy(), index=frame.index.copy(), columns=frame.columns.copy())


def load_tasks(source: Path) -> list[Task]:
    """Require complete H3/H6 pairs and identical reconstructed histories."""
    files = {path.name: path for path in source.glob("*.parquet")}
    if not files:
        raise ValueError("No Parquet datasets found")
    names = set()
    for name in files:
        match = re.fullmatch(r"(.+)_h(3|6)_(train|test)\.parquet", name)
        if not match:
            raise ValueError(f"Unexpected dataset filename: {name}")
        names.add(match[1])
    order = list(PANEL_LABELS)
    tasks = []
    for panel in sorted(names, key=lambda name: (order.index(name) if name in order else len(order), name)):
        history = None
        for horizon in (3, 6):
            paths = [f"{panel}_h{horizon}_{split}.parquet" for split in ("train", "test")]
            if any(path not in files for path in paths):
                raise ValueError(f"Missing train/test pair: {panel}, H{horizon}")
            train, test = [read_panel(files[path]) for path in paths]
            if not train.columns.equals(test.columns) or len(test) != horizon:
                raise ValueError(f"Inconsistent SKU columns or horizon: {panel}, H{horizon}")
            if test.index[0] != train.index[-1] + pd.offsets.MonthBegin(1):
                raise ValueError(f"Train/test gap or overlap: {panel}, H{horizon}")
            full = pd.concat([train, test])
            if history is not None and not history.equals(full):
                raise ValueError(f"H3/H6 histories differ: {panel}")
            history = full
            tasks.append(Task(panel, horizon, train, test))
    return tasks


def private_aliases(skus: set[str], path: Path) -> dict[str, str]:
    """Persist random aliases privately; never derive public IDs from raw SKU hashes."""
    aliases = {}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("version") != 1 or not isinstance(saved.get("sku_ids"), dict):
            raise ValueError("Invalid private mapping format")
        aliases = saved["sku_ids"]
    if any(not isinstance(value, str) or not ALIAS_PATTERN.fullmatch(value) for value in aliases.values()):
        raise ValueError("Invalid anonymous SKU in private mapping")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("Duplicate anonymous SKU in private mapping")
    used = set(aliases.values())
    for sku in sorted(skus - set(aliases)):
        alias = f"SKU_{secrets.token_hex(8)}"
        while alias in used or alias in skus:
            alias = f"SKU_{secrets.token_hex(8)}"
        aliases[sku] = alias
        used.add(alias)
    if set(aliases.values()) & skus:
        raise ValueError("An anonymous ID matches a source SKU")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".sku_mapping_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "sku_ids": aliases}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {sku: aliases[sku] for sku in skus}


def anonymize(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Replace identifiers, drop metadata, and sort by opaque anonymous ID."""
    anonymous = pd.DataFrame(
        frame.to_numpy(copy=True),
        index=pd.DatetimeIndex(frame.index.to_numpy(copy=True), name="date"),
        columns=[aliases[str(sku)] for sku in frame.columns],
    )
    return anonymous.reindex(sorted(anonymous.columns), axis=1)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write only explicit columns, without pandas or inherited schema metadata."""
    table = pa.Table.from_pandas(frame.reset_index(), preserve_index=False)
    table = table.replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd")


def checksum(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_card(panels: pd.DataFrame, tasks: pd.DataFrame, counts: dict) -> str:
    """Describe the actual release, its splits and identifier transformation."""
    panel_rows = "\n".join(
        f"| {row.panel} | {row.label} | {row.n_skus} | {row.n_months} | {row.zero_demand_common_train_skus} |"
        for row in panels.itertuples(index=False)
    )
    split_rows = "\n".join(
        f"| {row.dataset} | {row.n_train_months} | {row.train_start} to {row.train_end} | {row.test_start} to {row.test_end} |"
        for row in tasks.itertuples(index=False)
    )
    return f"""# Inventory-Aware Evaluation of Forecasting Models for Sparse Multi-SKU Demand — Dataset

## Release status and scope

Prepared locally for author review. Public repository, DOI, citation and dataset
license have not been specified. No reuse
license is granted by this draft. This package is not a publication record.

This benchmark contains {counts['n_skus']} distinct SKU series in
{counts['n_panels']} product panels, {counts['n_tasks']} forecasting tasks and
{counts['n_unique_observations']:,} unique SKU/month demand observations.
The dates, numeric demand, zero-demand observations and original train/test
boundaries are preserved. No forecasts or source spreadsheets are included.

## Panels

| Panel | Product family | SKUs | Months | No demand in common training history |
|---|---|---:|---:|---:|
{panel_rows}

Common training history ends before either test window (the H6 training split).
SKU counts include inactive products. Missing values are not introduced or
imputed; monthly zero demand is explicitly stored as zero. Values represent
monthly demand in the source benchmark's original quantity units, without
scaling, clipping, rounding or aggregation across SKUs.

## Files and schema

- `datasets/<panel>_h<H>_train.parquet` and matching `_test.parquet`: wide-format
  benchmark inputs, with `date` and one numeric column per anonymous SKU.
- `monthly_demand.csv`: one full history per panel, without duplicating H3/H6;
  columns `panel`, `id`, `date`, `demand`. Dates are ISO calendar month starts.
- `panels.csv`: product-family labels, series/month counts, date coverage, and
  zero-demand SKU count in the common training history.
- `tasks.csv`: exact split dates, horizon, SKU/month counts, training-only
  zero-demand counts and training-only constant-series counts for each task.
- `manifest.json`: version, counts and SHA-256/size of every other public file.

Each `(panel, id, date)` in the CSV is unique. Demand is finite and nonnegative.
Parquet dates are timezone-free timestamps. IDs have the form `SKU_<16 hex
digits>` and carry no product meaning. `panel_01`, etc. identify product panels;
English family labels are provided separately in `panels.csv`.

## Forecasting protocol

| Task | Training months | Training interval | Test interval |
|---|---:|---|---|
{split_rows}

H3 and H6 reconstruct the same full history within each panel. Their test
windows overlap; they are separate forecasting tasks, not independent datasets.
Fit models, select hyperparameters and compute historical normalization scales
using only the training split of the task being evaluated. H6 test observations
that fall in H3 training are legitimately available at the later H3 origin.

## Identifier anonymization

Original SKU identifiers are replaced by randomly assigned opaque codes. A
given source SKU always receives the same code across train/test, horizons and
panels. Public columns are sorted by anonymous code, not by source SKU.
The correspondence is stored privately outside this package and is excluded
from the ZIP, metadata and checksums. Parquet files are rebuilt without source
schema metadata. Source filenames and original SKU labels are not included.

This is identifier pseudonymization: the underlying time series, calendar dates,
volumes and product-family labels are retained. It does not establish anonymity
against auxiliary information about those demand patterns.

## Loading the data

Requires Python, pandas and pyarrow for Parquet (CSV needs only pandas).
Run from the extracted release directory:

```python
import pandas as pd

tasks = pd.read_csv('tasks.csv')
task = tasks.iloc[0]['dataset']
train = pd.read_parquet(f'datasets/{{task}}_train.parquet').set_index('date')
test = pd.read_parquet(f'datasets/{{task}}_test.parquet').set_index('date')
assert list(train.columns) == list(test.columns)

series = pd.read_csv('monthly_demand.csv', parse_dates=['date'])
```

The paired Parquet files are also compatible with this benchmark's DataLoader.
Use the task identifiers in `tasks.csv` rather than private source names.

## Reproducibility and provenance

The release is exported by `python -m src.utils.export_anonymized_dataset` from
the benchmark's existing paired datasets. No model is trained or run by the
export. Retaining the private mapping produces the same aliases in later
exports; losing it prevents correspondence with the original SKU identifiers.
Before publication, authors must supply the final license, provenance, citation
and hosting identifier in place of this draft's pending fields.
"""


def export_dataset(source: Path, destination: Path, mapping_path: Path) -> dict:
    """Validate and export a new public directory plus a whitelisted ZIP archive."""
    source, destination, mapping_path = [path.resolve() for path in (source, destination, mapping_path)]
    archive = destination.parent / f"{destination.name}.zip"
    sidecar = archive.with_suffix(".zip.sha256")
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Release directory must be separate from source datasets")
    if mapping_path.is_relative_to(destination) or mapping_path in (archive, sidecar):
        raise ValueError("Private mapping must be outside the public release")
    if any(path.exists() for path in (destination, archive, sidecar)):
        raise FileExistsError("Release output exists; choose a new destination")
    tasks = load_tasks(source)
    aliases = private_aliases({str(sku) for task in tasks for sku in task.train.columns}, mapping_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".dataset_release_", dir=destination.parent))
    archive_temp = stage / "release.zip"
    try:
        public = stage / "public"
        (public / "datasets").mkdir(parents=True)
        panel_codes = {name: f"panel_{i:02d}" for i, name in enumerate(dict.fromkeys(task.panel for task in tasks), 1)}
        task_rows, panel_rows, full_frames = [], [], []
        for task in tasks:
            code = panel_codes[task.panel]
            name = f"{code}_h{task.horizon}"
            train, test = [anonymize(frame, aliases) for frame in (task.train, task.test)]
            for split, frame in (("train", train), ("test", test)):
                path = public / "datasets" / f"{name}_{split}.parquet"
                write_parquet(frame, path)
                restored = pd.read_parquet(path).set_index("date")
                pd.testing.assert_frame_equal(frame, restored, check_freq=False)
                if pq.read_schema(path).metadata:
                    raise ValueError("Unexpected metadata in public Parquet")
            task_rows.append({
                "dataset": name, "panel": code, "horizon": task.horizon, "n_skus": len(train.columns),
                "n_train_months": len(train), "n_test_months": len(test),
                "train_start": str(train.index[0].date()), "train_end": str(train.index[-1].date()),
                "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
                "zero_demand_train_skus": int((train.sum(axis=0) == 0).sum()),
                "constant_train_skus": int((train.nunique(axis=0) == 1).sum()),
            })
            if task.horizon == 6:
                full = pd.concat([train, test])
                long = full.reset_index().melt(id_vars="date", var_name="id", value_name="demand")
                long.insert(0, "panel", code)
                full_frames.append(long[["panel", "id", "date", "demand"]])
                panel_rows.append({
                    "panel": code, "label": PANEL_LABELS.get(task.panel, "Product panel"),
                    "n_skus": len(full.columns), "n_months": len(full),
                    "start": str(full.index[0].date()), "end": str(full.index[-1].date()),
                    "zero_demand_common_train_skus": int((train.sum(axis=0) == 0).sum()),
                })
        panel_table, task_table = pd.DataFrame(panel_rows), pd.DataFrame(task_rows)
        panel_table.to_csv(public / "panels.csv", index=False)
        task_table.to_csv(public / "tasks.csv", index=False)
        series = pd.concat(full_frames, ignore_index=True)
        series.to_csv(public / "monthly_demand.csv", index=False, date_format="%Y-%m-%d")
        csv_roundtrip = pd.read_csv(public / "monthly_demand.csv", parse_dates=["date"], float_precision="round_trip")
        pd.testing.assert_frame_equal(series, csv_roundtrip)
        counts = {
            "n_panels": len(panel_rows), "n_tasks": len(tasks), "n_skus": len(aliases),
            "n_unique_observations": len(series), "n_parquet_files": 2 * len(tasks),
        }
        (public / "README.md").write_text(dataset_card(panel_table, task_table, counts), encoding="utf-8")
        manifest = {"version": 1, "license": None, "publication_status": "draft", "counts": counts, "files": {}}
        for path in sorted(public.rglob("*")):
            if path.is_file():
                manifest["files"][path.relative_to(public).as_posix()] = {"sha256": checksum(path), "bytes": path.stat().st_size}
        (public / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Explicit file list prevents the private mapping or staging files being packed.
        members = [*manifest["files"], "manifest.json"]
        with zipfile.ZipFile(archive_temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for member in sorted(members):
                info = zipfile.ZipInfo(f"{destination.name}/{member}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, (public / member).read_bytes())
        digest = checksum(archive_temp)
        os.replace(public, destination)
        os.replace(archive_temp, archive)
        sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        return {**counts, "directory": str(destination), "archive": str(archive), "sha256": digest}
    finally:
        shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/datasets"))
    parser.add_argument("--output", type=Path, default=Path("data/releases/monthly_demand_anonymized_v1"))
    parser.add_argument("--private-mapping", type=Path, default=Path("data/private/sku_release_mapping.json"))
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.source, args.output, args.private_mapping), indent=2))


if __name__ == "__main__":
    main()
