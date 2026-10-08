# Anonymized dataset release

This dataset accompanies **Inventory-Aware Evaluation of Forecasting Models for Sparse Multi-SKU Demand**.

The release exporter replaces original SKU identifiers with opaque random codes
while retaining monthly demand and the benchmark's original forecasting tasks.
It does not run or retrain forecasting models.

## Build

From the repository root, with pandas, NumPy and pyarrow installed:

```bash
.venvs/analysis/bin/python -m src.utils.export_anonymized_dataset \
  --source data/datasets \
  --output data/releases/monthly_demand_anonymized_v1 \
  --private-mapping data/private/sku_release_mapping.json
```

The analysis and Darts environments both include these dependencies. The module
can also run with another Python interpreter containing those packages.

The source must contain complete `<panel>_h3_{train,test}.parquet` and
`<panel>_h6_{train,test}.parquet` pairs. The exporter checks monthly continuity,
unique SKU/date fields, finite nonnegative demand, split continuity, horizon
length and equality of the complete histories reconstructed from H3 and H6.
Other source files, including spreadsheets, are never packaged.

The default outputs are:

- `data/releases/monthly_demand_anonymized_v1/`: extracted public files.
- `data/releases/monthly_demand_anonymized_v1.zip`: the shareable package.
- `data/releases/monthly_demand_anonymized_v1.zip.sha256`: archive checksum.
- `data/private/sku_release_mapping.json`: private correspondence, **never share**.

The release contains 1,375 distinct SKUs, five panels, 50 months (January 2022
through February 2026), 68,750 unique SKU/month observations and ten tasks.
H3 uses 47 training months and three test months; H6 uses 44 and six.
The two horizons reuse the same full histories and have overlapping test windows.
All 80 SKUs without demand in the common H6 training history are retained.

## Identifier handling

Each raw SKU receives one randomly assigned `SKU_<16 hex digits>` code. This
mapping is shared across panels, horizons and train/test files. Sorting public
columns by the anonymous codes avoids exposing source identifier order.
Panel filenames use `panel_01` through `panel_05`, with English product-family
labels in `panels.csv`. Parquet schema metadata is stripped before export.

The private mapping is written with owner-only file permissions. Keep it in a
private location and retain a private backup if future exports must use the same
IDs. It is outside the release directory and absent from the explicit ZIP file
list. Public demand values, dates and family labels are preserved: this is
pseudonymization of identifiers rather than a guarantee that external demand
patterns cannot be linked back to products.

## Inspect and verify

The generated `README.md` is the dataset card. `tasks.csv` records exact date
boundaries, historical zero-demand counts and historical constant-series counts.
`panels.csv` records panel sizes and historical coverage. `monthly_demand.csv`
contains one full series per panel/SKU without duplicate H3/H6 observations.
The 20 Parquet files retain the benchmark's wide input schema.

The exporter round-trips all Parquet files and the CSV and checks exact values
before creating the archive. `manifest.json` records SHA-256 and file sizes for
every public file except itself. The external checksum covers the complete ZIP:

```bash
cd data/releases
sha256sum -c monthly_demand_anonymized_v1.zip.sha256
unzip -t monthly_demand_anonymized_v1.zip
```

Run focused regression checks with:

```bash
.venvs/analysis/bin/python -m unittest discover -s tests -p test_anonymized_dataset.py -v
```

Checks cover exact demand/date preservation, shared SKU aliases, benchmark
DataLoader compatibility, CSV reconstruction, private metadata exclusion,
checksums, reproducible public file contents, and malformed input rejection.

Existing release outputs are never overwritten. To export again, choose a new
`--output` and reuse the same private mapping. Public file contents are
reproducible with unchanged sources, mapping and library versions; ZIP member
paths include the output directory's name. A new mapping produces new IDs.

## Publication status

This is a prepared local release, not a published dataset. The draft grants no
reuse license and deliberately leaves `license` unset in its manifest. Final
publication needs the authors' choice of host/license, provenance and citation
information. The generated dataset card
identifies these pending fields.

When finalizing those fields, update the exporter template, regenerate the
package and checksums, and publish only the release ZIP and its checksum.
Record the public DOI or permanent URL here and in the paper after publication.
Source datasets, source spreadsheets and the private mapping stay out of Git
and out of the public deposit.
