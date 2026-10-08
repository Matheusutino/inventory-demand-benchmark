# Inventory-Aware Evaluation of Forecasting Models for Sparse Multi-SKU Demand

An evaluation of forecasting models for sparse monthly demand across multiple
SKUs. The project compares statistical methods, classical machine learning
models, and time-series foundation models over H3 and H6 forecasting horizons.

In addition to traditional forecasting metrics, the benchmark uses
the individual components of the Inventory-Aware Evaluation Framework to capture operational
aspects of demand forecasting, including item-level error, aggregate volume,
SKU shares, allocation, sparsity, and temporal variation.

## Project Structure

```text
.
├── data/
│   ├── datasets/       # Train/test datasets
│   ├── predictions/    # Model predictions
│   └── analysis/       # Metrics, tables, and figures
├── requirements/       # Dependencies separated by model family
├── scripts/
│   ├── setup_venvs.sh  # Virtual environment setup
│   └── run_models.py   # Model runner
└── src/
    ├── analysis/       # Metrics and visualizations
    ├── evaluation/     # Independent inventory-aware evaluation components
    ├── models/         # Model wrappers
    └── utils/          # Data loading and output helpers
```

The `data/` directory is ignored by Git. Datasets and generated results must be
made available locally.

## Dataset Format

Each dataset consists of two files:

```text
data/datasets/<dataset>_train.parquet
data/datasets/<dataset>_test.parquet
```

Files must use a wide format:

```text
date | SKU_1 | SKU_2 | ... | SKU_N
```

- `date`: monthly timestamp;
- remaining columns: demand for each SKU;
- number of test rows: forecasting horizon.

Example:

```text
data/datasets/Filtros_h3_train.parquet
data/datasets/Filtros_h3_test.parquet
data/datasets/Filtros_h6_train.parquet
data/datasets/Filtros_h6_test.parquet
```

### Anonymized dataset release

Prepare a shareable dataset package without changing the private benchmark inputs:

```bash
.venvs/analysis/bin/python -m src.utils.export_anonymized_dataset
```

The exporter validates H3/H6 consistency and writes
`data/releases/monthly_demand_anonymized_v1.zip`, its SHA-256 checksum, and an
extracted release directory. The package contains 20 paired Parquet files, a
deduplicated long-format CSV, task/panel metadata and a dataset card. Dates,
demand values, zero-demand SKUs and train/test boundaries are preserved.

SKU identifiers are replaced by persistent random codes across all splits and
horizons. The correspondence file at `data/private/sku_release_mapping.json`
stays outside the release and must remain private. Do not publish the source
datasets, spreadsheets or this mapping. Product-family labels and demand
patterns remain in the public data; this is identifier pseudonymization.

The generated package is a local publication draft. Hosting, licensing and
citation metadata must be finalized before public
distribution. See [the release guide](docs/dataset_release.md) for verification
and reproducible exports. Datasets and release artifacts remain Git-ignored.

## Installation

The project uses one virtual environment per model family to avoid dependency
conflicts.

Create all environments:

```bash
scripts/setup_venvs.sh
```

Create selected environments:

```bash
scripts/setup_venvs.sh chronos darts analysis
```

Synchronize dependencies in existing environments:

```bash
scripts/setup_venvs.sh --sync-existing chronos darts analysis
```

Environments are created under:

```text
.venvs/analysis
.venvs/chronos
.venvs/darts
.venvs/granite_ttm
.venvs/morais
.venvs/tabpfn_time_series
.venvs/timer_sundial
.venvs/timesfm
.venvs/tirex
```

The model runner does not activate environments. It directly invokes the
appropriate interpreter from `.venvs/<environment>/bin/python`.

## Models

Time-series foundation models:

- Amazon Chronos-2;
- Salesforce Moirai 2.0 Small;
- Google TimesFM 2.5 200M;
- Prior Labs TabPFN-TS v3;
- NX-AI TiRex;
- IBM Granite TinyTimeMixer R2;
- THUML Sundial Base 128M;
- THUML Timer Base 84M.

Statistical and classical machine learning baselines implemented with Darts:

- ARIMA;
- Prophet;
- Moving Average;
- Linear Regression;
- Random Forest;
- XGBoost;
- Naive Aggregate;
- Naive Drift;
- Naive Seasonal.

Intermittent-demand baselines implemented with StatsForecast:

- Croston Classic;
- Syntetos–Boylan Approximation (SBA);
- Teunter–Syntetos–Babai (TSB).

Darts models use an internal validation split for each dataset. MAE is used as
the default validation metric.

### Intermittent-demand baselines

These models use the `.venvs/darts` environment. Synchronize its dependencies
and run all three with:

```bash
scripts/setup_venvs.sh --sync-existing darts
scripts/run_models.py --models croston sba tsb --datasets Filtros_h3
```

Omit `--datasets` to process all datasets. The models are also included in the
default model run and available as `statsforecast_croston`, `statsforecast_sba`,
and `statsforecast_tsb`, or together as `intermittent`.

Each SKU is fitted independently using its full available history. One parameter
configuration is selected per dataset, shared across SKUs. The last H training
months form the validation set; the selected model is then refitted on the full
training data to forecast H test months. By default, selection uses the mean
absolute error across all SKU/month validation cells. The minimum training
length is `12 + H + 1` months, matching the existing baselines.

| Model | Parameters | Candidates per dataset |
|---|---|---:|
| Croston | `alpha=0.1`, fixed by the library | 1 |
| SBA | `alpha=0.1`, fixed by the library | 1 |
| TSB | `alpha_d` and `alpha_p` each in `[0.05, 0.10, 0.20]` | 9 |

The public [StatsForecast API](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#crostonclassic)
does not expose configurable smoothing for Croston Classic or SBA. Their fixed
settings avoid duplicating identical candidates. TSB parameters are explicitly
specified; no additional automatic optimization is performed.

Inputs must have complete, consecutive month-start timestamps and finite,
nonnegative demand. Train/test SKU columns must match. Zero-demand SKUs are
retained. Prefer MAE over MAPE for intermittent demand; MAPE excludes zero
validation cells and is undefined when all validation demands are zero.

Predictions use the existing `dataset`, `id`, `timestamp`, `q50` schema. The
`q50` column stores the library's deterministic **mean** forecast for pipeline
compatibility; it is not an estimated median. Outputs are saved as
`data/predictions/statsforecast_<model>_all_datasets_predictions.parquet`, with
parameters, validation scores, candidate counts, and library version in the
corresponding `statsforecast_<model>_best_hyperparameters.json`.

Run the integration and regression checks with:

```bash
.venvs/darts/bin/python -m unittest discover -s tests -v
```

## Running Models

List available models and datasets:

```bash
scripts/run_models.py --list
```

Run all default models on all datasets:

```bash
scripts/run_models.py --device cuda
```

Models requiring training or external license acceptance are not included by
default.

Run one model:

```bash
scripts/run_models.py --models timesfm
```

Run selected models:

```bash
scripts/run_models.py \
  --models chronos2 moirai2_small tirex \
  --device cuda
```

Run a model on one dataset:

```bash
scripts/run_models.py \
  --models granite_ttm \
  --datasets Mecanismos_h3
```

Run selected model and dataset combinations:

```bash
scripts/run_models.py \
  --models timesfm tabpfn_ts \
  --datasets Filtros_h3 "Placas de Controle_h6" \
  --include-gated
```

Predictions are saved to:

```text
data/predictions/<model>_all_datasets_predictions.parquet
```

### TabPFN-TS

Local TabPFN-TS inference requires license acceptance and a Prior Labs API key:

1. open `https://ux.priorlabs.ai`;
2. accept the TabPFN license;
3. copy the API key;
4. export the token:

```bash
export TABPFN_TOKEN="<your-api-key>"
```

Then run:

```bash
scripts/run_models.py \
  --models tabpfn_ts \
  --include-gated
```

## Running the Analyses

Create the analysis environment:

```bash
scripts/setup_venvs.sh analysis
```

Run the complete analysis pipeline:

```bash
src/analysis/run_all_analysis.sh \
  --python .venvs/analysis/bin/python
```

Generate only the dataset overview (aggregate demand with H3/H6 test windows,
training-only ADI/CV², and their combined figure):

```bash
.venvs/analysis/bin/python -m src.analysis.plot_dataset_overview
```

Figures are saved as PDF in `data/analysis/dataset/overview/`, together with their CSV
tables. Aggregate demand defaults to an index with the common training mean
equal to 100; use `--scale absolute` for original units. Details are documented
in [`src/analysis/README.md`](src/analysis/README.md).

Also generate prediction plots for a selected dataset:

```bash
src/analysis/run_all_analysis.sh \
  --python .venvs/analysis/bin/python \
  --dataset Mecanismos_h3
```

Skip selected analyses:

```bash
src/analysis/run_all_analysis.sh \
  --python .venvs/analysis/bin/python \
  --exclude cd_diagram.py plot_dataset_predictions.py
```

## Inventory-Aware Evaluation Framework

The framework evaluates nine operational dimensions separately:

- `item`: mean absolute error (MAE) across individual SKUs;
- `sum`: relative squared error of aggregate demand;
- `share`: Jensen–Shannon divergence of SKU demand shares;
- `alloc`: squared allocation error after removing the monthly mean residual;
- `scaled` (MASE): absolute error divided by the SKU's mean absolute lag-one training change;
- `cap`: squared positive exceedance above twice the SKU's maximum training demand;
- `asym`: pinball score with tau = 0.7;
- `delta`: squared error of temporal changes within the forecast window, without a training anchor;
- `zero`: mean signed forecast divided by mean training demand, over zero-demand test observations.

Scaled excludes constant-history SKUs (`q_i = 0`); Zero excludes SKUs with no
training demand (mean = 0). There is no epsilon replacement for either scale.
Each task uses its own paired training split (44 months for H6, 47 for H3),
with no test demand used in scales or caps. `training_scales_by_sku.csv`,
`component_coverage.csv` and `component_coverage_report.md` document the scales
and exclusions. A metric with no eligible observations is undefined, and
cannot enter the complete ranking comparison. Saved forecasts are used
without clipping, so Zero can be negative when forecasts are negative.

Weighted, pointwise Relative, Robust and Sparse are excluded, along with
`neg`, `tv`, `int`, and `total`. Components are neither normalized nor combined
into a global score. Both Scaled and Zero favor zero forecasts; Asym penalizes
underprediction and Sum penalizes underestimating the total. Zero measures
forecasts on zero-demand observations rather than compensating for that bias.
The implementation is in `src/evaluation/inventory.py`.
`InventoryAwareEvaluator.evaluate()` returns the nine components separately,
without gradients or a combined training objective. Data alignment and saved
forecast evaluation helpers are in `src/evaluation/forecast_evaluation.py`.

Regenerate the framework analyses:

```bash
.venvs/analysis/bin/python -m src.analysis.evaluate_inventory_framework
.venvs/analysis/bin/python -m src.analysis.plot_inventory_component_rank_heatmap
.venvs/analysis/bin/python -m src.analysis.analyze_component_concordance
```

The central model × component heatmap ranks models within each task (panel ×
horizon), then averages ranks over tasks with equal weight. Lower ranks are
better and exact ties receive average ranks. Rows follow five families: naive,
statistical, intermittent-demand, machine learning, and foundation models,
with alphabetical ordering within each family. Missing or nonfinite results
and inconsistent model/task coverage are rejected.

The primary concordance figure computes Spearman across models within each
of the ten tasks, then shows the mean ± sample standard deviation (ddof=1)
of those coefficients. The dispersion describes differences between tasks,
not confidence intervals. Figures show only the lower triangle, excluding
the diagonal. Correlations of mean task ranks and Kendall tau-b are
supplementary. Constant rankings remain undefined; valid-task counts are
exported. High agreement indicates similar rankings, not statistical
independence. Model-specific rank differences use `item` as the reference.

Win counts are supplementary and remain separate for each component. Legacy
figures that combine normalized components are excluded from the standard
pipeline. The CD diagram generated by the current pipeline uses MAE only.

## Analysis Outputs

Figures are exported as **PDF only**. CSV tables and reports live with their
corresponding analysis:

```text
data/analysis/
  dataset/                       # Dataset and SKU descriptions
    overview/                    # Aggregate demand and ADI × CV²
  accuracy/
    metrics/                     # Traditional metrics by model and dataset
    aggregate/                   # Comparisons of forecasting accuracy
    item_sum_mae/                # Item-level and aggregate-sum MAE
    cd_diagram/                  # MAE critical-difference diagram
  diagnostics/
    horizon_degradation/         # H3 versus H6
    mae_vs_horizon/               # Accuracy versus horizon sensitivity
  framework/
    components/                  # component_values.csv (nine components)
    ranks/                       # Model × component mean-rank heatmap
    concordance/                 # Spearman, Kendall, dispersion and report
    wins/                        # Supplementary component win counts
  predictions/<dataset>/         # Optional forecast figures
  archive/                       # Superseded outputs from before the cleanup
```

The main framework figures are:

```text
data/analysis/framework/ranks/inventory_component_rank_heatmap.pdf
data/analysis/framework/concordance/component_concordance_spearman.pdf
```

Use `--analysis-dir` to relocate the complete output tree, or `--output-dir`
on individual analysis scripts. See [the analysis guide](src/analysis/README.md)
for commands and methodology.

## Background Execution

Run all models in the background:

```bash
nohup scripts/run_models.py --device cuda \
  > run_models.log 2>&1 &
```

Monitor progress:

```bash
tail -f run_models.log
```

Run selected models in the background:

```bash
nohup scripts/run_models.py \
  --models granite_ttm timer_base timesfm \
  --device cuda \
  > run_selected_models.log 2>&1 &
```

## Reproducibility

- Run all commands from the project root.
- Use the files under `requirements/` to reproduce the environments.
- Keep dataset names consistent between train and test files.
- Regenerate predictions before running `run_all_analysis.sh` for final
  comparisons.
- External checkpoints and APIs may require authentication or license
  acceptance.

Additional analysis documentation is available in
[`src/analysis/README.md`](src/analysis/README.md).
