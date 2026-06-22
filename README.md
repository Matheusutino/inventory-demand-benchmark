# Beyond Point-Wise Accuracy: An Inventory-Aware Benchmark for Multi-SKU Enterprise Demand Forecasting

A benchmark of monthly enterprise demand forecasting models across multiple
SKUs. The project compares statistical methods, classical machine learning
models, and time-series foundation models over H3 and H6 forecasting horizons.

In addition to traditional forecasting metrics, the benchmark uses the
`InventoryDemandLoss`, a composite evaluation designed to capture operational
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
    ├── losses/         # InventoryDemandLoss
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
- Global Naive Aggregate;
- Global Naive Drift;
- Global Naive Seasonal.

Darts models use an internal validation split for each dataset. MAE is used as
the default validation metric.

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

## Inventory Demand Loss

The implementation is available at:

```text
src/losses/loss.py
```

The current analysis uses the following components:

- `item`: mean error across individual SKUs;
- `sum`: aggregate-volume error;
- `share`: difference between actual and predicted SKU shares;
- `alloc`: allocation error across items;
- `weighted_item`: scale-weighted item error;
- `rel`: relative error;
- `cap`: excessive-forecast penalty;
- `asym`: asymmetric error penalty;
- `sparse`: behavior under sparse demand;
- `delta`: temporal-change error;
- `robust`: error with limited influence from extreme residuals.

The `neg`, `tv`, and `int` components are not included in the primary
comparison. For the critical-difference diagrams and heatmaps, every component
is normalized between 0 and 1 within each dataset before aggregation.

## Main Outputs

Traditional forecasting metrics:

```text
data/analysis/metrics/
```

Dataset characterization:

```text
data/analysis/dataset/dataset_summary.csv
data/analysis/dataset/dataset_sku_summary.csv
data/analysis/dataset/dataset_panel_summary.csv
```

Wins by Inventory Demand Loss component:

```text
data/analysis/loss_wins/loss_component_values.csv
data/analysis/loss_wins/loss_component_winners.csv
data/analysis/loss_wins/loss_component_win_counts_summary.csv
```

Critical-difference diagrams:

```text
data/analysis/cd_diagram/inventory_loss_cd.png
data/analysis/cd_diagram/mae_cd.png
```

Main figures:

```text
data/analysis/plots/item_sum_mae_overall.pdf
data/analysis/plots/item_mae_vs_horizon_degradation.pdf
data/analysis/plots/inventory_loss_panel_heatmap.pdf
data/analysis/plots/dataset_panel_inventory_rank_heatmap.pdf
```

Corresponding tables are saved as CSV files in the same output directories.

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
