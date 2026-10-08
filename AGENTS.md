# Repository Guidelines

## Project Structure & Module Organization

The project, "Inventory-Aware Evaluation of Forecasting Models for Sparse Multi-SKU Demand", compares monthly multi-SKU demand forecasts at H3 and H6 horizons using forecasting metrics and the Inventory-Aware Evaluation Framework.

- `src/models/`: model wrappers and Darts baselines.
- `src/utils/`: dataset loading, prediction persistence, and evaluation helpers.
- `src/evaluation/`: independent inventory-aware evaluation components and forecast alignment helpers.
- `src/analysis/`: metrics, tables, plots, and analysis documentation.
- `scripts/`: environment setup and model orchestration.
- `requirements/`: dependencies separated by model family.
- `tests/`: regression checks for dataset statistics and intermittent-baseline CLI integration.
- `data/datasets/`, `data/predictions/`, `data/analysis/`: local datasets and generated artifacts; all are Git-ignored.

## Build, Test, and Development Commands

Run commands from the repository root. There is no separate build step.

- `scripts/setup_venvs.sh darts analysis`: create selected environments under `.venvs/`.
- `scripts/setup_venvs.sh --sync-existing darts analysis`: synchronize existing environments with their requirements.
- `scripts/run_models.py --list`: list registered models and available datasets.
- `scripts/run_models.py --models darts_naive_moving_average --datasets Filtros_h3`: run a focused baseline experiment.
- `src/analysis/run_all_analysis.sh --python .venvs/analysis/bin/python`: regenerate metrics, tables, and figures from predictions.

The runner invokes each model family's interpreter directly; activation is unnecessary. Analysis outputs are grouped under `data/analysis/{dataset,accuracy,diagnostics,framework,predictions}/`; export figures as PDF only. The framework evaluates nine separate components: `item` (MAE), `sum`, `share`, `alloc`, `scaled` (MASE), `cap`, `asym`, `delta`, and `zero`. MASE/Zero scales and Cap limits use training history; `weighted_item`, `rel`, `robust`, and `sparse` are excluded.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` constants. Follow nearby code's quoting and import conventions; add type hints and concise docstrings to public helpers. No formatter or linter is configured. Register new wrappers in `MODEL_SPECS` in `scripts/run_models.py` and supply matching family requirements.

## Testing Guidelines

Use standard-library `unittest`: `.venvs/darts/bin/python -m unittest discover -s tests -v`. No coverage threshold is configured. Validate changes with one model/dataset run, then the affected analysis. Check SKU/timestamp alignment, horizon length, prediction schema, and finite metrics. For evaluation component changes, compare perfect forecasts, zero demand, and over-/underprediction on small synthetic arrays. Name tests `tests/test_<module>.py`.

## Commit & Pull Request Guidelines

Existing commits use brief messages (`update`, `update README`); no stricter convention is established. Prefer descriptive imperative subjects such as `Fix forecast timestamp alignment`. PRs should explain the change, link relevant issues, identify models/datasets and validation commands, and attach figures when plots change.

## Data & Configuration

Use paired `<dataset>_train.parquet` and `<dataset>_test.parquet` files containing `date` plus SKU columns. Preserve train/test separation. Keep datasets, checkpoints, and credentials out of commits; provide `TABPFN_TOKEN` through the environment when required.
