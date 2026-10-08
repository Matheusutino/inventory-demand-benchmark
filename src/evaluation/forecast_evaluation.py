"""Align saved forecasts and evaluate independent inventory-aware components."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.inventory import InventoryAwareEvaluator
from src.utils.data_loader import DataLoader, discover_datasets
from src.analysis.inventory_framework import COMPONENTS, EXCLUDED_COMPONENTS


DEFAULT_EVALUATION_CONFIG = {
    "base_kind": "mae",
    "huber_delta": 1.0,
    "rel_eps": 1e-06,
    "sum_relative": True,
    "share_divergence": "js",
    "share_mode": "relu",
    "share_eps": 1e-08,
    "alloc_relative": False,
    "quantile_tau": 0.7,
    "delta_kind": "mse",
    "cap_mode": "factor_history_max",
    "cap_value": None,
    "cap_factor": 2.0,
    "components_axis": -1,
}

IGNORED_COMPONENTS = EXCLUDED_COMPONENTS


def load_truth_by_dataset(data_dir: str) -> dict[str, pd.DataFrame]:
    datasets = discover_datasets(data_dir)
    truth = {}
    for dataset_name, paths in datasets.items():
        _, df_test_long, _ = DataLoader.load_dataset(paths["train"], paths["test"])
        df_test_long = df_test_long.copy()
        df_test_long["id"] = df_test_long["id"].astype(str)
        df_test_long["timestamp"] = pd.to_datetime(df_test_long["timestamp"])
        truth[dataset_name] = df_test_long
    return truth


def load_history_by_dataset(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load only the training split used at each task's forecast origin."""
    histories = {}
    for dataset, paths in discover_datasets(data_dir).items():
        history = DataLoader._to_long_format(pd.read_parquet(paths["train"]))
        history["id"] = history["id"].astype(str)
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        histories[dataset] = history
    return histories


def history_statistics(history: pd.DataFrame, truth: pd.DataFrame) -> tuple[torch.Tensor, pd.DataFrame, dict]:
    """Align training scales and report eligibility independently of model predictions."""
    if history.duplicated(["id", "timestamp"]).any():
        raise ValueError("Histórico com pares SKU/timestamp duplicados.")
    ids = sorted(truth["id"].unique())
    if set(history["id"]) != set(ids):
        raise ValueError("SKUs do histórico e do teste devem coincidir.")
    wide = pivot_matrix(history, "target", ids)
    values = wide.to_numpy(dtype=float)
    if len(wide) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Histórico incompleto, negativo ou não finito; são necessários ao menos dois meses.")
    expected_dates = pd.date_range(wide.index[0], periods=len(wide), freq="MS")
    if not np.array_equal(wide.index.to_numpy(dtype="datetime64[ns]"), expected_dates.to_numpy()):
        raise ValueError("Histórico deve ter meses consecutivos para a escala MASE de lag 1.")
    if wide.index[-1] >= truth.timestamp.min():
        raise ValueError("Histórico invade o período de teste.")
    scale = np.abs(np.diff(values, axis=0)).mean(axis=0)
    mean = values.mean(axis=0)
    maximum = values.max(axis=0)
    test_zeros = truth.assign(is_zero=truth.target == 0).groupby("id").is_zero.sum().reindex(ids)
    stats = pd.DataFrame({
        "id": ids, "n_train": len(wide), "train_start": wide.index[0], "train_end": wide.index[-1],
        "training_mean": mean, "training_max": maximum, "mase_scale": scale,
        "cap_limit": 2 * maximum, "mase_eligible": scale > 0, "zero_eligible": mean > 0,
        "test_zero_observations": test_zeros.to_numpy(),
    })
    horizon = truth.timestamp.nunique()
    summary = {
        "n_train": len(wide), "train_start": wide.index[0], "train_end": wide.index[-1],
        "mase_eligible_skus": int((scale > 0).sum()),
        "mase_excluded_constant_skus": int((scale == 0).sum()),
        "mase_observations": int((scale > 0).sum() * horizon),
        "zero_eligible_skus": int((mean > 0).sum()),
        "zero_excluded_no_demand_skus": int((mean == 0).sum()),
        "zero_scored_skus": int(((mean > 0) & (test_zeros.to_numpy() > 0)).sum()),
        "zero_observations": int(test_zeros.to_numpy()[mean > 0].sum()),
        "zero_excluded_observations": int(test_zeros.to_numpy()[mean == 0].sum()),
    }
    return torch.tensor(values, dtype=torch.float64).unsqueeze(0), stats, summary


def list_prediction_files(predictions_dir: str, models: list[str] | None) -> list[Path]:
    pred_dir = Path(predictions_dir)
    if models:
        paths = []
        for model_name in models:
            path = pred_dir / f"{model_name}_all_datasets_predictions.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Arquivo de predição não encontrado: {path}")
            paths.append(path)
        return paths

    files = sorted(pred_dir.glob("*_all_datasets_predictions.parquet"))
    if not files:
        raise ValueError(f"Nenhum arquivo de previsão encontrado em '{predictions_dir}'.")
    return files


def model_name_from_path(path: Path) -> str:
    return path.stem.replace("_all_datasets_predictions", "")


def pivot_matrix(df: pd.DataFrame, value_col: str, ids_order: list[str] | None = None) -> pd.DataFrame:
    wide = df.pivot(index="timestamp", columns="id", values=value_col).sort_index()
    if ids_order is not None:
        wide = wide.reindex(columns=ids_order)
    return wide


def aligned_tensors(df_true: pd.DataFrame, df_pred: pd.DataFrame):
    for label, frame in (("truth", df_true), ("prediction", df_pred)):
        if frame.duplicated(["id", "timestamp"]).any():
            raise ValueError(f"{label}: pares SKU/timestamp duplicados.")
    keys = ["id", "timestamp"]
    expected = pd.MultiIndex.from_frame(df_true[keys])
    observed = pd.MultiIndex.from_frame(df_pred[keys])
    if len(expected.difference(observed)) or len(observed.difference(expected)):
        raise ValueError("Cobertura SKU/timestamp incompleta ou diferente entre truth e prediction.")
    merged = df_true.merge(
        df_pred[["id", "timestamp", "q50"]],
        on=["id", "timestamp"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Nenhuma linha casou entre truth e prediction.")

    ids_order = sorted(merged["id"].unique())
    true_wide = pivot_matrix(merged, "target", ids_order=ids_order)
    pred_wide = pivot_matrix(merged, "q50", ids_order=ids_order)

    if true_wide.shape != pred_wide.shape:
        raise ValueError(f"Shapes desalinhados: {true_wide.shape} vs {pred_wide.shape}")

    if not np.isfinite(true_wide.to_numpy()).all() or not np.isfinite(pred_wide.to_numpy()).all():
        raise ValueError("Demanda ou previsões não finitas/incompletas.")
    y = torch.tensor(true_wide.to_numpy(), dtype=torch.float64).unsqueeze(0)
    y_hat = torch.tensor(pred_wide.to_numpy(), dtype=torch.float64).unsqueeze(0)
    return y_hat, y, len(merged), len(ids_order), true_wide.shape[0]


def compute_components(
    y_hat: torch.Tensor, y: torch.Tensor, evaluation_config: dict, history: torch.Tensor,
) -> dict[str, float]:
    """Evaluate the nine components using aligned training history for MASE, Cap and Zero."""
    evaluator = InventoryAwareEvaluator(**evaluation_config)
    values = evaluator.evaluate(y_hat, y, history)
    return {name: float(value.cpu().item()) for name, value in values.items()}


def evaluate_models(
    truth_by_dataset: dict[str, pd.DataFrame], prediction_files: list[Path], evaluation_config: dict,
    history_by_dataset: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Recalculate saved forecasts with complete test coverage and training-only scales."""
    rows = []
    histories = {
        dataset: history_statistics(history_by_dataset[dataset], truth)
        for dataset, truth in truth_by_dataset.items()
    }

    for pred_path in prediction_files:
        model_name = model_name_from_path(pred_path)
        pred_df = pd.read_parquet(pred_path)
        pred_df = pred_df.copy()
        pred_df["id"] = pred_df["id"].astype(str)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        for dataset_name, df_true in truth_by_dataset.items():
            df_pred = pred_df[pred_df["dataset"] == dataset_name].copy()
            if df_pred.empty:
                continue

            y_hat, y, matched_rows, n_series, horizon = aligned_tensors(df_true, df_pred)
            history, _, metadata = histories[dataset_name]
            terms = compute_components(y_hat, y, evaluation_config, history)

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "matched_rows": matched_rows,
                "n_series": n_series,
                "horizon": horizon,
                **metadata,
            }
            row.update(terms)
            rows.append(row)

    if not rows:
        raise ValueError("Nenhuma combinação válida de dataset e modelo foi avaliada.")

    return pd.DataFrame(rows)


def active_components(component_values: pd.DataFrame) -> list[str]:
    return [name for name in COMPONENTS if name in component_values]


def compute_winners(component_values: pd.DataFrame, tol: float = 1e-8):
    components = active_components(component_values)
    winner_rows = []
    count_rows = []

    for component in components:
        counts = {}
        for dataset_name, group in component_values.groupby("dataset", sort=True):
            min_value = group[component].min()
            winners = group[np.isclose(group[component], min_value, atol=tol, rtol=tol)]

            for _, row in winners.iterrows():
                winner_rows.append(
                    {
                        "dataset": dataset_name,
                        "component": component,
                        "model": row["model"],
                        "value": row[component],
                    }
                )
                counts[row["model"]] = counts.get(row["model"], 0) + 1

        for model, wins in sorted(counts.items()):
            count_rows.append({"component": component, "model": model, "wins": wins})

    winners_df = pd.DataFrame(winner_rows)
    counts_df = pd.DataFrame(count_rows)
    models = sorted(component_values["model"].unique())
    if counts_df.empty:
        summary_df = pd.DataFrame({"model": models})
        for component in components:
            summary_df[component] = 0
        return winners_df, counts_df, summary_df

    summary_df = (
        counts_df.pivot(index="model", columns="component", values="wins")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    summary_df = (
        pd.DataFrame({"model": models})
        .merge(summary_df, on="model", how="left")
        .fillna(0)
    )
    for component in components:
        if component not in summary_df.columns:
            summary_df[component] = 0
        summary_df[component] = summary_df[component].astype(int)
    return winners_df, counts_df, summary_df[["model", *components]].sort_values("model")
