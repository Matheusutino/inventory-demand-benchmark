"""Croston, SBA and TSB per SKU using the public StatsForecast API."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsforecast
from statsforecast.models import CrostonClassic, CrostonSBA, TSB

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


MODEL_NAMES = ("croston", "sba", "tsb")
SMOOTHING_VALUES = (0.05, 0.10, 0.20)


def parameter_candidates(model_name: str) -> list[dict[str, float]]:
    """Use fixed library defaults for Croston/SBA and a 3x3 grid for TSB."""
    if model_name in {"croston", "sba"}:
        return [{"alpha": 0.1}]
    if model_name == "tsb":
        return [
            {"alpha_d": alpha_d, "alpha_p": alpha_p}
            for alpha_d, alpha_p in product(SMOOTHING_VALUES, repeat=2)
        ]
    raise ValueError(f"Modelo não suportado: {model_name}")


def create_model(model_name: str, params: dict[str, float]):
    """Construct a library model without hidden parameter optimization."""
    if model_name in {"croston", "sba"}:
        if params != {"alpha": 0.1}:
            raise ValueError("StatsForecast usa alpha=0.1 fixo para Croston/SBA.")
        return CrostonClassic() if model_name == "croston" else CrostonSBA()
    if model_name == "tsb":
        return TSB(alpha_d=params["alpha_d"], alpha_p=params["alpha_p"])
    raise ValueError(f"Modelo não suportado: {model_name}")


def to_monthly_panel(df_long: pd.DataFrame) -> pd.DataFrame:
    """Require a complete monthly panel; missing demand is not treated as zero."""
    panel = df_long.pivot(index="timestamp", columns="id", values="target").sort_index()
    # Parquet may store microseconds; compare timestamps at a common precision.
    panel.index = pd.DatetimeIndex(panel.index).as_unit("ns")
    panel.columns = panel.columns.astype(str)
    panel = panel.astype(float)
    if panel.empty:
        raise ValueError("Painel vazio.")
    expected_dates = pd.date_range(panel.index.min(), periods=len(panel), freq="MS")
    if not panel.index.equals(expected_dates):
        raise ValueError("Datas devem ser mensais, consecutivas e no início do mês.")
    if not np.isfinite(panel.to_numpy()).all() or (panel.to_numpy() < 0).any():
        raise ValueError("Demanda deve ser finita, não negativa e presente para cada SKU/mês.")
    return panel


def forecast_values(
    model_name: str,
    params: dict[str, float],
    history: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Fit each SKU independently with the same candidate parameters."""
    predictions = np.column_stack([
        create_model(model_name, params).forecast(
            y=history[sku].to_numpy(dtype=float), h=horizon
        )["mean"]
        for sku in history.columns
    ])
    if predictions.shape != (horizon, len(history.columns)):
        raise ValueError("Dimensões de previsão incompatíveis com horizonte/SKUs.")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("StatsForecast retornou previsões inválidas.")
    return predictions


def validation_score(truth: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    """Score aligned SKU/month cells, matching the existing ML validation."""
    residual = truth - prediction
    if metric == "mae":
        return float(np.mean(np.abs(residual)))
    if metric == "mse":
        return float(np.mean(residual ** 2))
    if metric == "rmse":
        return float(np.sqrt(np.mean(residual ** 2)))
    if metric == "mape":
        nonzero = truth != 0
        if not nonzero.any():
            raise ValueError("MAPE indefinido: validação contém apenas zeros. Use MAE.")
        return float(np.mean(np.abs(residual[nonzero] / truth[nonzero])) * 100)
    raise ValueError(f"Métrica não suportada: {metric}")


def select_parameters(
    model_name: str, panel: pd.DataFrame, horizon: int, metric: str
) -> tuple[dict[str, float], float]:
    """Hold out the last H training months and select once per dataset."""
    if horizon < 1 or len(panel) < 12 + horizon + 1:
        raise ValueError("Treino muito curto: requer 12 + horizonte + 1 meses.")
    history, validation = panel.iloc[:-horizon], panel.iloc[-horizon:]
    candidates = parameter_candidates(model_name)
    best_params, best_score = None, float("inf")
    for index, params in enumerate(candidates, start=1):
        prediction = forecast_values(model_name, params, history, horizon)
        score = validation_score(validation.to_numpy(), prediction, metric)
        print(f"  [{index}/{len(candidates)}] {params} -> {metric.upper()}={score:.4f}")
        if score < best_score:
            best_params, best_score = params, score
    if best_params is None:
        raise ValueError("Nenhuma configuração com score de validação finito.")
    return best_params, best_score


def run_model(
    model_name: str,
    data_dir: str,
    output_dir: str,
    datasets_filter: list[str] | None,
    validation_metric: str,
) -> int:
    """Select parameters, refit all training months, and save standard outputs."""
    datasets = discover_datasets(data_dir)
    if datasets_filter:
        unknown = set(datasets_filter) - set(datasets)
        if unknown:
            raise ValueError(f"Datasets não encontrados: {sorted(unknown)}")
        datasets = {name: paths for name, paths in datasets.items() if name in datasets_filter}
    if not datasets:
        raise ValueError(f"Nenhum dataset encontrado em {data_dir}.")

    predictions, selections = [], []
    failures = 0
    for dataset_name, paths in datasets.items():
        print(f"\nModelo: {model_name} | Dataset: {dataset_name}")
        try:
            train_long, test_long, horizon = DataLoader.load_dataset(paths["train"], paths["test"])
            panel = to_monthly_panel(train_long)
            test_panel = to_monthly_panel(test_long)
            dates = pd.date_range(panel.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            if not panel.columns.equals(test_panel.columns) or not dates.equals(test_panel.index):
                raise ValueError("Teste deve continuar os meses do treino com os mesmos SKUs.")

            params, score = select_parameters(model_name, panel, horizon, validation_metric)
            values = forecast_values(model_name, params, panel, horizon)
            frame = pd.DataFrame(values, index=dates, columns=panel.columns)
            frame.index.name = "timestamp"
            frame = frame.reset_index().melt(id_vars="timestamp", var_name="id", value_name="q50")
            frame["dataset"] = dataset_name
            predictions.append(frame[["dataset", "id", "timestamp", "q50"]])
            selections.append({
                "dataset": dataset_name,
                **params,
                "validation_metric": validation_metric,
                "validation_score": score,
                "num_candidates": len(parameter_candidates(model_name)),
                "statsforecast_version": statsforecast.__version__,
            })
        except Exception as exc:
            failures += 1
            print(f"✗ {dataset_name}: {exc}")

    if predictions:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        label = f"statsforecast_{model_name}"
        prediction_path = save_predictions(pd.concat(predictions, ignore_index=True), "all_datasets", label, output_dir)
        params_path = output_path / f"{label}_best_hyperparameters.json"
        params_path.write_text(json.dumps(selections, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"✓ Previsões: {prediction_path}\n✓ Parâmetros: {params_path}")
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--output-dir", default="data/predictions")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--validation-metric", choices=["mae", "mse", "rmse", "mape"], default="mae")
    args = parser.parse_args()
    failed = False
    for name in MODEL_NAMES if args.model == "all" else [args.model]:
        try:
            failed |= bool(run_model(name, args.data_dir, args.output_dir, args.datasets, args.validation_metric))
        except ValueError as exc:
            failed = True
            print(f"✗ {name}: {exc}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
