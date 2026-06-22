"""
Baselines univariados locais usando Darts com decomposição top-down.

Modelos suportados:
- NaiveMovingAverage
- ARIMA
- Prophet

Usage:
    python -m src.models.darts_univariate_baselines --model naive_moving_average
    python -m src.models.darts_univariate_baselines --model arima --datasets Mecanismos_h6
    python -m src.models.darts_univariate_baselines --model prophet
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA, NaiveMovingAverage, Prophet

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


def prophet_available() -> bool:
    """Checa se o wrapper Prophet do Darts está realmente disponível."""
    return type(Prophet).__name__ != "NotImportedModule"


def model_available(model_name: str) -> bool:
    """Retorna se o modelo está utilizável no ambiente atual."""
    if model_name == "prophet":
        return prophet_available()
    return True


def create_model(model_name: str, params: dict):
    """Cria um modelo univariado a partir do nome e dos parâmetros."""
    if model_name == "naive_moving_average":
        return NaiveMovingAverage(input_chunk_length=params["input_chunk_length"])

    if model_name == "arima":
        return ARIMA(p=params["p"], d=params["d"], q=params["q"])

    if model_name == "prophet":
        if not prophet_available():
            raise ImportError(
                "Prophet não está disponível no ambiente atual. "
                "Instale a dependência do Prophet no darts-env antes de usar este modelo."
            )
        return Prophet()

    raise ValueError(f"Modelo '{model_name}' não suportado.")


def get_hyperparameter_grid(model_name: str):
    """Grid simples por dataset para modelos univariados."""
    grids = {
        "naive_moving_average": {
            "input_chunk_length": [3, 6, 12],
        },
        "arima": {
            "p": [0, 1, 2],
            "d": [0, 1],
            "q": [0, 1, 2],
        },
        "prophet": {
            "dummy": [0],
        },
    }
    return grids[model_name]


def normalize_params(model_name: str, params: dict) -> dict:
    """Remove placeholders usados apenas para simplificar o grid."""
    if model_name == "prophet":
        return {}
    return params


def df_to_aggregate_timeseries(df: pd.DataFrame) -> TimeSeries:
    """Converte DataFrame long em uma série agregada total."""
    df_total = (
        df.groupby("timestamp", as_index=False)["target"]
        .sum()
        .sort_values("timestamp")
    )
    return TimeSeries.from_dataframe(
        df_total,
        time_col="timestamp",
        value_cols="target",
        freq="MS",
    )


def split_train_val(train_series: TimeSeries, horizon: int):
    """Divide a série agregada em treino e validação."""
    return train_series[:-horizon], train_series[-horizon:]


def compute_recent_shares(df_train_long: pd.DataFrame, window: int = 12) -> pd.Series:
    """
    Calcula pesos de decomposição top-down a partir da participação recente.
    """
    df_sorted = df_train_long.sort_values(["timestamp", "id"]).copy()
    unique_timestamps = sorted(df_sorted["timestamp"].unique())
    effective_window = min(window, len(unique_timestamps))
    recent_timestamps = unique_timestamps[-effective_window:]

    df_recent = df_sorted[df_sorted["timestamp"].isin(recent_timestamps)].copy()
    totals = df_recent.groupby("timestamp")["target"].sum().rename("total")
    df_recent = df_recent.merge(totals, on="timestamp", how="left")
    df_recent["share"] = np.where(df_recent["total"] != 0, df_recent["target"] / df_recent["total"], 0.0)

    shares = df_recent.groupby("id")["share"].mean()
    shares = shares.fillna(0.0)

    if shares.sum() <= 0:
        counts = df_recent["id"].value_counts().sort_index()
        shares = pd.Series(1.0 / len(counts), index=counts.index)
    else:
        shares = shares / shares.sum()

    shares.index = shares.index.astype(str)
    return shares


def decompose_forecast(aggregate_forecast: TimeSeries, shares: pd.Series) -> pd.DataFrame:
    """
    Decompõe a previsão agregada em previsões por série individual.
    """
    forecast_values = aggregate_forecast.values().flatten()
    timestamps = pd.to_datetime(aggregate_forecast.time_index)
    pred_frames = []

    for series_id, share in shares.items():
        pred_frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "q50": forecast_values * float(share),
                    "id": str(series_id),
                }
            )
        )

    return pd.concat(pred_frames, ignore_index=True)


def validation_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric == "mape":
        denom = np.where(y_true == 0, np.nan, y_true)
        return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    raise ValueError(f"Métrica de validação não suportada: {metric}")


def train_with_gridsearch(
    model_name: str,
    train_series: TimeSeries,
    val_series: TimeSeries,
    param_grid: dict,
    horizon: int,
    validation_metric: str,
):
    """Seleciona hiperparâmetros por dataset usando métrica na série agregada de validação."""
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_score = float("inf")
    best_params = None

    total_combinations = int(np.prod([len(v) for v in param_values]))
    print(f"  Grid: {param_grid}")
    print(f"  Testando {total_combinations} combinações...")

    for i, values in enumerate(product(*param_values), start=1):
        raw_params = dict(zip(param_names, values))
        params = normalize_params(model_name, raw_params)

        try:
            model = create_model(model_name, params)
            model.fit(train_series)
            pred = model.predict(n=horizon)
            score = validation_score(
                val_series.values().flatten(),
                pred.values().flatten(),
                validation_metric,
            )
            status = f"{params} -> {validation_metric.upper()}={score:.4f}"
        except Exception:
            score = float("inf")
            status = f"{params} -> falhou"

        if score < best_score:
            best_score = score
            best_params = params
            print(f"    [{i}/{total_combinations}] Nova melhor: {status}")
        else:
            print(f"    [{i}/{total_combinations}] {status}")

    if best_params is None:
        raise RuntimeError(f"Nenhuma combinação válida encontrada para {model_name}.")

    print(f"  ✓ Melhores parâmetros: {best_params}")
    print(f"  ✓ Melhor {validation_metric.upper()} (validação): {best_score:.4f}")
    return best_params


def predict_top_down(model_name: str, params: dict, train_series: TimeSeries, shares: pd.Series, horizon: int) -> pd.DataFrame:
    """Treina no agregado e decompõe a previsão para as séries individuais."""
    model = create_model(model_name, params)
    model.fit(train_series)
    aggregate_forecast = model.predict(n=horizon)
    return decompose_forecast(aggregate_forecast, shares)


def main(model_name, data_dir, output_dir, datasets_filter, validation_metric):
    if not model_available(model_name):
        print(f"✗ Modelo indisponível no ambiente atual: {model_name}")
        if model_name == "prophet":
            print("  Instale a dependência do Prophet no darts-env para habilitar este baseline.\n")
        return

    datasets = discover_datasets(data_dir)

    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets")
    print(f"Modelo: {model_name}\n")

    all_predictions = []
    all_best_params = []

    for dataset_name, paths in datasets.items():
        print(f"{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        try:
            df_train_long, df_test_long, horizon = DataLoader.load_dataset(paths["train"], paths["test"])
            info = DataLoader.get_dataset_info(df_train_long, df_test_long, horizon)
            print(f"Séries: {info['n_series']} | Train: {info['train_length']} | Horizon: {horizon}")

            min_required = 12 + horizon + 1
            if info["train_length"] < min_required:
                print(f"⚠️  PULADO: Train muito curto ({info['train_length']} < {min_required})")
                continue

            train_series_full = df_to_aggregate_timeseries(df_train_long)
            shares = compute_recent_shares(df_train_long)

            print("Dividindo train/val...")
            train_series, val_series = split_train_val(train_series_full, horizon)

            print("Grid Search com validação...")
            param_grid = get_hyperparameter_grid(model_name)
            best_params = train_with_gridsearch(
                model_name,
                train_series,
                val_series,
                param_grid,
                horizon,
                validation_metric,
            )

            dataset_params = {"dataset": dataset_name, **best_params}
            all_best_params.append(dataset_params)

            print("Retreinando no agregado e decompondo para predição no teste...")
            pred_test_df = predict_top_down(model_name, best_params, train_series_full, shares, horizon)
            pred_test_df["dataset"] = dataset_name

            all_predictions.append(pred_test_df)
            print("✓ Dataset processado\n")

        except Exception as exc:
            print(f"✗ Erro: {exc}\n")
            import traceback
            traceback.print_exc()
            continue

    if not all_predictions:
        print("\n✗ Nenhum dataset processado")
        return

    model_label = f"darts_{model_name}"
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_best_params = pd.DataFrame(all_best_params)

    pred_cols = ["dataset", "id", "timestamp", "q50"]
    final_predictions = final_predictions[pred_cols]

    pred_path = save_predictions(final_predictions, "all_datasets", model_label, output_dir)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    params_path = output_dir_path / f"{model_label}_best_hyperparameters.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(final_best_params.to_dict("records"), f, indent=2)

    print(f"\n{'=' * 70}")
    print("SALVAMENTO FINAL")
    print(f"{'=' * 70}")
    print(f"✓ Previsões (teste): {pred_path}")
    print(f"✓ Melhores hiperparâmetros: {params_path}")
    print(f"\nTotal de datasets: {len(all_predictions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Darts Univariate Baselines Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["naive_moving_average", "arima", "prophet", "all"],
        help="Modelo a usar (all = todos os modelos univariados).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets",
        help="Diretório dos datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/predictions",
        help="Diretório de saída.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets específicos (vazio = todos).",
    )
    parser.add_argument(
        "--validation-metric",
        type=str,
        default="mae",
        choices=["mae", "mse", "rmse", "mape"],
        help="Métrica usada apenas no grid search de validação.",
    )
    args = parser.parse_args()

    if args.model == "all":
        for current_model in ["naive_moving_average", "arima", "prophet"]:
            print(f"\n{'#' * 80}")
            print(f"# RODANDO MODELO: {current_model.upper()}")
            print(f"{'#' * 80}\n")
            main(
                model_name=current_model,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                datasets_filter=args.datasets,
                validation_metric=args.validation_metric,
            )
    else:
        main(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            datasets_filter=args.datasets,
            validation_metric=args.validation_metric,
        )
