import argparse
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


def future_month_starts(last_timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(last_timestamp) + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )


def parse_version(version_string: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version_string.split(".")[:3] if part.isdigit())


def load_timesfm(model_path: str, max_context: int, max_horizon: int, torch_compile: bool):
    import timesfm

    installed_version = version("timesfm")
    if parse_version(installed_version) < (2, 0, 0):
        raise ImportError(
            f"timesfm=={installed_version} instalado. "
            'Use `pip install "timesfm[torch]>=2.0.0,<3"` para a API nova.'
        )

    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise ImportError(
            "A instalação atual de timesfm não expõe TimesFM_2p5_200M_torch. "
            'Reinstale com `pip install --upgrade "timesfm[torch]>=2.0.0,<3"`.'
        )

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model_path,
        torch_compile=torch_compile,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model, installed_version


def forecast_timesfm(
    model,
    df_train_long: pd.DataFrame,
    horizon: int,
    quantiles: list[float],
) -> pd.DataFrame:
    """TimesFM público é univariado; aqui cada canal é previsto separadamente."""
    forecasts = []

    for serie_id, serie_data in df_train_long.groupby("id", sort=False):
        serie_data = serie_data.sort_values("timestamp")
        values = serie_data["target"].astype("float32").to_numpy()

        point_forecast, quantile_forecast = model.forecast(
            horizon=horizon,
            inputs=[values],
        )
        point = np.asarray(point_forecast[0])[:horizon]
        quantile_values = {"q50": point}

        if quantile_forecast is not None:
            q_arr = np.asarray(quantile_forecast[0])[:horizon]
            if q_arr.ndim == 2 and q_arr.shape[1] >= 10:
                for q in set(quantiles + [0.5]):
                    col_name = f"q{int(q * 100)}"
                    quantile_index = int(round(q * 10))
                    quantile_index = min(max(quantile_index, 1), 9)
                    quantile_values[col_name] = q_arr[:, quantile_index]

        timestamps = future_month_starts(serie_data["timestamp"].max(), horizon)
        forecast_dict = {
            "id": str(serie_id),
            "timestamp": timestamps,
        }
        forecast_dict.update(quantile_values)
        forecasts.append(pd.DataFrame(forecast_dict))

    return pd.concat(forecasts, ignore_index=True)


def main(model_path, data_dir, output_dir, device, max_context, max_horizon, torch_compile, quantiles, datasets_filter):
    if device == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    model_name = model_path.split("/")[-1]
    datasets = discover_datasets(data_dir)
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets")
    print(f"Modelo: {model_path}")
    print("Modo: multivariado por saída, previsão canal-a-canal (limitação do TimesFM público)\n")

    print("Carregando TimesFM...")
    model, timesfm_version = load_timesfm(
        model_path,
        max_context=max_context,
        max_horizon=max_horizon,
        torch_compile=torch_compile,
    )
    print(f"✓ Modelo carregado (timesfm={timesfm_version})\n")

    all_predictions = []

    for dataset_name, paths in datasets.items():
        print(f"{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        try:
            df_train_long, df_test_long, horizon = DataLoader.load_dataset(paths["train"], paths["test"])
            info = DataLoader.get_dataset_info(df_train_long, df_test_long, horizon)
            print(f"Séries: {info['n_series']} | Train: {info['train_length']} | Horizon: {horizon}")

            print("Gerando previsões...")
            pred_df = forecast_timesfm(model, df_train_long, horizon, quantiles)
            pred_df["dataset"] = dataset_name

            all_predictions.append(pred_df)

            print("✓ Dataset processado\n")

        except Exception as e:
            print(f"✗ Erro: {e}\n")
            import traceback

            traceback.print_exc()
            continue

    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)

        pred_cols = ["dataset", "id", "timestamp"] + [
            col for col in final_predictions.columns if col.startswith("q")
        ]
        final_predictions = final_predictions[pred_cols]

        pred_path = save_predictions(final_predictions, "all_datasets", model_name, output_dir)

        print(f"\n{'=' * 70}")
        print("SALVAMENTO FINAL")
        print(f"{'=' * 70}")
        print(f"✓ Todas as previsões salvas: {pred_path}")
        print(f"\nTotal de datasets processados: {len(all_predictions)}")
    else:
        print("\n✗ Nenhum dataset foi processado com sucesso")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimesFM Forecasting")
    parser.add_argument("--model-path", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--output-dir", type=str, default="data/predictions")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--max-horizon", type=int, default=256)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--datasets", type=str, nargs="+", default=None)

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        max_context=args.max_context,
        max_horizon=args.max_horizon,
        torch_compile=args.torch_compile,
        quantiles=args.quantiles,
        datasets_filter=args.datasets,
    )
