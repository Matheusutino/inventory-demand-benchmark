import argparse

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


def quantile_col_name(quantile: float) -> str:
    return f"q{int(round(quantile * 100))}"


def resolve_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA não está disponível. Use --device cpu.")
        return "cuda:0"
    return "cpu"


def quantile_indices(available_quantiles: list[float], requested_quantiles: list[float]) -> dict[float, int]:
    indices = {}
    for requested in requested_quantiles:
        matches = [
            idx
            for idx, available in enumerate(available_quantiles)
            if np.isclose(float(available), float(requested), atol=1e-6)
        ]
        if not matches:
            raise ValueError(
                f"Quantil {requested} não está disponível no TiRex. "
                f"Disponíveis: {available_quantiles}"
            )
        indices[requested] = matches[0]
    return indices


def forecast_tirex_channel_independent(
    model,
    df_train_long: pd.DataFrame,
    horizon: int,
    quantiles: list[float],
    batch_size: int,
    resample_strategy: str | None,
    max_accelerated_rollout_steps: int,
) -> pd.DataFrame:
    series_ids = []
    series_values = []
    last_timestamps = {}

    for serie_id, serie_data in df_train_long.groupby("id", sort=False):
        serie_data = serie_data.sort_values("timestamp")
        series_ids.append(str(serie_id))
        series_values.append(serie_data["target"].astype("float32").to_numpy(copy=True))
        last_timestamps[str(serie_id)] = serie_data["timestamp"].max()

    output_quantiles, output_mean = model.forecast(
        context=series_values,
        prediction_length=horizon,
        output_type="numpy",
        batch_size=batch_size,
        resample_strategy=resample_strategy,
        max_accelerated_rollout_steps=max_accelerated_rollout_steps,
    )

    output_quantiles = np.asarray(output_quantiles)
    output_mean = np.asarray(output_mean)
    available_quantiles = [float(q) for q in model.config.quantiles]
    q_indices = quantile_indices(available_quantiles, quantiles)

    pred_frames = []
    for row_idx, serie_id in enumerate(series_ids):
        forecast_dict = {
            "id": serie_id,
            "timestamp": future_month_starts(last_timestamps[serie_id], horizon),
        }

        for q in quantiles:
            forecast_dict[quantile_col_name(q)] = output_quantiles[row_idx, :horizon, q_indices[q]]

        if "q50" not in forecast_dict:
            forecast_dict["q50"] = output_mean[row_idx, :horizon]

        pred_frames.append(pd.DataFrame(forecast_dict))

    return pd.concat(pred_frames, ignore_index=True)


def main(
    model_path,
    data_dir,
    output_dir,
    device,
    backend,
    compile_model,
    batch_size,
    quantiles,
    datasets_filter,
    resample_strategy,
    max_accelerated_rollout_steps,
):
    from tirex import load_model

    resolved_device = resolve_device(device)
    model_name = model_path.split("/")[-1]

    datasets = discover_datasets(data_dir)
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets")
    print(f"Modelo: {model_path}")
    print("Modo: séries tratadas independentemente pelo TiRex\n")

    print("Carregando TiRex...")
    model = load_model(
        model_path,
        device=resolved_device,
        backend=backend,
        compile=compile_model,
    )
    model.eval()
    print(f"✓ Modelo carregado | device={resolved_device} | backend={backend}")
    print(f"Quantis disponíveis: {model.config.quantiles}\n")

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
            pred_df = forecast_tirex_channel_independent(
                model,
                df_train_long,
                horizon,
                quantiles=quantiles,
                batch_size=batch_size,
                resample_strategy=resample_strategy,
                max_accelerated_rollout_steps=max_accelerated_rollout_steps,
            )
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
        raise SystemExit(1)

    del model
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TiRex Forecasting")
    parser.add_argument("--model-path", type=str, default="NX-AI/TiRex")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--output-dir", type=str, default="data/predictions")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "cuda"])
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--resample-strategy", type=str, default=None, choices=[None, "frequency"])
    parser.add_argument("--max-accelerated-rollout-steps", type=int, default=1)

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        backend=args.backend,
        compile_model=args.compile_model,
        batch_size=args.batch_size,
        quantiles=args.quantiles,
        datasets_filter=args.datasets,
        resample_strategy=args.resample_strategy,
        max_accelerated_rollout_steps=args.max_accelerated_rollout_steps,
    )
