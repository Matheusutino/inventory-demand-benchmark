import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


def future_month_starts(last_timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(last_timestamp) + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )


def normalize(values: np.ndarray):
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        std = 1.0
    return (values - mean) / std, mean, std


def forecast_timer_channel_independent(
    model,
    df_train_long: pd.DataFrame,
    horizon: int,
    device: str,
    normalize_input: bool,
    min_input_length: int,
) -> pd.DataFrame:
    """Checkpoint público do Timer usa entrada 2D; cada variável vira uma amostra do batch."""
    series_ids = []
    arrays = []
    stats = []
    constant_forecasts = {}

    for serie_id, serie_data in df_train_long.groupby("id", sort=False):
        values = serie_data.sort_values("timestamp")["target"].astype("float32").to_numpy()
        series_id = str(serie_id)
        if np.all(values == 0.0):
            constant_forecasts[series_id] = np.zeros(horizon, dtype=np.float32)
            continue

        if normalize_input:
            values, mean, std = normalize(values)
        else:
            mean, std = 0.0, 1.0
        if len(values) < min_input_length:
            values = np.pad(values, (min_input_length - len(values), 0), mode="constant", constant_values=0.0)
        series_ids.append(series_id)
        arrays.append(values)
        stats.append((mean, std))

    generated_forecasts = {}
    if arrays:
        context = torch.tensor(np.stack(arrays), dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model.generate(context, max_new_tokens=horizon)

        output = output.detach().cpu().numpy()
        forecasts = output[:, -horizon:]

        for idx, series_id in enumerate(series_ids):
            mean, std = stats[idx]
            generated_forecasts[series_id] = forecasts[idx] * std + mean

    pred_frames = []
    all_forecasts = {**generated_forecasts, **constant_forecasts}
    for series_id, values in all_forecasts.items():
        serie_last_timestamp = df_train_long[
            df_train_long["id"].astype(str) == series_id
        ]["timestamp"].max()
        pred_frames.append(
            pd.DataFrame(
                {
                    "id": series_id,
                    "timestamp": future_month_starts(serie_last_timestamp, horizon),
                    "q50": values,
                }
            )
        )

    return pd.concat(pred_frames, ignore_index=True)


def main(model_path, data_dir, output_dir, device, normalize_input, min_input_length, datasets_filter):
    model_name = model_path.split("/")[-1]

    datasets = discover_datasets(data_dir)
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets")
    print(f"Modelo: {model_path}")
    print("Modo: multivariado por saída, previsão canal-a-canal (checkpoint HF público usa entrada 2D)\n")

    print("Carregando Timer/Timer-XL...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()
    print("✓ Modelo carregado\n")

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
            pred_df = forecast_timer_channel_independent(
                model,
                df_train_long,
                horizon,
                device,
                normalize_input=normalize_input,
                min_input_length=min_input_length,
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

        final_predictions = final_predictions[["dataset", "id", "timestamp", "q50"]]

        pred_path = save_predictions(final_predictions, "all_datasets", model_name, output_dir)

        print(f"\n{'=' * 70}")
        print("SALVAMENTO FINAL")
        print(f"{'=' * 70}")
        print(f"✓ Todas as previsões salvas: {pred_path}")
        print(f"\nTotal de datasets processados: {len(all_predictions)}")
    else:
        print("\n✗ Nenhum dataset foi processado com sucesso")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timer/Timer-XL Forecasting")
    parser.add_argument("--model-path", type=str, default="thuml/timer-base-84m")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--output-dir", type=str, default="data/predictions")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-input-length", type=int, default=96)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        normalize_input=args.normalize_input,
        min_input_length=args.min_input_length,
        datasets_filter=args.datasets,
    )
