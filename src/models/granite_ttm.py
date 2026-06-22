import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tsfm_public.toolkit.get_model import get_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Converte long format em matriz temporal multivariada."""
    return (
        df_long.pivot(index="timestamp", columns="id", values="target")
        .sort_index()
        .astype("float32")
    )


def future_month_starts(last_timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(last_timestamp) + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )


def forecast_ttm(
    model,
    df_train_long: pd.DataFrame,
    horizon: int,
    device: str,
    frequency_token: int,
) -> pd.DataFrame:
    """Gera forecast multivariado real com TTM usando todos os canais juntos."""
    train_wide = long_to_wide(df_train_long)
    series_ids = train_wide.columns.astype(str).tolist()

    past_values = torch.tensor(
        train_wide.values,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    past_observed_mask = torch.isfinite(past_values).float()
    past_values = torch.nan_to_num(past_values, nan=0.0)

    context_length = getattr(model.config, "context_length", past_values.shape[1])
    if past_values.shape[1] < context_length:
        pad_length = context_length - past_values.shape[1]
        past_values = torch.nn.functional.pad(past_values, (0, 0, pad_length, 0), value=0.0)
        past_observed_mask = torch.nn.functional.pad(
            past_observed_mask,
            (0, 0, pad_length, 0),
            value=0.0,
        )
    elif past_values.shape[1] > context_length:
        past_values = past_values[:, -context_length:, :]
        past_observed_mask = past_observed_mask[:, -context_length:, :]

    model_kwargs = {}
    if getattr(model.config, "resolution_prefix_tuning", False):
        model_kwargs["freq_token"] = torch.full(
            (past_values.shape[0],),
            int(frequency_token),
            dtype=torch.long,
            device=device,
        )

    with torch.no_grad():
        outputs = model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            return_loss=False,
            **model_kwargs,
        )

    forecast = outputs.prediction_outputs[0, :horizon, :].detach().cpu().numpy()
    timestamps = future_month_starts(train_wide.index.max(), horizon)

    pred_wide = pd.DataFrame(forecast, columns=series_ids)
    pred_wide.insert(0, "timestamp", timestamps)
    pred_long = pred_wide.melt(id_vars="timestamp", var_name="id", value_name="q50")
    return pred_long[["id", "timestamp", "q50"]]


def main(
    model_path,
    data_dir,
    output_dir,
    device,
    freq,
    freq_prefix_tuning,
    prefer_l1_loss,
    prefer_longer_context,
    force_return,
    frequency_token,
    datasets_filter,
):
    model_name = model_path.split("/")[-1]

    datasets = discover_datasets(data_dir)
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets")
    print(f"Modelo: {model_path}")
    print("Modo: multivariado nativo (todos os canais juntos)\n")

    all_predictions = []

    for dataset_name, paths in datasets.items():
        print(f"{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        try:
            df_train_long, df_test_long, horizon = DataLoader.load_dataset(
                paths["train"],
                paths["test"],
            )
            info = DataLoader.get_dataset_info(df_train_long, df_test_long, horizon)
            print(f"Séries: {info['n_series']} | Train: {info['train_length']} | Horizon: {horizon}")

            print("Carregando TTM adequado ao contexto/horizonte...")
            model = get_model(
                model_path=model_path,
                model_name="ttm",
                context_length=info["train_length"],
                prediction_length=horizon,
                freq=freq,
                freq_prefix_tuning=freq_prefix_tuning,
                prefer_l1_loss=prefer_l1_loss,
                prefer_longer_context=prefer_longer_context,
                force_return=force_return,
            )
            model = model.to(device)
            model.eval()
            print(
                "✓ Modelo carregado "
                f"(context={model.config.context_length}, prediction={model.config.prediction_length})"
            )

            print("Gerando previsões...")
            pred_df = forecast_ttm(model, df_train_long, horizon, device, frequency_token)
            pred_df["dataset"] = dataset_name

            all_predictions.append(pred_df)

            print("✓ Dataset processado\n")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBM Granite TinyTimeMixer Forecasting")
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--output-dir", type=str, default="data/predictions")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Frequência para seleção de modelo TTM. Use None para datasets mensais curtos.",
    )
    parser.add_argument("--freq-prefix-tuning", action="store_true")
    parser.add_argument("--prefer-l1-loss", action="store_true")
    parser.add_argument("--prefer-longer-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--frequency-token",
        type=int,
        default=0,
        help="Token de frequência para checkpoints com resolution prefix tuning. 0=oov, usado para frequência mensal.",
    )
    parser.add_argument(
        "--force-return",
        type=str,
        default="zeropad",
        choices=["zeropad", "rolling", "random_init_small", "random_init_medium", "random_init_large", None],
        help="Fallback do get_model quando contexto/horizonte não casam com checkpoints TTM.",
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        freq=args.freq,
        freq_prefix_tuning=args.freq_prefix_tuning,
        prefer_l1_loss=args.prefer_l1_loss,
        prefer_longer_context=args.prefer_longer_context,
        force_return=args.force_return,
        frequency_token=args.frequency_token,
        datasets_filter=args.datasets,
    )
