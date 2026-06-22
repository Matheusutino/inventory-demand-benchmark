import argparse
import os

import pandas as pd

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


def quantile_col_name(quantile: float) -> str:
    return f"q{int(round(quantile * 100))}"


def is_quantile_column(col) -> bool:
    return isinstance(col, str) and col.startswith("q")


def load_pipeline(tabpfn_mode: str, max_context_length: int, output_selection: str):
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    modes = {
        "client": TabPFNMode.CLIENT,
        "local": TabPFNMode.LOCAL,
    }

    return TabPFNTSPipeline(
        max_context_length=max_context_length,
        tabpfn_mode=modes[tabpfn_mode],
        tabpfn_output_selection=output_selection,
    )


def normalize_prediction_columns(pred_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    pred_df = pred_df.reset_index()

    if "item_id" in pred_df.columns:
        pred_df = pred_df.rename(columns={"item_id": "id"})
    elif "id" not in pred_df.columns:
        pred_df["id"] = "0"

    rename_map = {}
    quantile_names = {round(q, 6): quantile_col_name(q) for q in quantiles}
    for col in pred_df.columns:
        if isinstance(col, (int, float)) and round(float(col), 6) in quantile_names:
            rename_map[col] = quantile_names[round(float(col), 6)]
            continue

        try:
            numeric_col = round(float(str(col)), 6)
        except ValueError:
            continue

        if numeric_col in quantile_names:
            rename_map[col] = quantile_names[numeric_col]

    pred_df = pred_df.rename(columns=rename_map)
    if "q50" not in pred_df.columns and "target" in pred_df.columns:
        pred_df["q50"] = pred_df["target"]

    pred_df["id"] = pred_df["id"].astype(str)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

    expected_quantile_cols = [quantile_col_name(q) for q in quantiles]
    quantile_cols = [col for col in expected_quantile_cols if col in pred_df.columns]
    quantile_cols.extend(
        col for col in pred_df.columns if is_quantile_column(col) and col not in quantile_cols
    )
    return pred_df[["id", "timestamp", *quantile_cols]]


def forecast_tabpfn_ts(
    pipeline,
    df_train_long: pd.DataFrame,
    horizon: int,
    quantiles: list[float],
) -> pd.DataFrame:
    context_df = df_train_long[["id", "timestamp", "target"]].copy()
    context_df = context_df.rename(columns={"id": "item_id"})
    context_df["item_id"] = context_df["item_id"].astype(str)
    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])

    pred_df = pipeline.predict_df(
        context_df=context_df,
        prediction_length=horizon,
        quantiles=quantiles,
    )
    return normalize_prediction_columns(pred_df, quantiles)


def main(
    data_dir,
    output_dir,
    tabpfn_mode,
    max_context_length,
    output_selection,
    quantiles,
    datasets_filter,
    disable_telemetry,
):
    if disable_telemetry:
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    if "TABPFN_TOKEN" not in os.environ:
        print(
            "TabPFN-TS requer aceite de licença para inferência local/client.\n"
            "Abra https://ux.priorlabs.ai, aceite a licença e exporte TABPFN_TOKEN.\n"
            "Exemplo: export TABPFN_TOKEN=\"<sua-api-key>\"",
        )
        raise SystemExit(2)

    model_name = f"tabpfn-ts-{tabpfn_mode}"

    datasets = discover_datasets(data_dir)
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets")
    print(f"Modelo: TabPFN-TS ({tabpfn_mode})")
    print("Modo: séries tratadas independentemente pelo TabPFN-TS\n")

    print("Carregando TabPFN-TS...")
    pipeline = load_pipeline(
        tabpfn_mode=tabpfn_mode,
        max_context_length=max_context_length,
        output_selection=output_selection,
    )
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
            pred_df = forecast_tabpfn_ts(pipeline, df_train_long, horizon, quantiles)
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
            col for col in final_predictions.columns if is_quantile_column(col)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN-TS Forecasting")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--output-dir", type=str, default="data/predictions")
    parser.add_argument("--tabpfn-mode", type=str, default="local", choices=["local", "client"])
    parser.add_argument("--max-context-length", type=int, default=32768)
    parser.add_argument(
        "--output-selection",
        type=str,
        default="median",
        choices=["mean", "median", "mode"],
    )
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--disable-telemetry", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tabpfn_mode=args.tabpfn_mode,
        max_context_length=args.max_context_length,
        output_selection=args.output_selection,
        quantiles=args.quantiles,
        datasets_filter=args.datasets,
        disable_telemetry=args.disable_telemetry,
    )
