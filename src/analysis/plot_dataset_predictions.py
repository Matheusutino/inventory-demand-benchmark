import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from src.utils.data_loader import DataLoader, discover_datasets


PRETTY_MODEL_NAMES = {
    "darts_naive_moving_average": "Moving Average",
    "darts_arima": "ARIMA",
    "darts_prophet": "Prophet",
    "darts_randomforest": "Random Forest",
    "darts_lightgbm": "LightGBM",
    "darts_linear": "Linear Regression",
    "darts_global_naive_aggregate": "Global Naive Aggregate",
    "darts_global_naive_drift": "Global Naive Drift",
    "darts_global_naive_seasonal": "Global Naive Seasonal",
    "chronos-2": "Chronos-2",
    "chronos-2-finetuned-lora": "Chronos-2 Fine-tuned (LoRA)",
    "moirai2-small": "Moirai2-Small",
    "sundial-base-128m": "Sundial Base 128M",
}

MODEL_MARKERS = {
    "darts_naive_moving_average": "o",
    "darts_arima": "s",
    "darts_prophet": "^",
    "darts_randomforest": "D",
    "darts_lightgbm": "P",
    "darts_linear": "X",
    "darts_global_naive_aggregate": "v",
    "darts_global_naive_drift": "<",
    "darts_global_naive_seasonal": ">",
    "chronos-2": "h",
    "chronos-2-finetuned-lora": "H",
    "moirai2-small": "*",
    "sundial-base-128m": "8",
}


def pretty_model_name(model_name: str) -> str:
    return PRETTY_MODEL_NAMES.get(model_name, model_name.replace("darts_", "").replace("_", " ").title())


def model_marker(model_name: str) -> str:
    return MODEL_MARKERS.get(model_name, "o")


def load_dataset_truth(data_dir: str, dataset_name: str):
    datasets = discover_datasets(data_dir)
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado em '{data_dir}'.")

    paths = datasets[dataset_name]
    df_train_long, df_test_long, _ = DataLoader.load_dataset(paths["train"], paths["test"])
    df_train_long = df_train_long.copy()
    df_test_long = df_test_long.copy()
    df_train_long["id"] = df_train_long["id"].astype(str)
    df_test_long["id"] = df_test_long["id"].astype(str)
    return df_train_long, df_test_long


def discover_prediction_files(predictions_dir: str, model_names: list[str] | None):
    pred_dir = Path(predictions_dir)
    if model_names:
        files = []
        for model_name in model_names:
            path = pred_dir / f"{model_name}_all_datasets_predictions.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Arquivo de predição não encontrado: {path}")
            files.append(path)
        return files

    files = sorted(pred_dir.glob("*_all_datasets_predictions.parquet"))
    if not files:
        raise ValueError(f"Nenhum arquivo de previsão encontrado em '{predictions_dir}'.")
    return files


def load_predictions_for_dataset(predictions_dir: str, dataset_name: str, model_names: list[str] | None):
    model_frames = {}

    for path in discover_prediction_files(predictions_dir, model_names):
        model_name = path.stem.replace("_all_datasets_predictions", "")
        df = pd.read_parquet(path)
        df = df[df["dataset"] == dataset_name].copy()
        if df.empty:
            continue
        df["id"] = df["id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        model_frames[model_name] = df[["id", "timestamp", "q50"]]

    if not model_frames:
        raise ValueError(f"Nenhuma predição encontrada para o dataset '{dataset_name}'.")

    return model_frames


def plot_aggregate(train_df: pd.DataFrame, test_df: pd.DataFrame, pred_frames: dict, output_path: Path):
    test_agg = test_df.groupby("timestamp", as_index=False)["target"].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(test_agg["timestamp"], test_agg["target"], label="real_test", color="black", linewidth=2, linestyle="--")

    for model_name, pred_df in pred_frames.items():
        pred_agg = pred_df.groupby("timestamp", as_index=False)["q50"].sum()
        plt.plot(
            pred_agg["timestamp"],
            pred_agg["q50"],
            label=pretty_model_name(model_name),
            linewidth=1.8,
            marker=model_marker(model_name),
            markersize=6,
        )

    plt.title("Soma das Séries")
    plt.xlabel("Timestamp")
    plt.ylabel("Valor")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_individual_series(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_frames: dict,
    output_path: Path,
    selected_series: list[str] | None,
    max_series: int | None,
):
    all_ids = sorted(test_df["id"].unique())

    if selected_series:
        ids = [series_id for series_id in all_ids if series_id in selected_series]
    else:
        ids = all_ids

    if max_series is not None:
        ids = ids[:max_series]

    if not ids:
        raise ValueError("Nenhuma série selecionada para plot individual.")

    with PdfPages(output_path) as pdf:
        for series_id in ids:
            plt.figure(figsize=(12, 6))

            test_series = test_df[test_df["id"] == series_id].sort_values("timestamp")

            plt.plot(test_series["timestamp"], test_series["target"], label="real_test", color="black", linewidth=2, linestyle="--")

            for model_name, pred_df in pred_frames.items():
                model_series = pred_df[pred_df["id"] == series_id].sort_values("timestamp")
                if model_series.empty:
                    continue
                plt.plot(
                    model_series["timestamp"],
                    model_series["q50"],
                    label=pretty_model_name(model_name),
                    linewidth=1.6,
                    marker=model_marker(model_name),
                    markersize=5,
                )

            plt.title(f"Série {series_id}")
            plt.xlabel("Timestamp")
            plt.ylabel("Valor")
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_per_model_aggregate(test_df: pd.DataFrame, pred_frames: dict, output_dir: Path, dataset_name: str):
    """Gera uma figura agregada separada para cada modelo."""
    test_agg = test_df.groupby("timestamp", as_index=False)["target"].sum()

    for model_name, pred_df in pred_frames.items():
        pred_agg = pred_df.groupby("timestamp", as_index=False)["q50"].sum()
        pretty_name = pretty_model_name(model_name)

        plt.figure(figsize=(12, 6))
        plt.plot(test_agg["timestamp"], test_agg["target"], label="real_test", color="black", linewidth=2, linestyle="--")
        plt.plot(
            pred_agg["timestamp"],
            pred_agg["q50"],
            label=pretty_name,
            linewidth=1.8,
            marker=model_marker(model_name),
            markersize=6,
        )
        plt.title(f"Soma das Séries - {pretty_name}")
        plt.xlabel("Timestamp")
        plt.ylabel("Valor")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        output_path = output_dir / f"{dataset_name}_{model_name}_aggregate.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()


def plot_per_model_individual_series(
    test_df: pd.DataFrame,
    pred_frames: dict,
    output_dir: Path,
    dataset_name: str,
    selected_series: list[str] | None,
    max_series: int | None,
):
    """Gera um PDF individual por modelo com uma página por série."""
    all_ids = sorted(test_df["id"].unique())

    if selected_series:
        ids = [series_id for series_id in all_ids if series_id in selected_series]
    else:
        ids = all_ids

    if max_series is not None:
        ids = ids[:max_series]

    if not ids:
        raise ValueError("Nenhuma série selecionada para plot individual.")

    for model_name, pred_df in pred_frames.items():
        pretty_name = pretty_model_name(model_name)
        output_path = output_dir / f"{dataset_name}_{model_name}_individual_series.pdf"

        with PdfPages(output_path) as pdf:
            for series_id in ids:
                plt.figure(figsize=(12, 6))
                test_series = test_df[test_df["id"] == series_id].sort_values("timestamp")
                model_series = pred_df[pred_df["id"] == series_id].sort_values("timestamp")

                plt.plot(test_series["timestamp"], test_series["target"], label="real_test", color="black", linewidth=2, linestyle="--")
                if not model_series.empty:
                    plt.plot(
                        model_series["timestamp"],
                        model_series["q50"],
                        label=pretty_name,
                        linewidth=1.6,
                        marker=model_marker(model_name),
                        markersize=5,
                    )

                plt.title(f"Série {series_id} - {pretty_name}")
                plt.xlabel("Timestamp")
                plt.ylabel("Valor")
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
                plt.grid(alpha=0.25)
                plt.tight_layout()
                pdf.savefig()
                plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plota previsões individuais e agregadas para um dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Nome do dataset, ex: Mecanismos_h3")
    parser.add_argument("--data-dir", type=str, default="data/datasets", help="Diretório com datasets reais.")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions", help="Diretório com arquivos de previsão.")
    parser.add_argument("--models", nargs="+", default=None, help="Modelos específicos a comparar.")
    parser.add_argument("--series", nargs="+", default=None, help="IDs específicos para plot individual.")
    parser.add_argument("--max-series", type=int, default=None, help="Limite de séries no PDF individual.")
    parser.add_argument("--output-dir", type=str, default="data/analysis/plots", help="Diretório de saída.")
    parser.add_argument(
        "--per-model",
        action="store_true",
        help="Se ativado, gera também arquivos separados por modelo.",
    )
    args = parser.parse_args()

    train_df, test_df = load_dataset_truth(args.data_dir, args.dataset)
    pred_frames = load_predictions_for_dataset(args.predictions_dir, args.dataset, args.models)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_dir / f"{args.dataset}_aggregate_comparison.png"
    series_path = output_dir / f"{args.dataset}_individual_series.pdf"

    plot_aggregate(train_df, test_df, pred_frames, aggregate_path)
    plot_individual_series(train_df, test_df, pred_frames, series_path, args.series, args.max_series)

    print(f"Gráfico agregado salvo em: {aggregate_path}")
    print(f"PDF das séries individuais salvo em: {series_path}")

    if args.per_model:
        plot_per_model_aggregate(test_df, pred_frames, output_dir, args.dataset)
        plot_per_model_individual_series(test_df, pred_frames, output_dir, args.dataset, args.series, args.max_series)
        print(f"Gráficos agregados por modelo salvos em: {output_dir}")
        print(f"PDFs individuais por modelo salvos em: {output_dir}")


if __name__ == "__main__":
    main()
