import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from src.utils.data_loader import DataLoader, discover_datasets


DEFAULT_METRICS = ["mae_mean"]

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
    "darts_randomforest_topdown": "Random Forest Top-Down",
    "darts_lightgbm_topdown": "LightGBM Top-Down",
    "darts_linear_topdown": "Linear Regression Top-Down",
    "darts_xgboost_topdown": "XGBoost Top-Down",
}

MODEL_GROUPS = {
    "darts_naive_moving_average": "industry",
    "darts_arima": "industry",
    "darts_prophet": "industry",
    "darts_linear": "classic_ml",
    "darts_randomforest": "classic_ml",
    "darts_lightgbm": "classic_ml",
    "darts_global_naive_aggregate": "classic_ml",
    "darts_global_naive_drift": "classic_ml",
    "darts_global_naive_seasonal": "classic_ml",
    "chronos-2": "foundation",
    "chronos-2-finetuned-lora": "foundation",
    "moirai2-small": "foundation",
    "sundial-base-128m": "foundation",
}

GROUP_COLORS = {
    "industry": "#4C78A8",
    "classic_ml": "#F58518",
    "foundation": "#54A24B",
}

GROUP_LABELS = {
    "industry": "Indústria",
    "classic_ml": "Clássicos IA",
    "foundation": "Foundation Models",
}

GROUP_ORDER = ["classic_ml", "industry", "foundation"]


def list_metric_files(metrics_dir: str):
    files = sorted(Path(metrics_dir).glob("*_metrics_agg.csv"))
    if not files:
        raise ValueError(f"Nenhum arquivo *_metrics_agg.csv encontrado em '{metrics_dir}'.")
    return files


def model_name_from_file(path: Path) -> str:
    return path.stem.replace("_all_datasets_metrics_agg", "").replace("_metrics_agg", "")


def pretty_model_name(model_name: str) -> str:
    if model_name in PRETTY_MODEL_NAMES:
        return PRETTY_MODEL_NAMES[model_name]

    normalized = model_name.lower()
    if "naive_moving_average" in normalized:
        return "Moving Average"
    if "prophet" in normalized:
        return "Prophet"
    if "arima" in normalized:
        return "ARIMA"
    if "randomforest" in normalized:
        return "Random Forest"
    if "lightgbm" in normalized:
        return "LightGBM"
    if "xgboost" in normalized:
        return "XGBoost"
    if normalized.startswith("chronos-2-finetuned"):
        return "Chronos-2 Fine-tuned (LoRA)"
    if normalized.startswith("chronos"):
        return "Chronos-2"
    if normalized.startswith("moirai2"):
        return "Moirai2-Small"
    if normalized.startswith("sundial"):
        return "Sundial Base 128M"
    return model_name.replace("darts_", "").replace("_", " ").title()


def model_group(model_name: str) -> str:
    if model_name in MODEL_GROUPS:
        return MODEL_GROUPS[model_name]

    normalized = model_name.lower()
    if any(token in normalized for token in ["moving_average", "arima", "prophet"]):
        return "industry"
    if any(token in normalized for token in ["chronos", "moirai", "sundial"]):
        return "foundation"
    return "classic_ml"


def model_color(model_name: str) -> str:
    return GROUP_COLORS[model_group(model_name)]


def list_prediction_files(predictions_dir: str):
    files = sorted(Path(predictions_dir).glob("*_all_datasets_predictions.parquet"))
    if not files:
        raise ValueError(f"Nenhum arquivo *_all_datasets_predictions.parquet encontrado em '{predictions_dir}'.")
    return files


def load_truth_by_dataset(data_dir: str) -> dict[str, pd.DataFrame]:
    datasets = discover_datasets(data_dir)
    truth = {}
    for dataset_name, paths in datasets.items():
        _, df_test_long, _ = DataLoader.load_dataset(paths["train"], paths["test"])
        df_test_long = df_test_long.copy()
        df_test_long["timestamp"] = pd.to_datetime(df_test_long["timestamp"])
        truth[dataset_name] = df_test_long
    return truth


def aggregate_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby("timestamp", as_index=False)[value_col]
        .sum()
        .sort_values("timestamp")
    )


def compute_metric(y_true: pd.Series, y_pred: pd.Series, metric: str) -> float:
    if metric == "mae_mean":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "mse_mean":
        return float(np.mean((y_true - y_pred) ** 2))
    if metric == "mape_mean":
        values = np.abs((y_true - y_pred) / y_true)
        return float(np.mean(values) * 100)
    raise ValueError(f"Métrica não suportada no modo aggregate-only: {metric}")


def load_all_metric_rows(metrics_dir: str, metrics: list[str]) -> pd.DataFrame:
    rows = []

    for path in list_metric_files(metrics_dir):
        df = pd.read_csv(path)
        model_name = model_name_from_file(path)

        if "dataset" not in df.columns:
            print(f"Aviso: pulando {path.name}: sem coluna 'dataset'.")
            continue

        required_cols = {"dataset", *metrics}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"Aviso: pulando {path.name}: faltam colunas {sorted(missing)}.")
            continue

        for _, row in df.iterrows():
            item = {
                "dataset": row["dataset"],
                "model": model_name,
                "pretty_model": pretty_model_name(model_name),
                "group": model_group(model_name),
                "color": model_color(model_name),
            }
            for metric in metrics:
                item[metric] = row[metric]
            rows.append(item)

    if not rows:
        raise ValueError("Nenhum arquivo válido encontrado para comparação.")

    return pd.DataFrame(rows)


def load_all_aggregate_only_rows(predictions_dir: str, data_dir: str, metrics: list[str]) -> pd.DataFrame:
    truth_by_dataset = load_truth_by_dataset(data_dir)
    rows = []

    for path in list_prediction_files(predictions_dir):
        model_name = model_name_from_file(path)
        pred_df = pd.read_parquet(path)
        pred_df = pred_df.copy()
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        for dataset_name, df_true in truth_by_dataset.items():
            df_pred = pred_df[pred_df["dataset"] == dataset_name].copy()
            if df_pred.empty:
                continue

            true_agg = aggregate_series(df_true, "target")
            pred_agg = aggregate_series(df_pred, "q50")
            merged = true_agg.merge(pred_agg, on="timestamp", how="inner")
            if merged.empty:
                continue

            item = {
                "dataset": dataset_name,
                "model": model_name,
                "pretty_model": pretty_model_name(model_name),
                "group": model_group(model_name),
                "color": model_color(model_name),
            }
            for metric in metrics:
                item[metric] = compute_metric(merged["target"], merged["q50"], metric)
            rows.append(item)

    if not rows:
        raise ValueError("Nenhum arquivo válido encontrado para comparação aggregate-only.")

    return pd.DataFrame(rows)


def ensure_all_datasets_row(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if "ALL_DATASETS" in set(df["dataset"]):
        return df

    rows = []
    for model_name, group in df.groupby("model", sort=True):
        item = {
            "dataset": "ALL_DATASETS",
            "model": model_name,
            "pretty_model": group["pretty_model"].iloc[0],
            "group": group["group"].iloc[0],
            "color": group["color"].iloc[0],
        }
        for metric in metrics:
            item[metric] = group[metric].mean()
        rows.append(item)

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def dataset_order(df: pd.DataFrame) -> list[str]:
    datasets = sorted(d for d in df["dataset"].unique() if d != "ALL_DATASETS")
    return ["ALL_DATASETS"] + datasets


def plot_dataset_panel(ax, df: pd.DataFrame, dataset_name: str, metric: str):
    plot_df = (
        df[df["dataset"] == dataset_name][["pretty_model", "color", metric]]
        .sort_values(metric, ascending=True)
    )

    ax.barh(plot_df["pretty_model"], plot_df[metric], color=plot_df["color"])
    title = "Todos os Datasets" if dataset_name == "ALL_DATASETS" else dataset_name
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.25)


def plot_metrics(df: pd.DataFrame, metrics: list[str], output_path: str):
    datasets = dataset_order(df)
    if len(metrics) != 1:
        raise ValueError("A versão atual suporta uma métrica por figura. Use apenas mae_mean.")

    metric = metrics[0]
    n_panels = len(datasets)
    ncols = 4
    nrows = math.ceil(n_panels / ncols)
    fig_height = max(5, nrows * 4.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=(28, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    for ax, dataset_name in zip(axes_flat, datasets):
        plot_dataset_panel(ax, df, dataset_name, metric)

    for ax in axes_flat[len(datasets):]:
        ax.axis("off")

    legend_handles = [
        Patch(facecolor=GROUP_COLORS[group], label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]

    fig.suptitle("Comparação Agregada por Dataset")
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plota comparação horizontal de métricas agregadas.")
    parser.add_argument("--metrics-dir", type=str, default="data/analysis/metrics", help="Diretório com arquivos *_metrics_agg.csv.")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions", help="Diretório com arquivos *_predictions.parquet.")
    parser.add_argument("--data-dir", type=str, default="data/datasets", help="Diretório com os datasets reais.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Métricas a plotar.")
    parser.add_argument("--aggregate-only", action="store_true", help="Gera apenas a versão usando a série total agregada.")
    parser.add_argument("--output", type=str, default=None, help="Arquivo de saída PNG.")
    args = parser.parse_args()

    metrics = args.metrics
    metrics_label = "_".join(metrics)
    output_dir = Path("data/analysis/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.aggregate_only:
        jobs.append(("aggregate_only", load_all_aggregate_only_rows(args.predictions_dir, args.data_dir, metrics)))
    else:
        jobs.append(("individual", load_all_metric_rows(args.metrics_dir, metrics)))
        jobs.append(("aggregate_only", load_all_aggregate_only_rows(args.predictions_dir, args.data_dir, metrics)))

    for mode, df in jobs:
        df = ensure_all_datasets_row(df, metrics)

        if args.output is not None:
            output_base = Path(args.output)
            if len(jobs) == 1:
                output_path = str(output_base)
            else:
                suffix = "_aggregate_only" if mode == "aggregate_only" else "_individual"
                output_path = str(output_base.with_name(f"{output_base.stem}{suffix}{output_base.suffix or '.png'}"))
        else:
            suffix = "_aggregate_only" if mode == "aggregate_only" else ""
            output_path = str(output_dir / f"comparison_all_datasets_and_panels_{metrics_label}{suffix}.png")

        plot_metrics(df, metrics, output_path)
        print(f"Gráfico salvo em: {output_path}")


if __name__ == "__main__":
    main()
