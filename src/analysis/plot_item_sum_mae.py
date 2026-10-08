import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.utils.data_loader import DataLoader, discover_datasets

from src.analysis.model_style import (
    FAMILY_COLORS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    model_color,
    model_marker,
    model_family,
    pretty_model_name,
)


DATASET_LABELS = {
    "Compressores Herméticos": "Hermetic Compressors",
    "Filtros": "Filters",
    "Mecanismos": "Mechanisms",
    "Motores Ventiladores": "Fan Motors",
    "Placas de Controle": "Control Boards",
}

PANEL_ORDER = [
    "Hermetic Compressors",
    "Filters",
    "Mechanisms",
    "Fan Motors",
    "Control Boards",
]


def panel_name(dataset_name: str) -> str:
    raw_panel = dataset_name.rsplit("_h", 1)[0] if "_h" in dataset_name else dataset_name
    return DATASET_LABELS.get(raw_panel, raw_panel)


def horizon_label(dataset_name: str) -> str:
    if "_h" in dataset_name:
        suffix = dataset_name.rsplit("_h", 1)[1]
        if suffix.isdigit():
            return f"h{suffix}"
    return "unknown"


def model_name_from_path(path: Path) -> str:
    return path.stem.replace("_all_datasets_predictions", "")


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


def prediction_files(predictions_dir: str, models: list[str] | None) -> list[Path]:
    pred_dir = Path(predictions_dir)
    if models:
        files = []
        for model_name in models:
            path = pred_dir / f"{model_name}_all_datasets_predictions.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Arquivo de predição não encontrado: {path}")
            files.append(path)
        return files

    files = sorted(pred_dir.glob("*_all_datasets_predictions.parquet"))
    if not files:
        raise ValueError(f"Nenhum arquivo *_all_datasets_predictions.parquet encontrado em '{predictions_dir}'.")
    return files


def compute_item_sum_mae(data_dir: str, predictions_dir: str, models: list[str] | None) -> pd.DataFrame:
    truth_by_dataset = load_truth_by_dataset(data_dir)
    rows = []

    for pred_path in prediction_files(predictions_dir, models):
        model_name = model_name_from_path(pred_path)
        pred_df = pd.read_parquet(pred_path).copy()
        required_cols = {"dataset", "id", "timestamp", "q50"}
        missing = required_cols - set(pred_df.columns)
        if missing:
            print(f"Aviso: pulando {model_name}; faltam colunas {sorted(missing)}.")
            continue

        pred_df["id"] = pred_df["id"].astype(str)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        for dataset_name, truth_df in truth_by_dataset.items():
            model_pred = pred_df[pred_df["dataset"] == dataset_name].copy()
            if model_pred.empty:
                continue

            merged = truth_df.merge(
                model_pred[["id", "timestamp", "q50"]],
                on=["id", "timestamp"],
                how="inner",
            )
            if merged.empty:
                continue

            merged = merged[np.isfinite(merged["q50"])].copy()
            if merged.empty:
                print(f"Aviso: pulando {model_name}/{dataset_name}; q50 sem valores finitos.")
                continue

            merged["abs_error"] = (merged["target"] - merged["q50"]).abs()
            item_mae = float(
                merged.groupby("id")["abs_error"]
                .mean()
                .mean()
            )

            true_sum = merged.groupby("timestamp", as_index=False)["target"].sum()
            pred_sum = merged.groupby("timestamp", as_index=False)["q50"].sum()
            sum_merged = true_sum.merge(pred_sum, on="timestamp", how="inner")
            sum_mae = float((sum_merged["target"] - sum_merged["q50"]).abs().mean())

            rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_label": f"{panel_name(dataset_name)} {horizon_label(dataset_name)}",
                    "panel": panel_name(dataset_name),
                    "horizon": horizon_label(dataset_name),
                    "model": model_name,
                    "pretty_model": pretty_model_name(model_name),
                    "group": model_family(model_name),
                    "color": model_color(model_name),
                    "item_mae": item_mae,
                    "sum_mae": sum_mae,
                    "matched_rows": int(len(merged)),
                    "n_series": int(merged["id"].nunique()),
                }
            )

    if not rows:
        raise ValueError("Nenhuma combinação de modelo/dataset válida para MAE.")

    return pd.DataFrame(rows)


def aggregate_tables(by_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_panel = (
        by_dataset.groupby(["panel", "model", "pretty_model", "group", "color"], as_index=False)
        .agg(
            item_mae=("item_mae", "mean"),
            sum_mae=("sum_mae", "mean"),
            datasets=("dataset", "nunique"),
            n_series_mean=("n_series", "mean"),
        )
    )
    by_panel["panel"] = pd.Categorical(by_panel["panel"], categories=PANEL_ORDER, ordered=True)
    by_panel = by_panel.sort_values(["panel", "pretty_model"])

    overall = (
        by_dataset.groupby(["model", "pretty_model", "group", "color"], as_index=False)
        .agg(
            item_mae=("item_mae", "mean"),
            sum_mae=("sum_mae", "mean"),
            datasets=("dataset", "nunique"),
        )
        .sort_values("item_mae")
    )
    return by_panel, overall


def save_figure(fig, output_path: Path) -> None:
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def plot_by_panel(by_panel: pd.DataFrame, output_path: Path, log_scale: bool) -> None:
    models = by_panel[["model", "pretty_model"]].drop_duplicates().sort_values("pretty_model")
    panels = [panel for panel in PANEL_ORDER if panel in set(by_panel["panel"].astype(str))]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    metrics = [("item_mae", "Item-level MAE"), ("sum_mae", "Aggregate-sum MAE")]

    for ax, (metric, metric_label) in zip(axes, metrics):
        for _, model_row in models.iterrows():
            model = model_row["model"]
            pretty = model_row["pretty_model"]
            model_df = by_panel[by_panel["model"] == model].set_index("panel")
            values = [model_df.loc[panel, metric] if panel in model_df.index else np.nan for panel in panels]
            ax.plot(
                panels,
                values,
                marker=model_marker(model),
                linewidth=1.8,
                markersize=4.5,
                label=pretty,
                color=model_color(model),
            )

        ax.set_ylabel(f"{metric_label} (demand units)")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        if log_scale:
            ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_overall(overall: pd.DataFrame, output_path: Path, log_scale: bool) -> None:
    metrics = [("item_mae", "Item-level MAE"), ("sum_mae", "Aggregate-sum MAE")]
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, 0.38 * len(overall))), sharey=False)

    for ax, (metric, metric_label) in zip(axes, metrics):
        plot_df = overall.sort_values(metric, ascending=True).copy()
        colors = [model_color(model) for model in plot_df["model"]]
        y = np.arange(len(plot_df))

        ax.barh(y, plot_df[metric], color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["pretty_model"])
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric_label}: mean across all datasets")
        ax.grid(axis="x", alpha=0.25)
        if log_scale:
            ax.set_xscale("log")

        max_value = float(plot_df[metric].max())
        for y_pos, value in zip(y, plot_df[metric]):
            if log_scale:
                text_x = float(value) * 1.04 if value > 0 else max_value * 0.01
            else:
                text_x = float(value) + max_value * 0.01
            ax.text(
                text_x,
                y_pos,
                f"{value:,.1f}",
                va="center",
                ha="left",
                fontsize=8,
            )

        if log_scale:
            ax.set_xlim(right=max_value * 1.8)
        else:
            ax.set_xlim(right=max_value * 1.18)

    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[group], label=FAMILY_LABELS[group])
        for group in FAMILY_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_figure(fig, output_path)
    plt.close(fig)


def save_outputs(by_dataset: pd.DataFrame, by_panel: pd.DataFrame, overall: pd.DataFrame, output_dir: str) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    by_dataset_path = output_path / "item_sum_mae_by_dataset.csv"
    by_panel_path = output_path / "item_sum_mae_by_panel.csv"
    overall_path = output_path / "item_sum_mae_overall.csv"

    by_dataset.to_csv(by_dataset_path, index=False)
    by_panel.to_csv(by_panel_path, index=False)
    overall.to_csv(overall_path, index=False)
    return by_dataset_path, by_panel_path, overall_path


def split_csv_or_space(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out or None


def main():
    parser = argparse.ArgumentParser(description="Plota MAE item-level e MAE agregado por soma.")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions")
    parser.add_argument("--output-dir", type=str, default="data/analysis/accuracy/item_sum_mae")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--linear-scale", action="store_true", help="Usa escala linear em vez de log.")
    args = parser.parse_args()

    models = split_csv_or_space(args.models)
    by_dataset = compute_item_sum_mae(args.data_dir, args.predictions_dir, models)
    by_panel, overall = aggregate_tables(by_dataset)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_paths = save_outputs(by_dataset, by_panel, overall, args.output_dir)

    log_scale = not args.linear_scale
    by_panel_plot = output_path / "item_sum_mae_by_panel.pdf"
    overall_plot = output_path / "item_sum_mae_overall.pdf"
    plot_by_panel(by_panel, by_panel_plot, log_scale=log_scale)
    plot_overall(overall, overall_plot, log_scale=log_scale)

    print("Top modelos por Item MAE médio:")
    print(overall[["pretty_model", "item_mae", "sum_mae"]].head(10).to_string(index=False))
    print("\nArquivos salvos:")
    for path in table_paths:
        print(f"  {path}")
    print(f"  {by_panel_plot}")
    print(f"  {overall_plot}")


if __name__ == "__main__":
    main()
