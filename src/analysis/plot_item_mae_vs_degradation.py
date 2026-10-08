import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.analysis.plot_horizon_degradation import compute_degradation
from src.analysis.model_style import (
    FAMILY_COLORS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    model_color,
)
from src.analysis.plot_item_sum_mae import (
    compute_item_sum_mae,
    split_csv_or_space,
)


def build_item_mae_degradation_table(data_dir: str, predictions_dir: str, models: list[str] | None) -> pd.DataFrame:
    by_dataset = compute_item_sum_mae(data_dir, predictions_dir, models)
    _, degradation = compute_degradation(by_dataset)
    overall_item = (
        by_dataset.groupby(["model", "pretty_model", "group", "color"], as_index=False)
        .agg(
            item_mae=("item_mae", "mean"),
            datasets=("dataset", "nunique"),
        )
    )
    table = overall_item.merge(
        degradation[["model", "item_pct_increase", "item_ratio_h6_h3", "panels"]],
        on="model",
        how="inner",
    )
    return table.sort_values("item_mae", ascending=True)


def plot_item_mae_vs_degradation(table: pd.DataFrame, output_path: Path, label_top_n: int) -> None:
    plot_df = table.sort_values("item_mae", ascending=True).copy()
    y = np.arange(len(plot_df))
    colors = [model_color(model) for model in plot_df["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, 0.38 * len(plot_df))), sharey=True)

    metrics = [
        ("item_mae", "{:,.1f}", "Mean item-level MAE across all datasets"),
        ("item_pct_increase", "{:+.1f}%", "H6 vs H3 item-level MAE increase (%)"),
    ]

    for ax, (metric, value_fmt, xlabel) in zip(axes, metrics):
        values = plot_df[metric].astype(float).to_numpy()
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["pretty_model"])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)

        if metric == "item_pct_increase":
            ax.axvline(0, color="#333333", linewidth=0.8)

        finite = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite))) if len(finite) else 1.0
        pad = max_abs * 0.015 if metric == "item_mae" else max(3.0, max_abs * 0.035)

        for y_pos, value in zip(y, values):
            if not np.isfinite(value):
                continue
            if metric == "item_pct_increase" and value < 0:
                x = value - pad
                ha = "right"
            else:
                x = value + pad
                ha = "left"
            ax.text(x, y_pos, value_fmt.format(value), va="center", ha=ha, fontsize=8)

        if metric == "item_pct_increase":
            upper = max(0.0, float(np.nanmax(values))) + max_abs * 0.28
            ax.set_xlim(0.0, upper)
        else:
            ax.set_xlim(right=float(np.nanmax(values)) * 1.18)

    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[group], label=FAMILY_LABELS[group])
        for group in FAMILY_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_outputs(table: pd.DataFrame, output_dir: str, label_top_n: int) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "item_mae_vs_horizon_degradation.csv"
    plot_base = output_path / "item_mae_vs_horizon_degradation"
    table.to_csv(csv_path, index=False)
    plot_item_mae_vs_degradation(table, plot_base, label_top_n=label_top_n)
    return csv_path, plot_base.with_suffix(".pdf")


def main():
    parser = argparse.ArgumentParser(description="Combina Item-level MAE médio e degradação H6 vs H3 em uma figura.")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions")
    parser.add_argument("--output-dir", type=str, default="data/analysis/diagnostics/mae_vs_horizon")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--label-top-n", type=int, default=5)
    args = parser.parse_args()

    models = split_csv_or_space(args.models)
    table = build_item_mae_degradation_table(args.data_dir, args.predictions_dir, models)
    paths = save_outputs(table, args.output_dir, label_top_n=args.label_top_n)

    print("Melhores modelos por Item-level MAE médio:")
    print(table[["pretty_model", "item_mae", "item_pct_increase"]].head(10).to_string(index=False))
    print("\nArquivos salvos:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
