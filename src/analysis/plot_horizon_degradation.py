import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.analysis.plot_item_sum_mae import (
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    compute_item_sum_mae,
    model_color,
    split_csv_or_space,
)


def compute_degradation(by_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = by_dataset.pivot_table(
        index=["panel", "model", "pretty_model", "group", "color"],
        columns="horizon",
        values=["item_mae", "sum_mae"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{horizon}" for metric, horizon in pivot.columns]
    pivot = pivot.reset_index()

    required = {"item_mae_h3", "item_mae_h6", "sum_mae_h3", "sum_mae_h6"}
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"Não foi possível calcular degradação; faltam colunas {sorted(missing)}.")

    by_panel = pivot.dropna(subset=list(required)).copy()
    by_panel = by_panel[
        (by_panel["item_mae_h3"] > 0)
        & (by_panel["sum_mae_h3"] > 0)
        & np.isfinite(by_panel["item_mae_h3"])
        & np.isfinite(by_panel["item_mae_h6"])
        & np.isfinite(by_panel["sum_mae_h3"])
        & np.isfinite(by_panel["sum_mae_h6"])
    ].copy()

    by_panel["item_ratio_h6_h3"] = by_panel["item_mae_h6"] / by_panel["item_mae_h3"]
    by_panel["sum_ratio_h6_h3"] = by_panel["sum_mae_h6"] / by_panel["sum_mae_h3"]
    by_panel["item_pct_increase"] = (by_panel["item_ratio_h6_h3"] - 1.0) * 100.0
    by_panel["sum_pct_increase"] = (by_panel["sum_ratio_h6_h3"] - 1.0) * 100.0

    overall = (
        by_panel.groupby(["model", "pretty_model", "group", "color"], as_index=False)
        .agg(
            item_mae_h3=("item_mae_h3", "mean"),
            item_mae_h6=("item_mae_h6", "mean"),
            sum_mae_h3=("sum_mae_h3", "mean"),
            sum_mae_h6=("sum_mae_h6", "mean"),
            panels=("panel", "nunique"),
        )
    )
    overall["item_ratio_h6_h3"] = overall["item_mae_h6"] / overall["item_mae_h3"]
    overall["sum_ratio_h6_h3"] = overall["sum_mae_h6"] / overall["sum_mae_h3"]
    overall["item_pct_increase"] = (overall["item_ratio_h6_h3"] - 1.0) * 100.0
    overall["sum_pct_increase"] = (overall["sum_ratio_h6_h3"] - 1.0) * 100.0
    overall = overall.sort_values("item_pct_increase", ascending=True)

    return by_panel, overall


def plot_degradation(overall: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("item_pct_increase", "Item-level MAE"),
        ("sum_pct_increase", "Aggregate-sum MAE"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, 0.38 * len(overall))), sharey=False)

    for ax, (metric, title) in zip(axes, metrics):
        plot_df = overall.sort_values(metric, ascending=True).copy()
        y = np.arange(len(plot_df))
        colors = [model_color(model) for model in plot_df["model"]]

        ax.barh(y, plot_df[metric], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["pretty_model"])
        ax.invert_yaxis()
        ax.set_xlabel("H6 vs H3 MAE increase (%)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

        values = plot_df[metric].astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite))) if len(finite) else 1.0
        pad = max(4.0, max_abs * 0.04)
        for y_pos, value in zip(y, values):
            if not np.isfinite(value):
                continue
            ha = "left" if value >= 0 else "right"
            x = value + pad if value >= 0 else value - pad
            ax.text(x, y_pos, f"{value:+.1f}%", va="center", ha=ha, fontsize=8)

        lower = min(0.0, float(np.nanmin(values))) - max_abs * 0.25
        upper = max(0.0, float(np.nanmax(values))) + max_abs * 0.35
        ax.set_xlim(lower, upper)

    legend_handles = [
        Patch(facecolor=GROUP_COLORS[group], label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_outputs(by_panel: pd.DataFrame, overall: pd.DataFrame, output_dir: str) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    by_panel_path = output_path / "horizon_degradation_by_panel.csv"
    overall_path = output_path / "horizon_degradation_overall.csv"
    plot_path = output_path / "horizon_degradation_overall.pdf"

    by_panel.to_csv(by_panel_path, index=False)
    overall.to_csv(overall_path, index=False)
    plot_degradation(overall, plot_path)
    return by_panel_path, overall_path, plot_path


def main():
    parser = argparse.ArgumentParser(description="Plota degradação de MAE de H3 para H6.")
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions")
    parser.add_argument("--output-dir", type=str, default="data/analysis/plots")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    models = split_csv_or_space(args.models)
    by_dataset = compute_item_sum_mae(args.data_dir, args.predictions_dir, models)
    by_panel, overall = compute_degradation(by_dataset)
    paths = save_outputs(by_panel, overall, args.output_dir)

    print("Menor degradação média H6 vs H3:")
    print(
        overall[
            ["pretty_model", "item_pct_increase", "sum_pct_increase", "item_ratio_h6_h3", "sum_ratio_h6_h3"]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("\nArquivos salvos:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
