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
    PANEL_ORDER,
    model_color,
    model_group,
    pretty_model_name,
)


PANEL_LABELS = {
    "Compressores Herméticos": "Hermetic Compressors",
    "Filtros": "Filters",
    "Mecanismos": "Mechanisms",
    "Motores Ventiladores": "Fan Motors",
    "Placas de Controle": "Control Boards",
}


def panel_name(dataset_name: str) -> str:
    raw_panel = dataset_name.rsplit("_h", 1)[0] if "_h" in dataset_name else dataset_name
    return PANEL_LABELS.get(raw_panel, raw_panel)


def load_inventory_performance(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    required = {"classifier_name", "dataset_name", "normalized_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} sem colunas obrigatórias: {sorted(missing)}")

    df = df.rename(columns={"classifier_name": "model", "dataset_name": "dataset"})
    df = df[np.isfinite(df["normalized_loss"])].copy()
    df["panel"] = df["dataset"].map(panel_name)
    df["pretty_model"] = df["model"].map(pretty_model_name)
    df["group"] = df["model"].map(model_group)
    df["color"] = df["model"].map(model_color)
    return df


def compute_panel_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_loss = (
        df.groupby(["model", "pretty_model", "group", "color", "panel"], as_index=False)
        .agg(normalized_loss=("normalized_loss", "mean"), datasets=("dataset", "nunique"))
    )
    panel_loss["rank"] = panel_loss.groupby("panel")["normalized_loss"].rank(method="average", ascending=True)

    rank_table = panel_loss.pivot_table(
        index=["model", "pretty_model", "group", "color"],
        columns="panel",
        values="rank",
        aggfunc="mean",
    ).reset_index()

    available_panels = [panel for panel in PANEL_ORDER if panel in rank_table.columns]
    rank_table["avg_rank"] = rank_table[available_panels].mean(axis=1)
    rank_table = rank_table.sort_values(["avg_rank", "pretty_model"], ascending=[True, True])

    loss_table = panel_loss.pivot_table(
        index=["model", "pretty_model", "group", "color"],
        columns="panel",
        values="normalized_loss",
        aggfunc="mean",
    ).reset_index()
    loss_table["avg_normalized_loss"] = loss_table[available_panels].mean(axis=1)
    loss_table = loss_table.sort_values(["avg_normalized_loss", "pretty_model"], ascending=[True, True])

    return rank_table, loss_table


def plot_rank_heatmap(rank_table: pd.DataFrame, output_path: Path, top_n: int | None) -> None:
    panels = [panel for panel in PANEL_ORDER if panel in rank_table.columns]
    plot_df = rank_table.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n).copy()

    values = plot_df[panels].to_numpy(dtype=float)

    fig_height = max(4.8, 0.42 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=1, vmax=np.nanmax(values))

    ax.set_xticks(np.arange(len(panels)))
    ax.set_xticklabels(panels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["pretty_model"])
    ax.set_xlabel("Business panel")
    ax.set_ylabel("Model")

    for tick_label, model_name in zip(ax.get_yticklabels(), plot_df["model"]):
        tick_label.set_color(model_color(model_name))

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Average rank, lower is better")

    legend_handles = [
        Patch(facecolor=GROUP_COLORS[group], label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    rank_table: pd.DataFrame,
    loss_table: pd.DataFrame,
    output_dir: str,
    top_n: int | None,
) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rank_path = output_path / "dataset_panel_inventory_rank_heatmap.csv"
    loss_path = output_path / "dataset_panel_inventory_loss.csv"
    plot_base = output_path / "dataset_panel_inventory_rank_heatmap"

    rank_table.to_csv(rank_path, index=False)
    loss_table.to_csv(loss_path, index=False)
    plot_rank_heatmap(rank_table, plot_base, top_n)

    return rank_path, loss_path, plot_base.with_suffix(".pdf"), plot_base.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description="Gera heatmap de rank por painel usando InventoryDemandLoss normalizada.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/analysis/cd_diagram/inventory_loss_cd_performance.csv",
        help="CSV gerado pelo cd_diagram com normalized_loss.",
    )
    parser.add_argument("--output-dir", type=str, default="data/analysis/plots")
    parser.add_argument("--top-n", type=int, default=10, help="Número de modelos no heatmap. Use 0 para todos.")
    args = parser.parse_args()

    top_n = None if args.top_n == 0 else args.top_n
    df = load_inventory_performance(args.input)
    rank_table, loss_table = compute_panel_tables(df)
    paths = save_outputs(rank_table, loss_table, args.output_dir, top_n)

    print("Top modelos por rank médio em InventoryDemandLoss normalizada:")
    print(rank_table[["pretty_model", "avg_rank"]].head(top_n or len(rank_table)).to_string(index=False))
    print("\nArquivos salvos:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
