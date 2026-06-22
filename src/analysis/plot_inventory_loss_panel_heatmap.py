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


def load_component_scores(components_path: str) -> pd.DataFrame:
    path = Path(components_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    required = {"dataset", "model", "component", "normalized_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} sem colunas obrigatórias: {sorted(missing)}")

    df = df[np.isfinite(df["normalized_loss"])].copy()
    df["panel"] = df["dataset"].map(panel_name)
    return df


def load_cd_ranks(ranks_path: str | None) -> pd.DataFrame | None:
    if not ranks_path:
        return None

    path = Path(ranks_path)
    if not path.exists():
        return None

    ranks = pd.read_csv(path)
    required = {"classifier_name", "avg_rank", "mean_normalized_loss"}
    missing = required - set(ranks.columns)
    if missing:
        raise ValueError(f"{path} sem colunas obrigatórias: {sorted(missing)}")

    return ranks.rename(columns={"classifier_name": "model"})


def build_panel_loss_table(component_df: pd.DataFrame, ranks_df: pd.DataFrame | None) -> pd.DataFrame:
    dataset_loss = (
        component_df.groupby(["dataset", "panel", "model"], as_index=False)
        .agg(normalized_loss=("normalized_loss", "mean"))
    )

    panel_loss = (
        dataset_loss.groupby(["model", "panel"], as_index=False)
        .agg(normalized_loss=("normalized_loss", "mean"), datasets=("dataset", "nunique"))
    )

    table = panel_loss.pivot_table(
        index="model",
        columns="panel",
        values="normalized_loss",
        aggfunc="mean",
    ).reset_index()

    available_panels = [panel for panel in PANEL_ORDER if panel in table.columns]
    table["Overall"] = table[available_panels].mean(axis=1)
    table["pretty_model"] = table["model"].map(pretty_model_name)
    table["group"] = table["model"].map(model_group)
    table["color"] = table["model"].map(model_color)

    if ranks_df is not None:
        table = table.merge(
            ranks_df[["model", "avg_rank", "mean_normalized_loss"]],
            on="model",
            how="left",
        )
    else:
        table["avg_rank"] = table[available_panels].rank(axis=0, method="average", ascending=True).mean(axis=1)
        table["mean_normalized_loss"] = table["Overall"]

    table = table.sort_values(["avg_rank", "mean_normalized_loss", "pretty_model"], ascending=[True, True, True])

    ordered_cols = [
        "model",
        "pretty_model",
        "group",
        "avg_rank",
        "mean_normalized_loss",
        *available_panels,
        "Overall",
    ]
    return table[ordered_cols]


def plot_heatmap(table: pd.DataFrame, output_path: Path, top_n: int | None) -> None:
    panel_cols = [panel for panel in PANEL_ORDER if panel in table.columns]
    value_cols = [*panel_cols, "Overall"]
    plot_df = table.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n).copy()

    values = plot_df[value_cols].to_numpy(dtype=float)
    fig_height = max(5.5, 0.42 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))

    cmap = plt.get_cmap("YlOrRd")
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0.0, vmax=max(1.0, np.nanmax(values)))

    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["pretty_model"])
    ax.set_xlabel("Business panel")
    ax.set_ylabel("Model")

    for tick_label, model_name in zip(ax.get_yticklabels(), plot_df["model"]):
        tick_label.set_color(model_color(model_name))

    if "Overall" in value_cols:
        overall_idx = value_cols.index("Overall")
        ax.axvline(overall_idx - 0.5, color="black", linewidth=1.2)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = "white" if value >= 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized Inventory Demand Loss, lower is better")

    legend_handles = [
        Patch(facecolor=GROUP_COLORS[group], label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(table: pd.DataFrame, output_dir: str, top_n: int | None) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "inventory_loss_panel_heatmap.csv"
    plot_base = output_path / "inventory_loss_panel_heatmap"

    table.to_csv(csv_path, index=False)
    plot_heatmap(table, plot_base, top_n)
    return csv_path, plot_base.with_suffix(".pdf"), plot_base.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(
        description="Heatmap modelo x painel com Inventory Demand Loss normalizada por componente."
    )
    parser.add_argument(
        "--components",
        type=str,
        default="data/analysis/cd_diagram/inventory_loss_cd_components_normalized.csv",
        help="CSV de componentes normalizados gerado pelo cd_diagram.",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="data/analysis/cd_diagram/inventory_loss_cd_ranks.csv",
        help="CSV de ranks médios gerado pelo cd_diagram.",
    )
    parser.add_argument("--output-dir", type=str, default="data/analysis/plots")
    parser.add_argument("--top-n", type=int, default=0, help="Número de modelos no heatmap. 0 mostra todos.")
    args = parser.parse_args()

    top_n = None if args.top_n == 0 else args.top_n
    components = load_component_scores(args.components)
    ranks = load_cd_ranks(args.ranks)
    table = build_panel_loss_table(components, ranks)
    paths = save_outputs(table, args.output_dir, top_n)

    print("Menor Inventory Demand Loss normalizada média:")
    print(table[["pretty_model", "avg_rank", "mean_normalized_loss", "Overall"]].head(10).to_string(index=False))
    print("\nArquivos salvos:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
