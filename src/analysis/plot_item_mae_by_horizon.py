"""Compare Item (MAE) at H3 and H6 with equal weight per business panel."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.analysis.model_style import (
    FAMILY_COLORS, FAMILY_LABELS, FAMILY_ORDER, model_color, model_family, pretty_model_name,
)
from src.analysis.plot_item_sum_mae import compute_item_sum_mae, split_csv_or_space


def build_horizon_table(by_dataset: pd.DataFrame) -> pd.DataFrame:
    """Average per-SKU MAE across the same panels separately for each horizon."""
    selected = by_dataset.loc[by_dataset.horizon.isin(["h3", "h6"])].copy()
    keys = ["model", "panel", "horizon"]
    if selected.empty or selected.duplicated(keys).any():
        raise ValueError("É necessário um resultado único por modelo/painel/horizonte.")
    values = selected.item_mae.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Item MAE deve ser finito e não negativo.")
    expected = pd.MultiIndex.from_product(
        [selected.model.unique(), selected.panel.unique(), ["h3", "h6"]], names=keys,
    )
    if len(expected.difference(pd.MultiIndex.from_frame(selected[keys]))):
        raise ValueError("H3 e H6 devem cobrir os mesmos painéis e modelos.")
    means = selected.groupby(["model", "horizon"]).item_mae.mean().unstack("horizon")
    table = means.rename(columns={"h3": "item_mae_h3", "h6": "item_mae_h6"}).reset_index()
    table.columns.name = None
    table["pretty_model"] = table.model.map(pretty_model_name)
    table["family"] = table.model.map(model_family)
    table["color"] = table.model.map(model_color)
    table["n_panels"] = selected.panel.nunique()
    table["mean_item_mae"] = table[["item_mae_h3", "item_mae_h6"]].mean(axis=1)
    return table.sort_values(["mean_item_mae", "pretty_model"]).reset_index(drop=True)


def plot_item_mae_by_horizon(table: pd.DataFrame, output_path: Path) -> None:
    """Show H3 and H6 on the same model ordering and linear MAE scale."""
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(
            1, 2, figsize=(16, max(6, 0.38 * len(table) + 1.5)), sharex=True, sharey=True,
        )
        y = np.arange(len(table))
        maximum = float(table[["item_mae_h3", "item_mae_h6"]].to_numpy().max())
        pad = max(maximum * 0.015, 0.1)
        for ax, horizon in zip(axes, ("h3", "h6")):
            values = table[f"item_mae_{horizon}"]
            ax.barh(y, values, color=table.color, zorder=2)
            ax.set_yticks(y, labels=table.pretty_model)
            ax.set_xlabel(f"Item MAE — {horizon.upper()} (demand units)")
            ax.grid(axis="x", alpha=0.25, zorder=0)
            ax.set_xlim(0, max(maximum * 1.18, 1))
            for position, value in zip(y, values):
                ax.text(value + pad, position, f"{value:,.1f}", va="center", fontsize=9)
        axes[0].invert_yaxis()
        present = [family for family in FAMILY_ORDER if family in set(table.family)]
        fig.legend(
            handles=[Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in present],
            loc="lower center", ncol=len(present), fontsize=9, frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output-dir", default="data/analysis/accuracy/item_mae_by_horizon")
    args = parser.parse_args()
    by_dataset = compute_item_sum_mae(args.data_dir, args.predictions_dir, split_csv_or_space(args.models))
    table = build_horizon_table(by_dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "item_mae_h3_h6_overall.csv", index=False)
    plot_item_mae_by_horizon(table, output / "item_mae_h3_h6_overall")
    print(f"✓ Item MAE: {len(table)} modelos × H3/H6; {table.n_panels.iloc[0]} painéis com pesos iguais.")
    print(f"PDF e CSV: {output}")


if __name__ == "__main__":
    main()
