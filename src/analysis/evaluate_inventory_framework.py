"""Evaluate the nine Inventory-Aware Evaluation Framework components independently."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.evaluation.forecast_evaluation import (
    DEFAULT_EVALUATION_CONFIG, compute_winners, evaluate_models,
    history_statistics, list_prediction_files, load_history_by_dataset, load_truth_by_dataset,
)
from src.analysis.inventory_framework import COMPONENTS, COMPONENT_LABELS, FRAMEWORK_NAME
from src.analysis.model_style import (
    FAMILY_COLORS, FAMILY_LABELS, FAMILY_ORDER, model_family, pretty_model_name,
)
from src.analysis.plot_inventory_component_rank_heatmap import compute_component_ranks


def save_win_figure(summary: pd.DataFrame, output: Path) -> None:
    """Show wins per component, with no cross-component total or overall ordering."""
    table = summary.copy()
    table["family"] = table.model.map(model_family)
    table["label"] = table.model.map(pretty_model_name)
    table["order"] = table.family.map({name: i for i, name in enumerate(FAMILY_ORDER)})
    table = table.sort_values(["order", "label"])
    values = table[list(COMPONENTS)].to_numpy()
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(12, 8.5))
        im = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0)
        ax.set_xticks(range(len(COMPONENTS)), labels=[COMPONENT_LABELS[c] for c in COMPONENTS], rotation=30, ha="right")
        ax.set_yticks(range(len(table)), labels=table.label)
        for i, family in enumerate(table.family):
            ax.get_yticklabels()[i].set_color(FAMILY_COLORS[family])
            if i and family != table.family.iloc[i - 1]:
                ax.axhline(i - .5, color="white", linewidth=2)
        for row in range(len(table)):
            for col in range(len(COMPONENTS)):
                ax.text(col, row, str(values[row, col]), ha="center", va="center",
                        color="white" if im.norm(values[row, col]) > .6 else "black", fontsize=9)
        ax.set_xlabel(f"{FRAMEWORK_NAME} component")
        fig.colorbar(im, ax=ax, fraction=.035, pad=.025).set_label("Task wins (ties credited to each model)")
        present = [f for f in FAMILY_ORDER if f in set(table.family)]
        fig.legend(handles=[Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in present],
                   loc="lower center", ncol=len(present), fontsize=8.5, frameon=False)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)


def save_history_outputs(truths: dict, histories: dict, output: Path) -> None:
    """Export per-SKU training scales and per-task exclusion counts without double-counting models."""
    sku_tables, rows = [], []
    for dataset, truth in truths.items():
        _, stats, metadata = history_statistics(histories[dataset], truth)
        stats.insert(0, "dataset", dataset)
        sku_tables.append(stats)
        rows.append({"dataset": dataset, "n_series": truth.id.nunique(),
                     "horizon": truth.timestamp.nunique(), **metadata})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(output / "component_coverage.csv", index=False)
    pd.concat(sku_tables, ignore_index=True).to_csv(output / "training_scales_by_sku.csv", index=False)
    lines = [
        "# Framework: training scales and component coverage", "",
        "Scaled uses the mean absolute lag-one change in each task's training split. "
        "SKUs with q_i = 0 are excluded from Scaled. Zero excludes SKUs whose training "
        "mean is zero, then averages forecast / training mean only over test observations "
        "with exactly zero demand. No epsilon replaces these denominators.", "",
        "Cap uses 2 × the maximum training demand per SKU, including a zero cap for "
        "SKUs with no training demand. All other components retain all SKUs. "
        "Each task uses its own forecast origin; H3 and H6 exclusions need not match.", "",
        "| Task | Training months | SKUs | Scaled excluded (constant history) | Zero excluded (no history demand) | Scaled observations | Zero observations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in coverage.itertuples():
        lines.append(f"| {row.dataset} | {row.n_train} | {row.n_series} | "
                     f"{row.mase_excluded_constant_skus} | {row.zero_excluded_no_demand_skus} | "
                     f"{row.mase_observations} | {row.zero_observations} |")
    lines.extend(["", "Exclusions summed across panels within each horizon (not across horizons):", ""])
    for horizon, group in coverage.groupby("horizon", sort=True):
        lines.append(f"- H{horizon}: {group.mase_excluded_constant_skus.sum()} constant-history SKUs "
                     f"excluded from Scaled; {group.zero_excluded_no_demand_skus.sum()} no-demand "
                     "history SKUs excluded from Zero.")
    lines.extend([
        "", "The H6 histories coincide with the common training window used in the dataset "
        "overview (Table 1). H3 additionally includes the three months available at its "
        "later forecast origin. A SKU can therefore be excluded in H6 and eligible in H3.", "",
        "Zero follows the supplied formula literally using signed saved forecasts, "
        "without clipping; negative forecasts can produce negative component values. "
        "Both Scaled and Zero favor zero forecasts. Asym (pinball tau=0.7) penalizes "
        "underprediction and Sum penalizes underestimating aggregate demand. "
        "Delta compares changes within the forecast window, without anchoring its first "
        "term to the last training observation.", "",
        "If a task has no eligible observations, Scaled or Zero is undefined (NaN); "
        "ranking rejects this task rather than replacing an undefined metric with zero.", "",
        "`training_scales_by_sku.csv` records every SKU's scale, mean, maximum, cap, "
        "eligibility and test-zero count. `component_coverage.csv` records task counts "
        "once per task, independently of the number of models.", "",
    ])
    (output / "component_coverage_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output-dir", default="data/analysis/framework/components")
    parser.add_argument("--wins-dir", default="data/analysis/framework/wins")
    args = parser.parse_args()
    truths = load_truth_by_dataset(args.data_dir)
    histories = load_history_by_dataset(args.data_dir)
    values = evaluate_models(truths, list_prediction_files(args.predictions_dir, args.models),
                             DEFAULT_EVALUATION_CONFIG, histories)
    compute_component_ranks(values)  # Require the same complete model/task coverage as the figures.
    winners, counts, summary = compute_winners(values)
    output, wins = Path(args.output_dir), Path(args.wins_dir)
    output.mkdir(parents=True, exist_ok=True)
    wins.mkdir(parents=True, exist_ok=True)
    values.to_csv(output / "component_values.csv", index=False)
    save_history_outputs(truths, histories, output)
    winners.to_csv(wins / "component_winners.csv", index=False)
    counts.to_csv(wins / "component_win_counts_long.csv", index=False)
    summary.to_csv(wins / "component_win_counts_summary.csv", index=False)
    save_win_figure(summary, wins / "component_win_counts_summary.pdf")
    print(f"✓ {FRAMEWORK_NAME}: {values.model.nunique()} modelos × {values.dataset.nunique()} tarefas × {len(COMPONENTS)} componentes.")
    print(f"Componentes: {output}; vitórias suplementares: {wins}")


if __name__ == "__main__":
    main()
