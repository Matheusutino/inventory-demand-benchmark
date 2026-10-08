"""Compare Inventory-Aware Evaluation Framework rankings with Spearman rho and Kendall tau-b."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from src.analysis.inventory_framework import FRAMEWORK_NAME

from src.analysis.plot_inventory_component_rank_heatmap import (
    COMPONENTS, COMPONENT_LABELS, compute_component_ranks,
)


LABELS = COMPONENT_LABELS


def prepare_ranks(component_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the same nine components, models and tasks as the central heatmap."""
    long, means = compute_component_ranks(component_values)
    if means.shape[0] < 3:
        raise ValueError("A concordância exige pelo menos três modelos.")
    tasks = long.pivot(index=["dataset", "model"], columns="component", values="task_rank")
    return tasks[list(COMPONENTS)].reset_index(), means


def correlation_matrix(ranks: pd.DataFrame, method: str) -> pd.DataFrame:
    """Preserve ties; constant rankings have undefined correlations, even on the diagonal."""
    if method not in {"spearman", "kendall"}:
        raise ValueError(f"Método desconhecido: {method}")
    values = ranks.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Ranks não finitos; não é permitido excluir modelos por par.")
    result = pd.DataFrame(np.nan, index=ranks.columns, columns=ranks.columns)
    for i, first in enumerate(ranks.columns):
        if ranks[first].nunique() <= 1:
            continue
        result.loc[first, first] = 1.0
        for second in ranks.columns[i + 1:]:
            if ranks[second].nunique() <= 1:
                continue
            x, y = ranks[first], ranks[second]
            value = spearmanr(x, y).statistic if method == "spearman" else kendalltau(x, y, variant="b").statistic
            result.loc[first, second] = result.loc[second, first] = value
    return result


def compute_concordance(
    task_ranks: pd.DataFrame, means: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Summarize task-wise correlations; mean-rank correlations are supplementary."""
    metrics = list(COMPONENTS)
    matrices = {method: correlation_matrix(means[metrics], method) for method in ("spearman", "kendall")}
    rows = []
    for dataset, task in task_ranks.groupby("dataset", sort=True):
        task_matrices = {method: correlation_matrix(task[metrics], method) for method in matrices}
        for first, second in combinations(metrics, 2):
            rows.append({
                "dataset": dataset, "component_a": first, "component_b": second,
                "n_models": len(task),
                **{method: matrix.loc[first, second] for method, matrix in task_matrices.items()},
            })
    by_task = pd.DataFrame(rows)
    pairs = []
    for first, second in combinations(metrics, 2):
        group = by_task.loc[(by_task.component_a == first) & (by_task.component_b == second)]
        row = {"component_a": first, "component_b": second, "n_models": len(means), "n_tasks": len(group)}
        for method, matrix in matrices.items():
            values = group[method].dropna()
            row[f"{method}_mean_ranks"] = matrix.loc[first, second]
            row[f"{method}_valid_tasks"] = len(values)
            for statistic in ("mean", "std", "median", "min", "max"):
                row[f"{method}_task_{statistic}"] = getattr(values, statistic)()
        pairs.append(row)
    return matrices, by_task, pd.DataFrame(pairs)


def plot_concordance(
    matrix: pd.DataFrame, output_path: Path, colorbar_label: str,
    dispersion: pd.DataFrame | None = None,
) -> None:
    """Show the strict lower triangle with a fixed signed scale and larger labels."""
    values = matrix.to_numpy(dtype=float)
    hidden = np.triu(np.ones(values.shape, dtype=bool)) | ~np.isfinite(values)
    # The first row and last column have no lower-triangle cells. Trim them
    # while preserving the original component identities on both axes.
    displayed = np.ma.array(values, mask=hidden)[1:, :-1]
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(10, 8.5))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        im = ax.imshow(displayed, cmap=cmap, vmin=-1, vmax=1)
        labels = [LABELS[name] for name in matrix.columns]
        ax.set_xticks(np.arange(len(labels) - 1), labels=labels[:-1], rotation=40, ha="right")
        ax.set_yticks(np.arange(len(labels) - 1), labels=labels[1:])
        ax.tick_params(length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for index, label in enumerate(ax.get_xticklabels()):
            if matrix.columns[index] == "item":
                label.set_fontweight("bold")
        for row in range(1, len(labels)):
            for column in range(row):
                value = values[row, column]
                annotation = f"{value:.2f}" if np.isfinite(value) else "—"
                if dispersion is not None:
                    std = dispersion.iloc[row, column]
                    annotation += f"\n±{std:.2f}" if np.isfinite(std) else "\n±—"
                ax.text(column, row - 1, annotation,
                        ha="center", va="center", fontsize=11 if dispersion is not None else 12,
                        color="white" if np.isfinite(value) and abs(value) >= 0.65 else "black")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025).set_label(colorbar_label)
        fig.tight_layout()
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def write_report(pairs: pd.DataFrame, means: pd.DataFrame, output: Path, threshold: float) -> None:
    """Report descriptive agreement; a correlation threshold is not an independence test."""
    stable = pairs.loc[
        (pairs.spearman_task_min >= threshold)
        & (pairs.spearman_valid_tasks == pairs.n_tasks)
    ].copy()
    average_only = pairs.loc[
        (pairs.spearman_mean_ranks >= threshold)
        & ~pairs.index.isin(stable.index)
    ].copy()
    # Keep the first metric as representative only when its agreement with the
    # removed metric is high in every observed task.
    redundant = {frozenset((row.component_a, row.component_b)) for row in stable.itertuples()}
    representatives, removed = [], []
    for metric in COMPONENTS:
        match = next((other for other in representatives if frozenset((metric, other)) in redundant), None)
        if match is None:
            representatives.append(metric)
        else:
            removed.append(f"`{metric}` → `{match}`")

    lines = [
        f"# {FRAMEWORK_NAME}: component concordance", "",
        f"{len(means)} models; {int(means.n_tasks.iloc[0])} tasks; {len(COMPONENTS)} components.", "",
        "Ranks are computed within each task/metric, with average ranks for exact ties. "
        "For RQ1, the primary analysis computes Spearman across models separately in each task, "
        "then averages the task coefficients with equal weight. Heatmap cells show mean ± sample "
        "standard deviation (ddof=1) across tasks; this is descriptive dispersion, not a standard "
        "error or confidence interval. The pair table also reports median, minimum and maximum.", "",
        "Figures display only the lower triangle, excluding the diagonal and duplicate pairs. "
        "CSV matrices retain both triangles and the diagonal.", "",
        "The supplementary matrix correlates the mean task ranks across models and describes "
        "their stable ordering across tasks. Averaging ranks can smooth task-specific differences; "
        "an increase in correlation is not guaranteed. Kendall tau-b supplies a sensitivity check.", "",
        "Undefined correlations from constant rankings remain missing and valid-task counts are exported. "
        "No p-values or claims of statistical independence are made: models are a fixed benchmark set, "
        "and H3/H6 tasks share histories and test months.", "",
        "## Item versus other components", "",
        "| Component | Mean task Spearman | Task SD | Task range | Spearman of mean ranks (supplementary) | Mean task Kendall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairs.loc[pairs.component_a == "item"].itertuples():
        lines.append(f"| {row.component_b} | {row.spearman_task_mean:.3f} | {row.spearman_task_std:.3f} | "
                     f"{row.spearman_task_min:.3f}–{row.spearman_task_max:.3f} | {row.spearman_mean_ranks:.3f} | "
                     f"{row.kendall_task_mean:.3f} |")
    for title, frame in (("Consistently similar rankings", stable), ("High agreement in mean ranks only", average_only)):
        lines.extend(["", f"## {title}", "", f"Descriptive threshold: Spearman ≥ {threshold:.2f}.", "",
                      "| Pair | Mean task Spearman | Task SD | Task range | Spearman of mean ranks (supplementary) |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for row in frame.itertuples():
            lines.append(f"| {row.component_a} × {row.component_b} | {row.spearman_task_mean:.3f} | "
                         f"{row.spearman_task_std:.3f} | {row.spearman_task_min:.3f}–{row.spearman_task_max:.3f} | "
                         f"{row.spearman_mean_ranks:.3f} |")
    lines.extend([
        "", "## Conservative reduced view", "",
        "Retain: " + ", ".join(f"`{name}`" for name in representatives) + ".", "",
        "Representative substitutions: " + (", ".join(removed) if removed else "none") + ".", "",
        "This deterministic reduced view starts with item, then follows the manuscript component order. "
        "A component is removed only when agreement with a retained representative meets the threshold "
        "in every task. It is a descriptive screen for ranking redundancy, "
        "not proof that the retained metrics are independent. Sparse is excluded from the framework "
        "analysis because its rankings closely matched item in the preceding analysis.", "",
        "`item` is MAE, matching the traditional item-level accuracy figure. `scaled` is MASE "
        "with the mean absolute lag-one change from the task's training history as its scale; "
        "constant-history SKUs are excluded. Cap uses twice each SKU's training maximum. "
        "Zero averages signed forecast / training mean over zero-demand test observations, "
        "excluding no-demand training histories; saved forecasts are not clipped. "
        "Coverage and exclusions are reported in `../components/component_coverage_report.md`. "
        "Weighted, pointwise Relative, Robust and Sparse are excluded.", "",
        "Both Scaled and Zero reward zero forecasts. Asym (pinball tau=0.7) penalizes underprediction; "
        "Sum penalizes underestimating the total. Delta compares forecast-window differences "
        "without anchoring to the last training value.", "",
        "## Model-specific differences (supplementary mean ranks)", "",
        "`component_concordance_model_rank_differences.csv` contains each mean component rank "
        "minus the model's mean item rank. Positive differences indicate worse relative standing "
        "under the component. No averaging across components is performed.", "",
    ])
    if "chronos-2" in set(means.model):
        chronos = means.loc[means.model == "chronos-2"].iloc[0]
        lines.extend([
            f"Chronos-2 has mean item rank {chronos['item']:.2f}, compared with "
            f"{chronos['sum']:.2f} for aggregate-volume error, {chronos.delta:.2f} for "
            f"temporal-change error, {chronos.scaled:.2f} for Scaled, and {chronos.zero:.2f} for "
            "zero-demand forecasts. These differences "
            "illustrate changes in a model's standing under specific objectives.", "",
        ])
    lines.extend([
        "Methods: [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), "
        "[SciPy Kendall tau-b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).", "",
    ])
    (output / "component_concordance_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    task_ranks: pd.DataFrame, means: pd.DataFrame, matrices: dict[str, pd.DataFrame],
    by_task: pd.DataFrame, pairs: pd.DataFrame, output_dir: str, threshold: float,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_ranks.to_csv(output / "component_concordance_task_ranks.csv", index=False)
    means.to_csv(output / "component_concordance_mean_ranks.csv", index=False)
    by_task.to_csv(output / "component_concordance_by_task.csv", index=False)
    pairs.to_csv(output / "component_concordance_pairs.csv", index=False)
    differences = means[["model", "pretty_model", "family", "item"]].copy()
    for component in COMPONENTS:
        if component != "item":
            differences[f"{component}_minus_item"] = means[component] - means["item"]
    differences.to_csv(output / "component_concordance_model_rank_differences.csv", index=False)
    for method, matrix in matrices.items():
        matrix.to_csv(output / f"component_concordance_{method}_mean_ranks.csv", index_label="component")
        label = "Spearman ρ of mean task ranks" if method == "spearman" else "Kendall τ-b of mean task ranks"
        plot_concordance(matrix, output / f"component_concordance_{method}_mean_ranks", label)
    metrics = list(COMPONENTS)
    task_mean = pd.DataFrame(np.nan, index=metrics, columns=metrics)
    task_std = pd.DataFrame(np.nan, index=metrics, columns=metrics)
    valid_counts = pd.DataFrame(0, index=metrics, columns=metrics)
    for metric in metrics:
        count = sum(task[metric].nunique() > 1 for _, task in task_ranks.groupby("dataset"))
        task_mean.loc[metric, metric] = 1 if count else np.nan
        task_std.loc[metric, metric] = 0 if count > 1 else np.nan
        valid_counts.loc[metric, metric] = count
    for row in pairs.itertuples():
        task_mean.loc[row.component_a, row.component_b] = task_mean.loc[row.component_b, row.component_a] = row.spearman_task_mean
        task_std.loc[row.component_a, row.component_b] = task_std.loc[row.component_b, row.component_a] = row.spearman_task_std
        valid_counts.loc[row.component_a, row.component_b] = valid_counts.loc[row.component_b, row.component_a] = row.spearman_valid_tasks
    task_mean.to_csv(output / "component_concordance_spearman_task_mean.csv", index_label="component")
    task_std.to_csv(output / "component_concordance_spearman_task_std.csv", index_label="component")
    task_mean.to_csv(output / "component_concordance_spearman.csv", index_label="component")
    valid_counts.to_csv(output / "component_concordance_spearman_valid_tasks.csv", index_label="component")
    plot_concordance(task_mean, output / "component_concordance_spearman", "Within-task Spearman ρ (cells: mean ± SD)", dispersion=task_std)
    write_report(pairs, means, output, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/analysis/framework/components/component_values.csv")
    parser.add_argument("--output-dir", default="data/analysis/framework/concordance")
    parser.add_argument("--similarity-threshold", type=float, default=0.9,
                        help="Limiar descritivo para redundância de rankings; não testa independência.")
    args = parser.parse_args()
    if not 0 < args.similarity_threshold <= 1:
        parser.error("--similarity-threshold deve estar em (0, 1].")
    tasks, means = prepare_ranks(pd.read_csv(args.input))
    matrices, by_task, pairs = compute_concordance(tasks, means)
    save_outputs(tasks, means, matrices, by_task, pairs, args.output_dir, args.similarity_threshold)
    print(f"✓ {len(means)} modelos; {means.n_tasks.iloc[0]} tarefas; {len(COMPONENTS)} componentes.")
    print(pairs.loc[pairs.component_a == "item", [
        "component_b", "spearman_task_mean", "spearman_task_std", "spearman_task_min", "spearman_task_max", "spearman_mean_ranks",
    ]].round(3).to_string(index=False))
    print(f"Figuras, CSVs e relatório: {args.output_dir}")


if __name__ == "__main__":
    main()
