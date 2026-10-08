"""Compare models by mean task rank for each Inventory-Aware Evaluation Framework component."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from src.analysis.model_style import (
    FAMILY_COLORS, FAMILY_LABELS, FAMILY_ORDER, model_family, pretty_model_name,
)
from src.analysis.inventory_framework import COMPONENTS, COMPONENT_LABELS, FRAMEWORK_NAME


def compute_component_ranks(
    values: pd.DataFrame,
    components: list[str] | tuple[str, ...] = COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank raw component values within task/component, then average only over tasks."""
    components = list(components)
    if not components or len(set(components)) != len(components):
        raise ValueError("Selecione componentes distintos e não vazios.")
    if unsupported := set(components) - set(COMPONENTS):
        raise ValueError(f"Componentes fora da comparação principal: {sorted(unsupported)}")
    missing = {"dataset", "model", *components} - set(values.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if values.empty or values[["dataset", "model"]].isna().any().any():
        raise ValueError("Resultados vazios ou com identificadores ausentes.")
    if values.duplicated(["dataset", "model"]).any():
        raise ValueError("Cada par tarefa/modelo deve aparecer exatamente uma vez.")
    values = values.copy()
    values[list(components)] = values[list(components)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values[list(components)].to_numpy()).all():
        raise ValueError("Componentes não finitos: complete os resultados antes de ranquear.")
    # Zero follows the supplied signed-forecast formula; saved predictions can be negative.
    nonnegative = [name for name in components if name != "zero"]
    if (values[nonnegative] < 0).any().any():
        raise ValueError("Os componentes, exceto Zero assinado, devem ter valores não negativos.")

    models = sorted(values.model.unique())
    tasks = sorted(values.dataset.unique())
    expected = pd.MultiIndex.from_product([tasks, models], names=["dataset", "model"])
    observed = pd.MultiIndex.from_frame(values[["dataset", "model"]])
    if len(absent := expected.difference(observed)):
        raise ValueError(f"Comparação desbalanceada: faltam {len(absent)} pares tarefa/modelo; exemplos: {list(absent[:5])}")
    # If supplied, coverage metadata must agree across competitors for each task.
    metadata = [name for name in (
        "matched_rows", "n_series", "horizon", "n_train", "train_start", "train_end",
        "mase_eligible_skus", "mase_excluded_constant_skus", "mase_observations",
        "zero_eligible_skus", "zero_excluded_no_demand_skus", "zero_scored_skus",
        "zero_observations", "zero_excluded_observations",
    ) if name in values]
    for name in metadata:
        if values[name].isna().any() or (values.groupby("dataset")[name].nunique() != 1).any():
            raise ValueError(f"Cobertura diferente entre modelos: {name}.")
    if {"matched_rows", "n_series", "horizon"}.issubset(metadata) and (values.matched_rows != values.n_series * values.horizon).any():
        raise ValueError("Cobertura incompleta de SKU/mês nos resultados.")

    long = values.melt(id_vars=["dataset", "model"], value_vars=components, var_name="component", value_name="value")
    long["task_rank"] = long.groupby(["dataset", "component"])["value"].rank(method="average", ascending=True)
    long["n_models"] = len(models)
    means = long.groupby(["model", "component"])["task_rank"].mean().unstack("component").reindex(columns=components).reset_index()
    means["pretty_model"] = means.model.map(pretty_model_name)
    means["family"] = means.model.map(model_family)
    means["n_tasks"] = len(tasks)
    # Alphabetical ordering inside families avoids an implicit overall score.
    means["family_order"] = means.family.map({family: index for index, family in enumerate(FAMILY_ORDER)})
    means = means.sort_values(["family_order", "pretty_model", "model"]).drop(columns="family_order")
    means = means[["model", "pretty_model", "family", "n_tasks", *components]].reset_index(drop=True)
    return long.sort_values(["dataset", "component", "task_rank", "model"]).reset_index(drop=True), means


def plot_heatmap(means: pd.DataFrame, components: list[str], output_path: Path) -> None:
    values = means[components].to_numpy(dtype=float)
    n_models = len(means)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(max(8, 0.83 * len(components) + 3.2), max(4, 0.35 * n_models + 1.6)))
        cmap = plt.get_cmap("YlOrRd")
        norm = plt.Normalize(vmin=1, vmax=max(2, n_models))
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(components)), labels=[COMPONENT_LABELS[name] for name in components], rotation=30, ha="right")
        ax.set_yticks(np.arange(n_models), labels=means.pretty_model)
        ax.set_xlabel(f"{FRAMEWORK_NAME} component")
        ax.set_ylabel("Model")
        ax.set_xlim(-0.72, len(components) - 0.5)
        for index, family in enumerate(means.family):
            color = FAMILY_COLORS[family]
            ax.get_yticklabels()[index].set_color(color)
            ax.add_patch(Rectangle((-0.70, index - 0.5), 0.14, 1, facecolor=color, edgecolor="none", clip_on=False))
            if index and family != means.family.iloc[index - 1]:
                ax.axhline(index - 0.5, color="white", linewidth=2.4)
        for row in range(n_models):
            for column in range(len(components)):
                value = values[row, column]
                red, green, blue, _ = cmap(norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8.5, color="black" if luminance > 0.5 else "white")
        colorbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        colorbar.set_label("Mean task rank (1 = best)")
        present = [family for family in FAMILY_ORDER if family in set(means.family)]
        fig.legend(handles=[Patch(facecolor=FAMILY_COLORS[family], label=FAMILY_LABELS[family]) for family in present], loc="lower center", ncol=len(present), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def save_outputs(task_ranks: pd.DataFrame, means: pd.DataFrame, components: list[str], output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_ranks.to_csv(output / "inventory_component_task_ranks.csv", index=False)
    means.to_csv(output / "inventory_component_mean_ranks.csv", index=False)
    plot_heatmap(means, components, output / "inventory_component_rank_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/analysis/framework/components/component_values.csv", help="CSV de componentes gerado por evaluate_inventory_framework.")
    parser.add_argument("--output-dir", default="data/analysis/framework/ranks")
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--models", nargs="+", default=None, help="Modelos incluídos antes do cálculo dos ranks.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Tarefas incluídas antes do cálculo dos ranks.")
    args = parser.parse_args()
    values = pd.read_csv(args.input)
    for column, requested in (("model", args.models), ("dataset", args.datasets)):
        if requested:
            if absent := set(requested) - set(values[column]):
                raise ValueError(f"{column}: identificadores não encontrados: {sorted(absent)}")
            values = values.loc[values[column].isin(requested)]
    task_ranks, means = compute_component_ranks(values, args.components)
    save_outputs(task_ranks, means, args.components, args.output_dir)
    print(f"✓ {len(means)} modelos × {len(args.components)} componentes; {means.n_tasks.iloc[0]} tarefas por célula.")
    for component in args.components:
        best = means.loc[means[component] == means[component].min()]
        print(f"  {component}: {', '.join(best.pretty_model)} (rank médio {best[component].iloc[0]:.2f})")
    print(f"Figuras PDF e CSVs: {args.output_dir}")


if __name__ == "__main__":
    main()
