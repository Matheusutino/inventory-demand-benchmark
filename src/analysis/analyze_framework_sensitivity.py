"""Recalculate one-factor framework sensitivity and leave-one-panel-out ranks from saved forecasts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from src.evaluation.forecast_evaluation import (
    DEFAULT_EVALUATION_CONFIG, aligned_tensors, compute_components, history_statistics,
    list_prediction_files, load_history_by_dataset, load_truth_by_dataset, model_name_from_path,
)
from src.analysis.inventory_framework import COMPONENTS, COMPONENT_LABELS
from src.analysis.model_style import (
    FAMILY_COLORS, FAMILY_LABELS, FAMILY_ORDER, model_family, model_marker, pretty_model_name,
)
from src.analysis.plot_inventory_component_rank_heatmap import compute_component_ranks
from src.analysis.plot_item_sum_mae import panel_name
from src.analysis.plot_framework_sensitivity_overview import save_overview_figures


ORIGINAL_SETTINGS = {
    "asym": ("tau", "0.7"), "cap": ("kappa", "2"), "scaled": ("scale", "lag1"),
    "zero": ("normalization", "mean"), "sum": ("error", "squared"),
    "delta": ("error", "squared"),
}
SETTINGS = {
    "asym": ("0.5", "0.6", "0.7", "0.8", "0.9"),
    "cap": ("1.25", "1.5", "2", "3", "5"),
    "scaled": ("lag1", "lag12", "mean"), "zero": ("mean", "max", "none"),
    "sum": ("squared", "absolute"), "delta": ("squared", "absolute"),
}
SPECIFIED_MODELS = ("chronos-2", "timesfm-2.5-200m-pytorch", "tabpfn-ts-local", "TiRex", "moirai2-small")


@dataclass(frozen=True)
class Variation:
    component: str
    variation_id: str
    parameter: str
    setting: str
    is_original: bool = False
    excluded_panel: str = ""

    def metadata(self) -> dict:
        """Return CSV identifiers shared by values, ranks and comparisons."""
        label = f"{COMPONENT_LABELS[self.component]}: {self.parameter}={self.setting}"
        if self.excluded_panel:
            label = f"{COMPONENT_LABELS[self.component]}: sem {panel_name(self.excluded_panel)}"
        elif self.is_original:
            label += " (original)"
        return {"component": self.component, "variation_id": self.variation_id,
                "variation": label, "parameter": self.parameter, "setting": self.setting,
                "is_original": self.is_original, "excluded_panel": self.excluded_panel,
                "kind": "leave_panel_out" if self.excluded_panel else "parameter"}


def primary_variations() -> list[Variation]:
    """Include each component's original once and all requested one-factor alternatives."""
    result = []
    for component in COMPONENTS:
        parameter, original = ORIGINAL_SETTINGS.get(component, ("configuration", "original"))
        result.append(Variation(component, "original", parameter, original, is_original=True))
        for setting in SETTINGS.get(component, ()):
            if setting != original:
                result.append(Variation(component, f"{parameter}_{setting}", parameter, setting))
    return result


def training_scales(history: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build aligned training denominators; seasonal naive requires at least 13 months."""
    if history.shape[1] <= 12:
        raise ValueError("MASE sazonal lag 12 requer pelo menos 13 meses de treino.")
    return {"lag1": history.diff(dim=1).abs().mean(dim=1, keepdim=True),
            "lag12": (history[:, 12:] - history[:, :-12]).abs().mean(dim=1, keepdim=True),
            "mean": history.mean(dim=1, keepdim=True), "max": history.amax(dim=1, keepdim=True)}


def variation_value(pred: torch.Tensor, target: torch.Tensor, scales: dict,
                    variation: Variation, originals: dict[str, float]) -> float:
    """Change only the selected formula, retaining signed forecasts and original zero eligibility."""
    if variation.is_original:
        return originals[variation.component]
    component, setting = variation.component, variation.setting
    if component == "asym":
        error, tau = target - pred, float(setting)
        value = torch.maximum(tau * error, (tau - 1) * error).mean()
    elif component == "cap":
        value = (pred - float(setting) * scales["max"]).clamp(min=0).square().mean()
    elif component == "scaled":
        scale = scales[setting].expand_as(pred)
        eligible = scale > 0
        value = ((pred - target).abs()[eligible] / scale[eligible]).mean()
    elif component == "zero":
        eligible = (target == 0) & (scales["mean"] > 0)
        values = pred[eligible]
        if setting != "none":
            values = values / scales[setting].expand_as(pred)[eligible]
        value = values.mean()
    elif component == "sum" and setting == "absolute":
        total = target.sum(-1)
        denominator = total.abs().clamp(min=DEFAULT_EVALUATION_CONFIG["rel_eps"])
        value = ((pred.sum(-1) - total) / denominator).abs().mean()
    elif component == "delta" and setting == "absolute":
        value = (pred.diff(dim=1) - target.diff(dim=1)).abs().mean()
    else:
        raise ValueError(f"Variação não implementada: {variation}")
    return float(value)


def variation_coverage(target: torch.Tensor, scales: dict, variation: Variation) -> dict:
    """Count eligible SKUs and actual scored terms once per task and variation."""
    n_skus, horizon = target.shape[-1], target.shape[1]
    mask = torch.ones_like(target, dtype=torch.bool)
    eligible_skus = n_skus
    if variation.component == "scaled":
        mask = (scales[variation.setting] > 0).expand_as(target)
        eligible_skus = int((scales[variation.setting] > 0).sum())
    elif variation.component == "zero":
        mask = (target == 0) & (scales["mean"] > 0)
        eligible_skus = int((scales["mean"] > 0).sum())
    observations = int(mask.sum())
    if variation.component in ("sum", "share"):
        observations = horizon if variation.component == "sum" else int((target.sum(-1) > DEFAULT_EVALUATION_CONFIG["share_eps"]).sum())
    elif variation.component == "delta":
        observations = (horizon - 1) * n_skus
    return {"eligible_skus": eligible_skus, "excluded_skus": n_skus - eligible_skus,
            "scored_skus": int(mask.any(dim=1).sum()), "n_observations": observations}


def cap_violations(pred: torch.Tensor, maximum: torch.Tensor, kappa: float) -> dict:
    """Count strictly above-limit SKU/month forecasts, retaining zero-history limits of zero."""
    mask = pred > kappa * maximum
    count = int(mask.sum())
    return {"kappa": kappa, "n_violations": count, "n_predictions": pred.numel(),
            "violation_share": count / pred.numel(), "n_violating_skus": int(mask.any(dim=1).sum()),
            "zero_history_violations": int((mask & (maximum == 0)).sum())}


def evaluate_sensitivity(truths: dict, histories: dict, files: list[Path]) -> dict[str, pd.DataFrame]:
    """Align each saved model once per task and evaluate every original and one-factor variant."""
    variations = primary_variations()
    prepared, scale_rows = {}, []
    for dataset, truth in truths.items():
        history, stats, metadata = history_statistics(histories[dataset], truth)
        scales = training_scales(history)
        prepared[dataset] = (history, scales, metadata)
        for name, scale in scales.items():
            stats[f"scale_{name}"] = scale.reshape(-1).numpy()
        stats.insert(0, "dataset", dataset)
        scale_rows.append(stats)
    rows, coverage, violations = [], [], []
    for index, path in enumerate(files):
        pred = pd.read_parquet(path)
        if missing := {"dataset", "id", "timestamp", "q50"} - set(pred):
            raise ValueError(f"{path.name}: colunas ausentes: {sorted(missing)}")
        if pred[["dataset", "id", "timestamp"]].isna().any().any():
            raise ValueError(f"{path.name}: identificadores ausentes.")
        if set(pred.dataset) != set(truths):
            raise ValueError(f"{path.name}: tarefas diferentes das esperadas.")
        pred["id"] = pred.id.astype(str)
        pred["timestamp"] = pd.to_datetime(pred.timestamp)
        for dataset, truth in truths.items():
            yh, y, matched, n_skus, horizon = aligned_tensors(truth, pred.loc[pred.dataset == dataset])
            history, scales, metadata = prepared[dataset]
            originals = compute_components(yh, y, DEFAULT_EVALUATION_CONFIG, history)
            base = {"dataset": dataset, "panel": dataset.rsplit("_h", 1)[0],
                    "panel_label": panel_name(dataset), "horizon": horizon,
                    "matched_rows": matched, "n_series": n_skus, "n_train": metadata["n_train"]}
            for variation in variations:
                rows.append({**base, "model": model_name_from_path(path), **variation.metadata(),
                             "value": variation_value(yh, y, scales, variation, originals)})
                if index == 0:
                    coverage.append({**base, **variation.metadata(), **variation_coverage(y, scales, variation)})
                if variation.component == "cap":
                    violations.append({**base, "model": model_name_from_path(path),
                                       **cap_violations(yh, scales["max"], float(variation.setting))})
    if not rows:
        raise ValueError("Nenhuma previsão avaliada.")
    return {"component_values": pd.DataFrame(rows), "component_coverage": pd.DataFrame(coverage),
            "training_scales": pd.concat(scale_rows, ignore_index=True),
            "cap_violations_by_task": pd.DataFrame(violations)}


def add_panel_omissions(values: pd.DataFrame) -> pd.DataFrame:
    """Reevaluate the original components on each eight-task subset, with both horizons omitted."""
    originals = values.loc[values.is_original]
    pieces = [values]
    for panel in sorted(originals.panel.unique()):
        subset = originals.loc[originals.panel != panel].copy()
        if subset.empty:
            raise ValueError("A robustez requer pelo menos dois painéis.")
        for component in COMPONENTS:
            variation = Variation(component, f"leave_out_{panel}", "excluded_panel", panel, excluded_panel=panel)
            mask = subset.component == component
            for key, value in variation.metadata().items():
                subset.loc[mask, key] = value
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True)


def rank_variations(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the RQ2 ranking helper within each task, then mean task ranks with equal task weights."""
    tasks, means = [], []
    originals = values.loc[values.is_original]
    for (component, _), group in values.groupby(["component", "variation_id"], sort=False):
        reference = originals.loc[originals.component == component]
        if set(group.model) != set(reference.model):
            raise ValueError("Conjunto de modelos diferente entre variação e original.")
        excluded = group.excluded_panel.iloc[0]
        expected_tasks = set(reference.loc[reference.panel != excluded, "dataset"])
        if set(group.dataset) != expected_tasks:
            raise ValueError("Conjunto de tarefas diferente do esperado para a variação.")
        fields = ["variation_id", "variation", "parameter", "setting", "is_original", "excluded_panel", "kind"]
        if len(group[fields].drop_duplicates()) != 1:
            raise ValueError("Identificadores de variação inconsistentes.")
        metadata = group[fields].iloc[0].to_dict()
        raw = group[["dataset", "model", "matched_rows", "n_series", "horizon", "n_train", "value"]].rename(columns={"value": component})
        task, mean = compute_component_ranks(raw, [component])
        mean = mean.rename(columns={component: "mean_rank"}).assign(component=component)
        tasks.append(task.assign(**metadata))
        means.append(mean.assign(**metadata))
    return pd.concat(tasks, ignore_index=True), pd.concat(means, ignore_index=True)


def ranking_members(ranks: pd.DataFrame, top_n: int = 5) -> dict:
    """Resolve exactly-five membership alphabetically, exposing all ties at the cutoff and leader."""
    ordered = ranks.sort_values(["mean_rank", "pretty_model", "model"])
    count = min(top_n, len(ordered))
    cutoff = ordered.mean_rank.iloc[count - 1]
    leaders = set(ordered.loc[ordered.mean_rank == ordered.mean_rank.min(), "model"])
    inclusive = set(ordered.loc[ordered.mean_rank <= cutoff, "model"])
    return {"top": set(ordered.head(count).model), "inclusive": inclusive,
            "boundary_tie": len(inclusive) > count, "leaders": leaders}


def compare_variations(means: pd.DataFrame) -> pd.DataFrame:
    """Compare each mean-rank vector with its component's original full ten-task vector."""
    rows = []
    for component, group in means.groupby("component", sort=False):
        original = group.loc[group.is_original]
        reference = original.set_index("model").mean_rank.sort_index()
        baseline = ranking_members(original)
        for _, variant in group.groupby("variation_id", sort=False):
            changed = variant.set_index("model").mean_rank.reindex(reference.index)
            if changed.isna().any():
                raise ValueError("Modelo ausente na comparação com o original.")
            rho = float(spearmanr(reference, changed).statistic) if reference.nunique() > 1 and changed.nunique() > 1 else np.nan
            selected = ranking_members(variant)
            fields = ["component", "variation_id", "variation", "parameter", "setting", "is_original", "excluded_panel", "kind", "n_tasks"]
            rows.append({**variant[fields].iloc[0].to_dict(), "spearman": rho,
                         "top5_overlap": len(baseline["top"] & selected["top"]),
                         "top5_size": len(baseline["top"]), "original_top5_boundary_tie": baseline["boundary_tie"],
                         "variant_top5_boundary_tie": selected["boundary_tie"],
                         "top5_tie_inclusive_overlap": len(baseline["inclusive"] & selected["inclusive"]),
                         "original_top5_tie_inclusive_size": len(baseline["inclusive"]),
                         "variant_top5_tie_inclusive_size": len(selected["inclusive"]),
                         "original_top5_models": " | ".join(sorted(baseline["top"])),
                         "top5_models": " | ".join(sorted(selected["top"])),
                         "original_leaders": " | ".join(sorted(baseline["leaders"])),
                         "leaders": " | ".join(sorted(selected["leaders"])),
                         "leader_names": " | ".join(pretty_model_name(m) for m in sorted(selected["leaders"])),
                         "n_original_leaders": len(baseline["leaders"]), "n_leaders": len(selected["leaders"]),
                         "leader_changed": baseline["leaders"] != selected["leaders"],
                         "leader_mean_rank": variant.mean_rank.min()})
    return pd.DataFrame(rows)


def select_highlights(means: pd.DataFrame) -> pd.DataFrame:
    """Highlight the requested foundation models and the original best of every other family per axis."""
    rows = []
    for component in ("asym", "cap"):
        original = means.loc[(means.component == component) & means.is_original]
        if absent := set(SPECIFIED_MODELS) - set(original.model):
            raise ValueError(f"Modelos solicitados ausentes da figura: {sorted(absent)}")
        for model in SPECIFIED_MODELS:
            rows.append({"component": component, "model": model, "reason": "specified", "family_best_ties": ""})
        for family in FAMILY_ORDER:
            if family == "foundation":
                continue
            candidates = original.loc[original.family == family].sort_values(["mean_rank", "pretty_model", "model"])
            if candidates.empty:
                continue
            best = candidates.loc[candidates.mean_rank == candidates.mean_rank.min()]
            rows.append({"component": component, "model": best.model.iloc[0], "reason": "best_original_in_family",
                         "family_best_ties": " | ".join(best.model)})
    result = pd.DataFrame(rows)
    result["pretty_model"] = result.model.map(pretty_model_name)
    result["family"] = result.model.map(model_family)
    return result


def plot_parameter_curves(means: pd.DataFrame, highlights: pd.DataFrame, output: Path) -> None:
    """Plot all twenty model curves; highlight specified models and component-specific family leaders."""
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(19, 8), sharey=True)
        for ax, component, parameter, symbol, original_value in zip(
                axes, ("asym", "cap"), ("tau", "kappa"), ("τ", "κ"), (.7, 2.)):
            subset = means.loc[(means.component == component) & (means.parameter == parameter)].copy()
            subset["x"] = subset.setting.astype(float)
            selected = highlights.loc[highlights.component == component]
            selected_models = set(selected.model)
            handles = {}
            models = sorted(subset.model.unique(), key=lambda m: (m in selected_models, pretty_model_name(m)))
            for model in models:
                curve = subset.loc[subset.model == model].sort_values("x")
                highlight = model in selected_models
                line, = ax.plot(curve.x, curve.mean_rank, color=FAMILY_COLORS[model_family(model)],
                                alpha=1 if highlight else .22, linewidth=2 if highlight else .8,
                                marker=model_marker(model) if highlight else None, markersize=6,
                                markerfacecolor="white" if highlight else None,
                                zorder=3 if highlight else 1, label=pretty_model_name(model))
                handles[model] = line
            ax.axvline(original_value, color="#555555", linestyle="--", alpha=.6, linewidth=1)
            ax.set_xticks(sorted(subset.x.unique()))
            ax.set_xlabel(f"{COMPONENT_LABELS[component]} — {symbol}")
            ax.set_ylim(means.model.nunique() + .4, .6)
            ax.set_yticks([1, 5, 10, 15, 20])
            ax.set_ylabel("Mean task rank (1 = best)")
            ax.grid(alpha=.2)
            ax.legend(handles=[handles[m] for m in selected.model], loc="upper center",
                      bbox_to_anchor=(.5, -.16), ncol=3, fontsize=8, frameon=False,
                      columnspacing=1.2, handlelength=2)
        fig.legend(handles=[Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in FAMILY_ORDER],
                   loc="lower center", ncol=len(FAMILY_ORDER), frameon=False, fontsize=9,
                   bbox_to_anchor=(.5, .055))
        fig.text(.5, .025, "Dashed vertical lines: original settings. Highlighted curves: specified foundation models "
                 "+ best original model per other family, selected separately for each component.",
                 ha="center", fontsize=9)
        axes[1].text(.98, .96, "Coincident curves indicate tied ranks", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .14, 1, 1), w_pad=3)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def save_report(tables: dict[str, pd.DataFrame], output: Path) -> None:
    """Write the requested summary table plus definitions, tie handling, coverage and main findings."""
    summary = tables["sensitivity_summary"]
    lines = ["# Sensibilidade do Inventory-Aware Evaluation Framework", "",
        "Previsões salvas, sem retreinamento e sem truncar previsões negativas. Cada variação muda "
        "um componente por vez. Em cada tarefa, os 20 modelos recebem ranks crescentes "
        "(1 = melhor, empates exatos recebem a média); o resultado é a média dos ranks nas "
        "10 tarefas, com pesos iguais, usando o mesmo código do heatmap da RQ2. "
        "Não se calcula score global entre componentes.", "",
        "Os ranks são calculados após ler `component_values.csv`, reproduzindo o fluxo da RQ2 "
        "e seus empates na representação numérica exportada.", "",
        "A configuração original de cada componente aparece uma vez no CSV, com `variation_id=original`. "
        "As curvas incluem esses pontos (τ=0.7 e κ=2). Todas as comparações, inclusive as de "
        "robustez, usam como referência os ranks médios originais das dez tarefas.", "",
        "## Visão geral das figuras", "",
        "[Visão geral da sensibilidade](sensitivity_overview.pdf): à esquerda, as 14 alternativas "
        "paramétricas, com Spearman e retenção do top-5; à direita, os nove componentes × cinco "
        "painéis retirados, com Spearman e top-5 em cada célula. Losangos indicam mudança no conjunto "
        "de líderes. O asterisco sinaliza empate no corte do top-5.", "",
        "[Todos os modelos e cenários](sensitivity_all_model_ranks.pdf): cada célula mostra o rank "
        "médio do modelo; a cor representa a diferença em relação ao original do mesmo componente "
        "(azul = melhora, vermelho = piora). São os 1.360 ranks dos vinte modelos nos 68 cenários, "
        "separando configurações paramétricas e exclusões de painel. `rank_changes.csv` registra "
        "os valores originais e as diferenças sem arredondamento de apresentação.", "",
        "## Definições", "",
        "- Asym: mean(max(τ(y−ŷ), (τ−1)(y−ŷ))), τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}.",
        "- Cap: mean(max(ŷ−κ max(y_train), 0)²), κ ∈ {1.25, 1.5, 2, 3, 5}. "
        "Violação significa ŷ estritamente maior que o limite, por par SKU/mês; igualdade não viola. "
        "Históricos sem demanda mantêm limite zero para qualquer κ.",
        "- Scaled: mean(|ŷ−y|/q_i). Lag 1 usa mean(|y_train[t]−y_train[t−1]|); "
        "lag 12 usa mean(|y_train[t]−y_train[t−12]|), sobre os T−12 pares; "
        "a terceira versão usa a média histórica. Cada versão exclui os SKUs com seu próprio q_i=0. "
        "A versão com média histórica é erro absoluto escalado pela média, não o MASE convencional.",
        "- Zero: média das previsões nas observações de teste com y=0, divididas pela média histórica, "
        "pelo máximo histórico ou sem divisão. As três versões mantêm os mesmos SKUs elegíveis "
        "(média histórica positiva) e as mesmas observações, isolando a normalização. "
        "A fórmula permanece assinada: previsões negativas podem produzir valores negativos.",
        "- Sum: mean(((sum_i ŷ−sum_i y)/max(|sum_i y|, ε))²) versus "
        "mean(|(sum_i ŷ−sum_i y)/max(|sum_i y|, ε)|), preservando ε=1e−6 e a normalização original.",
        "- Delta: mean(((ŷ[t]−ŷ[t−1])−(y[t]−y[t−1]))²) versus a média do valor absoluto "
        "da mesma diferença, somente dentro da janela prevista, sem ancoragem no treino.",
        "- Robustez: retirar um painel de cada vez, incluindo H3 e H6 desse painel. Os nove componentes "
        "originais são avaliados nas oito tarefas restantes, com os mesmos vinte competidores. "
        "Retirar uma tarefa não altera seu histórico nem os valores das outras tarefas.", "",
        "Os denominadores e limites usam exclusivamente o treino de cada tarefa. "
        "`training_scales.csv` e `component_coverage.csv` permitem verificar escalas, exclusões e "
        "número de termos avaliados. Componentes não finitos ou cobertura incompleta impedem o ranking.", "",
        "## Comparações e empates", "",
        "Spearman correlaciona os dois vetores de ranks médios, alinhados por modelo. "
        "Vetores constantes têm correlação indefinida (NaN). O top-5 usa exatamente cinco modelos: "
        "empates no rank médio de corte são resolvidos pelo nome exibido em ordem alfabética, depois "
        "pelo identificador. Esse desempate só define a seleção do top-5; os ranks continuam empatados. "
        "Um asterisco indica empate no corte original ou da variação. O CSV também fornece "
        "sobreposição e tamanhos incluindo todos os empatados no corte. O líder inclui todos os "
        "modelos empatados na menor média; mudança de líder significa mudança nesse conjunto.", "",
        "Na figura, destacam-se Chronos-2, TimesFM 2.5, TabPFN-TS, TiRex, Moirai2-Small e o melhor "
        "modelo original de cada outra família, escolhido separadamente para Asym e Cap. "
        "Em empate dentro da família, destaca-se o primeiro em ordem alfabética; os co-líderes "
        "estão registrados em `highlighted_models.csv`.", "",
        "## Tabela-resumo", "",
        "| Variação | Spearman | Sobreposição top-5 | Líder(es) | Líder muda? |",
        "| --- | ---: | ---: | --- | --- |"]
    for r in summary.itertuples():
        star = "*" if r.original_top5_boundary_tie or r.variant_top5_boundary_tie else ""
        rho = f"{r.spearman:.4f}" if np.isfinite(r.spearman) else "Indefinido"
        leaders = r.leader_names.replace(" | ", ", ") if r.n_leaders <= 3 else f"Empate ({r.n_leaders} modelos)"
        lines.append(f"| {r.variation} | {rho} | {r.top5_overlap}/{r.top5_size}{star} | "
                     f"{leaders} | {'Sim' if r.leader_changed else 'Não'} |")
    lines.extend(["", "Nos empates com mais de três líderes, a tabela indica o tamanho do conjunto; "
                  "os nomes completos estão no CSV e na lista de empates abaixo.", ""])
    lines.extend(["", "## Diagnóstico por componente", "",
                  "| Componente | Menor Spearman: parâmetros | Menor Spearman: retirar painel | "
                  "Menor sobreposição top-5: parâmetros | Menor sobreposição top-5: retirar painel |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for component in COMPONENTS:
        params = summary.loc[(summary.component == component) & ~summary.is_original & (summary.kind == "parameter")]
        omitted = summary.loc[(summary.component == component) & (summary.kind == "leave_panel_out")]
        minimum = f"{params.spearman.min():.4f}" if not params.empty else "—"
        overlap = str(params.top5_overlap.min()) if not params.empty else "—"
        lines.append(f"| {COMPONENT_LABELS[component]} | {minimum} | {omitted.spearman.min():.4f} | "
                     f"{overlap} | {omitted.top5_overlap.min()} |")
    coverage = tables["component_coverage"]
    scaled = coverage.loc[(coverage.component == "scaled") & (coverage.kind == "parameter")]
    lines.extend(["", "## Exclusões do componente Scaled", "",
                  "Contagens somadas entre painéis dentro de cada horizonte, sem multiplicar pelos modelos.", "",
                  "| Escala | Horizonte | SKUs excluídos | Termos avaliados |",
                  "| --- | --- | ---: | ---: |"])
    for (setting, horizon), group in scaled.groupby(["setting", "horizon"], sort=True):
        lines.append(f"| {setting} | H{horizon} | {group.excluded_skus.sum()} | {group.n_observations.sum()} |")
    lines.extend(["", "## Violações do Cap", "",
        "`cap_violations_by_task.csv` reporta cada modelo × tarefa × κ. `cap_violations_by_model.csv` "
        "soma as dez tarefas e reporta contagem e fração das previsões que excedem o limite. "
        "As janelas H3/H6 são aninhadas: a soma conta instâncias de previsões por tarefa, "
        "não pares SKU/mês únicos no calendário. `zero_history_violations` separa os excessos "
        "em SKUs com máximo histórico zero, cujo limite permanece zero para todos os κ.", "",
        "A tabela a seguir soma os modelos: são avaliações das previsões de vinte competidores, "
        "não observações únicas de demanda.", "",
        "| κ | Previsões que violam | Previsões avaliadas | Fração | Violações em histórico zero |",
        "| ---: | ---: | ---: | ---: | ---: |"])
    for kappa, group in tables["cap_violations_by_model"].groupby("kappa", sort=True):
        count, total = group.n_violations.sum(), group.n_predictions.sum()
        lines.append(f"| {kappa:g} | {count} | {total} | {100*count/total:.2f}% | {group.zero_history_violations.sum()} |")
    original_cap = summary.loc[(summary.component == "cap") & summary.is_original].iloc[0]
    cap_counts = tables["cap_violations_by_model"]
    original_counts = cap_counts.loc[cap_counts.kappa == 2]
    violations = original_counts.n_violations.sum()
    share_zero = 100 * original_counts.zero_history_violations.sum() / violations if violations else np.nan
    lines.extend(["", f"No Cap original, {original_cap.n_leaders} modelos empatam na liderança; "
                  "portanto, o top-5 de cinco nomes depende do desempate alfabético. "
                  "A sobreposição incluindo todos os empatados está no CSV. "
                  f"Das {violations} violações originais, {share_zero:.1f}% pertencem a históricos "
                  "sem demanda: aumentar κ não altera o limite desses SKUs.", "",
                  "## Empates de liderança", ""])
    for leaders, group in summary.loc[summary.n_leaders > 1].groupby("leaders", sort=False):
        names = group.leader_names.iloc[0].replace(" | ", ", ")
        labels = "; ".join(group.variation)
        lines.append(f"- **{group.n_leaders.iloc[0]} co-líderes** ({labels}): {names}.")
    lines.extend(["", "H3/H6 não são amostras independentes; estes resultados descrevem sensibilidade "
                  "dos rankings, sem testes de significância ou intervalos de confiança.", ""])
    (output / "sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--output-dir", default="data/analysis/framework/sensitivity")
    args = parser.parse_args()
    torch.set_num_threads(1)
    truths = load_truth_by_dataset(args.data_dir)
    histories = load_history_by_dataset(args.data_dir)
    files = list_prediction_files(args.predictions_dir, None)
    if len(files) != 20 or len(truths) != 10:
        raise ValueError(f"São esperados 20 modelos e 10 tarefas; encontrados {len(files)} e {len(truths)}.")
    panel_horizons = {}
    for dataset, truth in truths.items():
        panel, suffix = dataset.rsplit("_h", 1)
        if suffix not in ("3", "6") or truth.timestamp.nunique() != int(suffix):
            raise ValueError(f"Horizonte inválido ou incompleto: {dataset}")
        panel_horizons.setdefault(panel, set()).add(int(suffix))
    if len(panel_horizons) != 5 or any(h != {3, 6} for h in panel_horizons.values()):
        raise ValueError("São esperados cinco painéis, cada um com H3 e H6.")
    tables = evaluate_sensitivity(truths, histories, files)
    tables["component_values"] = add_panel_omissions(tables["component_values"])
    tables["component_coverage"] = add_panel_omissions(tables["component_coverage"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables["component_values"].to_csv(output / "component_values.csv", index=False)
    # RQ2 ranks its exported CSV. Use the same roundtrip so exact ties and numerical
    # precision agree with the published baseline, including near-zero JSD values.
    rank_input = pd.read_csv(output / "component_values.csv").fillna({"excluded_panel": ""})
    task_ranks, means = rank_variations(rank_input)
    tables.update(task_ranks=task_ranks, mean_ranks=means, sensitivity_summary=compare_variations(means))
    counts = tables["cap_violations_by_task"].groupby(["model", "kappa"]).agg(
        n_violations=("n_violations", "sum"), n_predictions=("n_predictions", "sum"),
        zero_history_violations=("zero_history_violations", "sum"), n_tasks=("dataset", "nunique"),
    ).reset_index()
    counts["violation_share"] = counts.n_violations / counts.n_predictions
    counts["pretty_model"] = counts.model.map(pretty_model_name)
    counts["family"] = counts.model.map(model_family)
    tables["cap_violations_by_model"] = counts
    tables["highlighted_models"] = select_highlights(means)
    for name, table in tables.items():
        if name != "component_values":
            table.to_csv(output / f"{name}.csv", index=False)
    plot_parameter_curves(means, tables["highlighted_models"], output / "asym_cap_sensitivity")
    save_overview_figures(means, tables["sensitivity_summary"], output)
    save_report(tables, output)
    summary = tables["sensitivity_summary"]
    print(summary.loc[~summary.is_original, ["variation", "spearman", "top5_overlap", "leader_names", "leader_changed"]].to_string(index=False))
    print(f"20 modelos × 10 tarefas; {len(summary)} cenários incluindo originais e robustez. Saídas: {output}")


if __name__ == "__main__":
    main()
