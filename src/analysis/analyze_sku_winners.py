"""Measure per-SKU MAE winners, count/volume win shares and a retrospective model-selection oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from src.analysis.analyze_predictions import load_ground_truth
from src.analysis.model_style import FAMILY_COLORS, FAMILY_LABELS, FAMILY_ORDER, model_family, pretty_model_name
from src.analysis.plot_dataset_overview import intermittency_statistics, load_panels
from src.analysis.plot_item_sum_mae import PANEL_ORDER, model_name_from_path, panel_name, prediction_files


SKU_KEYS = ["dataset", "id"]


def load_sku_mae(data_dir: str, predictions_dir: str, models: list[str] | None = None) -> pd.DataFrame:
    """Recalculate MAE from saved forecasts, requiring complete identical SKU/month coverage."""
    truth = load_ground_truth(data_dir)
    truth["timestamp"] = truth.timestamp.dt.as_unit("ns")
    keys = ["dataset", "id", "timestamp"]
    if truth.duplicated(keys).any() or not np.isfinite(truth.target.to_numpy()).all():
        raise ValueError("Targets duplicados ou não finitos.")
    expected = pd.MultiIndex.from_frame(truth[keys])
    coverage = truth.groupby(SKU_KEYS).agg(n_points=("target", "size"), test_total_demand=("target", "sum"))
    rows = []
    for path in prediction_files(predictions_dir, models):
        pred = pd.read_parquet(path)
        if missing := {*keys, "q50"} - set(pred.columns):
            raise ValueError(f"{path.name}: colunas ausentes: {sorted(missing)}")
        pred["id"] = pred.id.astype(str)
        pred["timestamp"] = pd.to_datetime(pred.timestamp).dt.as_unit("ns")
        if pred.duplicated(keys).any() or not np.isfinite(pred.q50.to_numpy()).all():
            raise ValueError(f"{path.name}: previsões duplicadas ou não finitas.")
        observed = pd.MultiIndex.from_frame(pred[keys])
        if len(expected.difference(observed)) or len(observed.difference(expected)):
            raise ValueError(f"{path.name}: cobertura SKU/timestamp diferente dos targets.")
        merged = truth.merge(pred[keys + ["q50"]], on=keys, validate="one_to_one")
        merged["mae"] = (merged.q50 - merged.target).abs()
        means = merged.groupby(SKU_KEYS).mae.mean().to_frame().join(coverage).reset_index()
        means["model"] = model_name_from_path(path)
        rows.append(means)
    return pd.concat(rows, ignore_index=True)


def load_training_categories(data_dir: str) -> pd.DataFrame:
    """Use the existing common-training SBC categories, with no test-demand classification."""
    tables = []
    for panel in load_panels(data_dir):
        stats = intermittency_statistics(panel)
        tables.append(stats[["panel", "id", "category", "n_months", "adi", "cv2_positive"]])
    return pd.concat(tables, ignore_index=True).rename(columns={"n_months": "category_train_months"})


def _oracle_row(group: pd.DataFrame, means: pd.Series) -> dict:
    """Compare the mean of per-SKU minima with the minimum of model mean MAEs."""
    best = means.min()
    best_models = sorted(means.index[means == best], key=lambda name: (pretty_model_name(name), name))
    oracle = group.groupby(SKU_KEYS).mae.min().mean()
    return {
        "best_model": best_models[0], "best_model_name": pretty_model_name(best_models[0]),
        "co_best_single_models": " | ".join(best_models), "best_single_mae": best,
        "oracle_mae": oracle, "oracle_gain_absolute": best - oracle,
        "oracle_gain_pct": 100 * (best - oracle) / best if best > 0 else np.nan,
    }


def compute_sku_winners(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Give exact tied models equal credit and exact tied families equal credit separately."""
    required = {"dataset", "id", "model", "mae"}
    if missing := required - set(metrics):
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    data = metrics.copy()
    keys = [*SKU_KEYS, "model"]
    if data.empty or data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError("Resultados vazios, identificadores ausentes ou pares SKU/modelo duplicados.")
    data["mae"] = pd.to_numeric(data.mae, errors="raise")
    if not np.isfinite(data.mae.to_numpy()).all() or (data.mae < 0).any():
        raise ValueError("MAEs devem ser finitos e não negativos.")
    data["id"] = data.id.astype(str)
    if data.duplicated(keys).any():
        raise ValueError("IDs de SKU duplicados após normalização.")
    data["panel"] = data.dataset.str.replace(r"_h[36]$", "", regex=True)
    suffix = data.dataset.str.extract(r"_h([36])$", expand=False)
    if suffix.isna().any():
        raise ValueError("As tarefas devem identificar H3 ou H6 com o sufixo _h3/_h6.")
    data["horizon"] = suffix.astype(int)
    data["panel_label"] = data.dataset.map(panel_name)
    data["pretty_model"] = data.model.map(pretty_model_name)
    data["family"] = data.model.map(model_family)
    n_models = data.model.nunique()
    if (data.groupby(SKU_KEYS).model.nunique() != n_models).any():
        raise ValueError("Todos os SKUs devem ter os mesmos modelos concorrentes.")
    if "n_points" in data and (data.n_points != data.horizon).any():
        raise ValueError("Cobertura mensal incompleta: n_points deve coincidir com H3/H6.")
    if "category" not in data:
        data["category"] = "unclassified"
    if "test_total_demand" in data:
        data["test_total_demand"] = pd.to_numeric(data.test_total_demand, errors="raise")
        if not np.isfinite(data.test_total_demand.to_numpy()).all() or (data.test_total_demand < 0).any():
            raise ValueError("Volumes de demanda devem ser finitos e não negativos.")
    for field in ("category", "test_total_demand"):
        if field in data and (data.groupby(SKU_KEYS)[field].nunique(dropna=False) != 1).any():
            raise ValueError(f"Metadados de SKU divergentes entre modelos: {field}")
    data["best_sku_mae"] = data.groupby(SKU_KEYS).mae.transform("min")
    winners = data.loc[data.mae == data.best_sku_mae].copy()
    winners["n_tied_models"] = winners.groupby(SKU_KEYS).model.transform("size")
    winners["model_win_credit"] = 1 / winners.n_tied_models
    winners["n_tied_families"] = winners.groupby(SKU_KEYS).family.transform("nunique")
    winners["sole_model_win"] = (winners.n_tied_models == 1).astype(int)
    family_winners = winners.drop_duplicates([*SKU_KEYS, "family"]).copy()
    family_winners["family_win_credit"] = 1 / family_winners.n_tied_families
    sku = data.drop_duplicates(SKU_KEYS)[[*SKU_KEYS, "panel", "panel_label", "horizon", "category", "best_sku_mae"]].copy()
    if "test_total_demand" in data:
        sku["test_total_demand"] = data.drop_duplicates(SKU_KEYS).test_total_demand.to_numpy()
    sku = sku.merge(winners.groupby(SKU_KEYS).agg(n_tied_models=("model", "size"),
                    n_tied_families=("family", "nunique")), on=SKU_KEYS, validate="one_to_one")
    sizes = sku.groupby("dataset").size()
    model_means = data.groupby(["dataset", "model"]).mae.mean().rename("mean_mae")
    model_wins = winners.groupby(["dataset", "model"]).agg(
        fractional_wins=("model_win_credit", "sum"), co_best_skus=("id", "size"),
        sole_wins=("sole_model_win", "sum"),
    ).reindex(model_means.index, fill_value=0).join(model_means).reset_index()
    model_wins["n_skus"] = model_wins.dataset.map(sizes)
    model_wins["win_share"] = model_wins.fractional_wins / model_wins.n_skus
    model_wins["co_best_share"] = model_wins.co_best_skus / model_wins.n_skus
    model_wins["sole_win_share"] = model_wins.sole_wins / model_wins.n_skus
    model_wins["pretty_model"] = model_wins.model.map(pretty_model_name)
    model_wins["family"] = model_wins.model.map(model_family)
    family_index = pd.MultiIndex.from_product([sizes.index, FAMILY_ORDER], names=["dataset", "family"])
    family_wins = family_winners.groupby(["dataset", "family"]).family_win_credit.sum().reindex(family_index, fill_value=0).rename("fractional_wins").reset_index()
    family_wins["n_skus"] = family_wins.dataset.map(sizes)
    family_wins["win_share"] = family_wins.fractional_wins / family_wins.n_skus
    task_metadata = sku[["dataset", "panel", "panel_label", "horizon"]].drop_duplicates()
    family_wins = family_wins.merge(task_metadata, on="dataset", validate="many_to_one")
    model_wins = model_wins.merge(task_metadata, on="dataset", validate="many_to_one")
    summary_rows = []
    for dataset, group in data.groupby("dataset", sort=True):
        task_skus = sku.loc[sku.dataset == dataset]
        row = {**task_metadata.loc[task_metadata.dataset == dataset].iloc[0].to_dict(),
               "n_skus": len(task_skus), "n_models": n_models,
               "tied_skus": int((task_skus.n_tied_models > 1).sum()),
               **_oracle_row(group, group.groupby("model").mae.mean())}
        row["tied_sku_share"] = row["tied_skus"] / row["n_skus"]
        if "test_total_demand" in task_skus:
            row["zero_test_skus"] = int((task_skus.test_total_demand == 0).sum())
        best_stats = model_wins.loc[(model_wins.dataset == dataset) & (model_wins.model == row["best_model"])].iloc[0]
        row.update({f"best_model_{col}": best_stats[col] for col in ("win_share", "co_best_share", "sole_win_share")})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    overall_rows = []
    for horizon, group in data.groupby("horizon", sort=True):
        # Give panels equal weight for MAE, preserving the benchmark's existing aggregation.
        means = group.groupby(["dataset", "model"]).mae.mean().groupby("model").mean()
        row = {"horizon": horizon, "n_panels": group.dataset.nunique(),
               "n_skus": len(group[SKU_KEYS].drop_duplicates()), **_oracle_row(group, means)}
        row["oracle_mae"] = summary.loc[summary.horizon == horizon].oracle_mae.mean()
        row["oracle_gain_absolute"] = row["best_single_mae"] - row["oracle_mae"]
        row["oracle_gain_pct"] = 100 * row["oracle_gain_absolute"] / row["best_single_mae"] if row["best_single_mae"] > 0 else np.nan
        chosen = model_wins.loc[(model_wins.horizon == horizon) & (model_wins.model == row["best_model"])]
        row["best_model_win_share"] = chosen.fractional_wins.sum() / row["n_skus"]
        row["best_model_co_best_share"] = chosen.co_best_skus.sum() / row["n_skus"]
        overall_rows.append(row)
    category_sizes = sku.groupby(["dataset", "category"]).size().rename("n_skus")
    cat_index = pd.MultiIndex.from_tuples(
        [(dataset, category, model) for dataset, category in category_sizes.index for model in data.model.unique()],
        names=["dataset", "category", "model"],
    )
    by_category = winners.groupby(["dataset", "category", "model"]).model_win_credit.sum().reindex(cat_index, fill_value=0).rename("fractional_wins").reset_index()
    by_category = by_category.merge(category_sizes.reset_index(), on=["dataset", "category"], validate="many_to_one")
    by_category["win_share"] = by_category.fractional_wins / by_category.n_skus
    by_category = by_category.merge(model_wins[["dataset", "model", "fractional_wins"]].rename(columns={"fractional_wins": "total_model_wins"}), on=["dataset", "model"], validate="many_to_one")
    by_category["share_of_model_wins"] = by_category.fractional_wins / by_category.total_model_wins.replace(0, np.nan)
    by_category["pretty_model"] = by_category.model.map(pretty_model_name)
    by_category = by_category.merge(task_metadata, on="dataset", validate="many_to_one")
    tables = {"sku_model_mae": data, "sku_winners": winners, "sku_family_winners": family_winners,
              "sku_summary": sku, "model_win_shares": model_wins, "family_win_shares": family_wins,
              "oracle_by_panel": summary, "oracle_overall": pd.DataFrame(overall_rows),
              "model_wins_by_category": by_category}
    if "test_total_demand" in sku:
        add_volume_win_shares(tables)
    return tables


def add_volume_win_shares(tables: dict[str, pd.DataFrame]) -> None:
    """Weight the existing MAE winner credits by realized test demand, without changing winners."""
    totals = tables["sku_summary"].groupby("dataset").test_total_demand.sum()
    for key in ("model", "family"):
        winner_table = tables[f"sku_{'winners' if key == 'model' else 'family_winners'}"]
        winner_table[f"{key}_volume_credit"] = winner_table.test_total_demand * winner_table[f"{key}_win_credit"]
        credited = winner_table.groupby(["dataset", key])[f"{key}_volume_credit"].sum()
        by_task = tables[f"{key}_win_shares"].set_index(["dataset", key])
        by_task["won_test_volume"] = credited.reindex(by_task.index, fill_value=0)
        by_task = by_task.reset_index()
        by_task["total_test_volume"] = by_task.dataset.map(totals)
        by_task["volume_win_share"] = by_task.won_test_volume / by_task.total_test_volume.replace(0, np.nan)
        tables[f"{key}_win_shares"] = by_task
        overall = by_task.groupby(["horizon", key]).agg(
            fractional_wins=("fractional_wins", "sum"), n_skus=("n_skus", "sum"),
            won_test_volume=("won_test_volume", "sum"), total_test_volume=("total_test_volume", "sum"),
            n_panels=("dataset", "nunique"), n_panels_with_volume=("volume_win_share", "count"),
            equal_panel_win_share=("win_share", "mean"),
            equal_panel_volume_win_share=("volume_win_share", "mean"),
        ).reset_index()
        overall["win_share"] = overall.fractional_wins / overall.n_skus
        overall["volume_win_share"] = overall.won_test_volume / overall.total_test_volume.replace(0, np.nan)
        if key == "model":
            overall["pretty_model"] = overall.model.map(pretty_model_name)
            overall["family"] = overall.model.map(model_family)
        tables[f"{key}_win_shares_overall"] = overall


def plot_count_and_volume_shares(tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Compare count and demand-volume family win shares by panel and horizon, plus pooled totals."""
    shares = tables["family_win_shares"]
    overall = tables["family_win_shares_overall"].copy()
    pooled_label = "All panels (pooled)"
    overall["panel_label"] = pooled_label
    shares = pd.concat([shares, overall], ignore_index=True)
    labels = [label for label in PANEL_ORDER if label in set(shares.panel_label)]
    labels += sorted(set(shares.panel_label) - set(labels) - {pooled_label})
    labels.append(pooled_label)
    horizons = sorted(shares.horizon.unique())
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10}):
        fig, axes = plt.subplots(len(horizons), 2, figsize=(16, 4 * len(horizons) + 1),
                                 sharey="row", squeeze=False)
        for row, horizon in enumerate(horizons):
            task = shares.loc[shares.horizon == horizon]
            for col, (field, label) in enumerate((("win_share", "SKU win share"),
                                                  ("volume_win_share", "Test-demand volume win share"))):
                ax = axes[row, col]
                y = np.arange(len(labels))
                offset = np.zeros(len(labels))
                for family in FAMILY_ORDER:
                    segment = task.loc[task.family == family].set_index("panel_label")[field].reindex(labels).fillna(0).to_numpy() * 100
                    ax.barh(y, segment, left=offset, color=FAMILY_COLORS[family], height=.7)
                    for pos, value, start in zip(y, segment, offset):
                        if value >= 4:
                            ax.text(start + value / 2, pos, f"{value:.1f}%", ha="center", va="center",
                                    color="white", fontsize=9)
                    offset += segment
                for pos in y[offset == 0]:
                    ax.text(50, pos, "No test demand", ha="center", va="center", color="#555555")
                ax.set_yticks(y, labels=labels)
                if col == 0:
                    ax.invert_yaxis()
                ax.set_xlim(0, 100)
                ax.xaxis.set_major_formatter(PercentFormatter(100))
                ax.set_xlabel(f"H{horizon} — {label}")
                ax.axhline(len(labels) - 1.5, color="#555555", linestyle=":", linewidth=.8)
        fig.legend(handles=[Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in FAMILY_ORDER],
                   loc="lower center", ncol=len(FAMILY_ORDER), frameon=False, fontsize=9, bbox_to_anchor=(.5, .035))
        fig.text(.5, .012,
                 "Volume = total observed demand per SKU over the test window (H3 or H6); "
                 "SKUs with zero test demand have zero volume weight.",
                 ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, .085, 1, 1), h_pad=2.5)
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_wins_and_oracle(tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot family shares and relative oracle improvement for each panel, separately by horizon."""
    summary, shares = tables["oracle_by_panel"], tables["family_win_shares"]
    horizons = sorted(summary.horizon.unique())
    labels = [label for label in PANEL_ORDER if label in set(summary.panel_label)]
    labels += sorted(set(summary.panel_label) - set(labels))
    maximum = summary.oracle_gain_pct.max()
    gain_limit = max(5., maximum * 1.25) if np.isfinite(maximum) else 5.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10}):
        fig, axes = plt.subplots(len(horizons), 2, figsize=(16, 3.5 * len(horizons) + 1),
                                 sharey="row", squeeze=False, gridspec_kw={"width_ratios": [1.5, 1]})
        for row, horizon in enumerate(horizons):
            left, right = axes[row]
            task = summary.loc[summary.horizon == horizon].set_index("panel_label").reindex(labels)
            y = np.arange(len(labels))
            offset = np.zeros(len(labels))
            for family in FAMILY_ORDER:
                segment = shares.loc[(shares.horizon == horizon) & (shares.family == family)].set_index("panel_label").win_share.reindex(labels, fill_value=0).to_numpy() * 100
                left.barh(y, segment, left=offset, color=FAMILY_COLORS[family])
                for pos, value, start in zip(y, segment, offset):
                    if value >= 4:
                        left.text(start + value / 2, pos, f"{value:.1f}%", ha="center", va="center", color="white", fontsize=9)
                offset += segment
            left.set_yticks(y, labels=labels)
            left.invert_yaxis()
            left.set_xlim(0, 100)
            left.xaxis.set_major_formatter(PercentFormatter(100))
            left.set_xlabel(f"H{horizon} — SKU win share (equal credit per tied family)")
            gains = task.oracle_gain_pct.to_numpy()
            right.barh(y, gains, color="#333333")
            for pos, value in zip(y, gains):
                right.text((value if np.isfinite(value) else 0) + gain_limit * .02, pos,
                           f"{value:.1f}%" if np.isfinite(value) else "—", va="center", fontsize=10)
            right.set_xlim(0, gain_limit)
            right.xaxis.set_major_formatter(PercentFormatter(100))
            right.set_xlabel(f"H{horizon} — Oracle MAE reduction vs best single model")
            right.grid(axis="x", alpha=.2)
            right.set_axisbelow(True)
        fig.legend(handles=[Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in FAMILY_ORDER],
                   loc="lower center", ncol=len(FAMILY_ORDER), frameon=False, fontsize=9, bbox_to_anchor=(.5, .005))
        fig.tight_layout(rect=(0, .055, 1, 1), h_pad=2.5)
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _volume_report(tables: dict[str, pd.DataFrame]) -> list[str]:
    if "family_win_shares_overall" not in tables:
        return []
    lines = ["", "## Win share ponderado pelo volume", "",
        "Para cada SKU, V_i = soma da demanda observada nos meses de teste. Mantêm-se os mesmos "
        "vencedores por MAE; a participação por volume é soma(V_i × crédito de vitória) / soma(V_i). "
        "Um empate divide o volume com o mesmo critério usado para contar vitórias: 1/k por modelo "
        "e 1/f por família. SKUs sem demanda no teste têm peso zero nesta versão, mas continuam "
        "na versão por contagem. Painéis com volume total zero têm participação por volume indefinida (NaN).", "",
        "A linha global soma os volumes dos painéis antes de normalizar. Portanto, os painéis "
        "com maior demanda pesam mais; H3 e H6 são calculados separadamente. Os CSVs também "
        "reportam a média das participações dos painéis com pesos iguais (`equal_panel_volume_win_share`), "
        "excluindo painéis sem volume e informando `n_panels_with_volume`. A estatística descreve "
        "a demanda dos SKUs vencidos em retrospecto, em unidades de demanda.", "",
        "Figura: [contagem de SKUs × volume](sku_family_wins_by_count_and_volume.pdf).", "",
        "**Legenda da figura:** Participação de vitórias por família, por contagem de SKUs (esquerda) "
        "e ponderada pelo volume (direita). Volume é a demanda total observada de cada SKU na janela "
        "de teste correspondente (H3 ou H6). SKUs sem demanda no teste têm peso zero na versão "
        "por volume e continuam contando na versão por SKUs. Empates dividem o crédito igualmente "
        "entre as famílias vencedoras.", "",
        "| Horizonte | Família | Win share por SKUs | Win share por volume | Volume creditado |",
        "| --- | --- | ---: | ---: | ---: |"]
    overall = tables["family_win_shares_overall"]
    tasks = tables["family_win_shares"]
    for horizon, group in overall.groupby("horizon", sort=True):
        for family in FAMILY_ORDER:
            r = group.loc[group.family == family].iloc[0]
            lines.append(f"| H{horizon} | {FAMILY_LABELS[family]} | {100*r.win_share:.1f}% | "
                         f"{100*r.volume_win_share:.1f}% | {r.won_test_volume:.2f} |")
    lines.extend(["", "Leitura dos resultados:", ""])
    for horizon, group in overall.groupby("horizon", sort=True):
        count_max = group.win_share.max()
        count_leaders = ", ".join(group.loc[group.win_share == count_max, "family"].map(FAMILY_LABELS))
        if group.volume_win_share.notna().any():
            volume_max = group.volume_win_share.max()
            volume_leaders = ", ".join(group.loc[group.volume_win_share == volume_max, "family"].map(FAMILY_LABELS))
            task = tasks.loc[tasks.horizon == horizon]
            maxima = task.groupby("dataset").volume_win_share.transform("max")
            foundation_leads = task.loc[(task.family == "foundation") & (task.volume_win_share == maxima)].dataset.nunique()
            lines.append(f"- H{horizon}: {count_leaders} lidera por número de SKUs ({100*count_max:.1f}%); "
                         f"{volume_leaders} lidera por volume ({100*volume_max:.1f}%). "
                         f"Foundation models lidera ou divide a liderança por volume em "
                         f"{foundation_leads}/{task.dataset.nunique()} painéis.")
        else:
            lines.append(f"- H{horizon}: nenhum volume de demanda observado; win share por volume indefinido.")
    lines.extend(["", "| Painel | Horizonte | Família(s) líder(es) por volume | Win share por volume |",
                  "| --- | --- | --- | ---: |"])
    for (_, horizon), group in tasks.groupby(["dataset", "horizon"], sort=True):
        maximum = group.volume_win_share.max()
        leaders = ", ".join(group.loc[group.volume_win_share == maximum, "family"].map(FAMILY_LABELS)) or "Indefinido"
        lines.append(f"| {group.panel_label.iloc[0]} | H{horizon} | {leaders} | {100*maximum:.1f}% |")
    return lines


def write_report(tables: dict[str, pd.DataFrame], output: Path) -> None:
    """Explain model-versus-family ties, panel weights and the retrospective nature of the oracle."""
    lines = ["# Vitórias por SKU e oráculo", "",
        "O MAE é calculado sobre os meses de teste de cada SKU, separadamente em H3 e H6. "
        "Todos os modelos competem nos mesmos SKUs e timestamps, usando as previsões salvas sem truncamento.", "",
        "Empates usam igualdade exata de MAE. Para modelos, cada co-vencedor recebe 1/k. "
        "Para famílias, cada família com algum co-vencedor recebe 1/f, independentemente do número "
        "de modelos empatados nela. Assim, as barras somam 100% e uma família com mais modelos "
        "não ganha crédito adicional só por empates. A participação de uma família pode diferir "
        "da soma das participações fracionadas de seus modelos. Vitórias exclusivas e presenças "
        "entre os co-vencedores são reportadas separadamente nos CSVs.", "",
        "Oráculo: média dos menores MAEs de cada SKU. Melhor modelo único: menor média de MAE "
        "de um modelo fixo em todos os SKUs do painel/horizonte. Ganho: "
        "100 × (MAE_melhor_único − MAE_oráculo) / MAE_melhor_único. Se o denominador é zero, "
        "o ganho percentual é indefinido (NaN). O ganho absoluto continua disponível.", "",
        "O oráculo é retrospectivo: escolhe um modelo por SKU usando toda a janela de teste, "
        "sem trocar de modelo entre meses. É um limite do ganho possível nestas previsões; "
        "uma política aplicável precisa escolher os modelos usando treino/validação e ser avaliada "
        "em teste separado. O experimento não valida essa política.", "",
        "## Por painel e horizonte", "",
        "| Painel | Horizonte | SKUs | Melhor modelo único | MAE único | MAE oráculo | Ganho | Vitórias do melhor modelo¹ | SKUs com empate | SKUs sem demanda no teste |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in tables["oracle_by_panel"].itertuples():
        lines.append(f"| {r.panel_label} | H{r.horizon} | {r.n_skus} | {r.best_model_name} | "
                     f"{r.best_single_mae:.3f} | {r.oracle_mae:.3f} | {r.oracle_gain_pct:.1f}% | "
                     f"{100*r.best_model_win_share:.1f}% | {r.tied_skus} | {getattr(r, 'zero_test_skus', '—')} |")
    lines.extend(["", "¹ Crédito fracionado entre modelos; não é a frequência de presença entre co-vencedores.", "",
                  "## Visão global por horizonte", "",
                  "MAEs globais dão peso igual a cada painel, como as figuras de Item do benchmark. "
                  "As frações de vitórias globais dão peso igual a cada par painel/SKU. "
                  "O melhor modelo único global é fixo em todos os painéis; ele pode diferir dos vencedores por painel.", ""])
    for r in tables["oracle_overall"].itertuples():
        lines.append(f"- H{r.horizon}: {r.best_model_name} tem o menor MAE médio ({r.best_single_mae:.3f}), "
                     f"com {100*r.best_model_win_share:.1f}% das vitórias fracionadas e presença "
                     f"entre os co-vencedores de {100*r.best_model_co_best_share:.1f}% dos SKUs. "
                     f"O oráculo atinge MAE {r.oracle_mae:.3f}: redução de {r.oracle_gain_pct:.1f}%.")
    lines.extend(_volume_report(tables))
    sku = tables["sku_summary"]
    if "test_total_demand" in sku:
        lines.extend(["", "## Influência dos SKUs sem demanda no teste", "",
            "O recorte abaixo dá peso igual a cada par painel/SKU. É descritivo: a demanda "
            "realizada no teste não é informação disponível para selecionar modelos antecipadamente.", ""])
        for horizon, group in sku.groupby("horizon", sort=True):
            n_zero = int((group.test_total_demand == 0).sum())
            n_tied = int((group.n_tied_models > 1).sum())
            lines.append(f"- H{horizon}: {n_zero}/{len(group)} SKUs ({100*n_zero/len(group):.1f}%) "
                         f"não tiveram demanda no teste; {n_tied} ({100*n_tied/len(group):.1f}%) "
                         "tiveram empate entre modelos.")
            family_winners = tables["sku_family_winners"].loc[
                tables["sku_family_winners"].horizon == horizon]
            for zero_test, description in ((True, "sem demanda"), (False, "com demanda")):
                count = int(((group.test_total_demand == 0) == zero_test).sum())
                if count:
                    subset = family_winners.loc[(family_winners.test_total_demand == 0) == zero_test]
                    credits = subset.groupby("family").family_win_credit.sum()
                    leader = credits.idxmax()
                    lines.append(f"  - Entre os {count} SKUs {description} no teste, "
                                 f"{FAMILY_LABELS[leader]} lidera com "
                                 f"{100*credits[leader]/count:.1f}% das vitórias fracionadas por família.")
    lines.extend(["", "## Categorias de demanda e TSB", "",
        "As categorias SBC vêm exclusivamente do treino comum de cada painel (o mesmo da análise "
        "ADI × CV²), mantido igual em H3 e H6. `intermittent` e `lumpy` são reportadas separadamente. "
        "Históricos sem demanda ou com um único evento permanecem como `no_demand` e "
        "`insufficient_history`; todos os SKUs entram na análise de vitórias, inclusive os sem demanda no teste. "
        "A categoria descreve o histórico e não altera a seleção do oráculo.", ""])
    tsb = tables["model_wins_by_category"].query('model == "statsforecast_tsb"')
    if not tsb.empty:
        for horizon, group in tsb.groupby("horizon", sort=True):
            wins = group.fractional_wins.sum()
            n_skus = group.n_skus.sum()
            lines.append(f"- H{horizon}: TSB recebe {wins:.2f} vitórias fracionadas ({100*wins/n_skus:.1f}% dos SKUs).")
            for category, subset in group.groupby("category", sort=True):
                credits = subset.fractional_wins.sum()
                share = 100 * credits / subset.n_skus.sum()
                concentration = 100 * credits / wins if wins else np.nan
                lines.append(f"  - `{category}`: vence {share:.1f}% dos SKUs da categoria; "
                             f"concentra {concentration:.1f}% das vitórias do TSB.")
    (output / "sku_winners_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output-dir", default="data/analysis/accuracy/sku_winners")
    args = parser.parse_args()
    metrics = load_sku_mae(args.data_dir, args.predictions_dir, args.models)
    metrics["panel"] = metrics.dataset.str.replace(r"_h[36]$", "", regex=True)
    metrics = metrics.merge(load_training_categories(args.data_dir), on=["panel", "id"], how="left", validate="many_to_one")
    if metrics.category.isna().any():
        raise ValueError("SKUs sem categoria no histórico de treino.")
    tables = compute_sku_winners(metrics)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    plot_wins_and_oracle(tables, output / "sku_family_wins_and_oracle")
    plot_count_and_volume_shares(tables, output / "sku_family_wins_by_count_and_volume")
    write_report(tables, output)
    print(tables["oracle_overall"].round(4).to_string(index=False))
    print(f"PDF, tabelas e relatório: {output}")


if __name__ == "__main__":
    main()
