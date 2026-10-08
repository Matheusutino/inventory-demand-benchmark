"""Plot aggregate demand and training-only ADI/CV² for each dataset panel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis.plot_item_sum_mae import DATASET_LABELS, PANEL_ORDER
from src.utils.data_loader import discover_datasets


ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49
CATEGORY_COLORS = {
    "smooth": "#0072B2",
    "erratic": "#E69F00",
    "intermittent": "#009E73",
    "lumpy": "#CC79A7",
}
HORIZON_COLORS = {3: "#7B3294", 6: "#D95F02"}


@dataclass
class Panel:
    name: str
    full: pd.DataFrame
    test_starts: dict[int, pd.Timestamp]

    @property
    def label(self) -> str:
        return DATASET_LABELS.get(self.name, self.name)

    @property
    def common_train(self) -> pd.DataFrame:
        return self.full.loc[self.full.index < min(self.test_starts.values())]


def read_wide(path: str) -> pd.DataFrame:
    """Read a complete monthly panel, normalizing Parquet timestamp precision."""
    frame = pd.read_parquet(path).set_index("date").sort_index().astype(float)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).as_unit("ns")
    frame.columns = frame.columns.astype(str)
    frame = frame.reindex(sorted(frame.columns), axis=1)
    if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"Painel vazio ou com datas/SKUs duplicados: {path}")
    expected = pd.date_range(frame.index.min(), periods=len(frame), freq="MS")
    if not frame.index.equals(expected):
        raise ValueError(f"Datas mensais incompletas ou fora do início do mês: {path}")
    if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() < 0).any():
        raise ValueError(f"Demanda ausente, não finita ou negativa: {path}")
    return frame


def load_panels(data_dir: str, names: list[str] | None = None) -> list[Panel]:
    """Reconstruct each history once and verify agreement across horizon files."""
    grouped = {}
    for name, paths in discover_datasets(data_dir).items():
        panel_name, separator, suffix = name.rpartition("_h")
        if not separator or not suffix.isdigit():
            continue
        if names and panel_name not in names:
            continue
        horizon = int(suffix)
        train, test = read_wide(paths["train"]), read_wide(paths["test"])
        if not train.columns.equals(test.columns) or len(test) != horizon:
            raise ValueError(f"SKUs/horizonte inconsistentes em {name}.")
        if test.index[0] != train.index[-1] + pd.offsets.MonthBegin(1):
            raise ValueError(f"Teste não continua os meses de treino em {name}.")
        full = pd.concat([train, test])
        if panel_name in grouped:
            existing = grouped[panel_name]
            if not existing.full.equals(full):
                raise ValueError(f"Históricos H3/H6 divergentes no painel {panel_name}.")
            if full.index[-horizon] != test.index[0]:
                raise ValueError(f"Janelas de teste não aninhadas em {name}.")
            existing.test_starts[horizon] = test.index[0]
        else:
            grouped[panel_name] = Panel(panel_name, full, {horizon: test.index[0]})
    if not grouped:
        raise ValueError(f"Nenhum painel *_hN encontrado em {data_dir}.")
    if names and (missing := set(names) - set(grouped)):
        raise ValueError(f"Painéis não encontrados: {sorted(missing)}")
    return sorted(grouped.values(), key=lambda panel: (
        PANEL_ORDER.index(panel.label) if panel.label in PANEL_ORDER else len(PANEL_ORDER),
        panel.label,
    ))


def classify_demand(adi: float, cv2: float) -> str:
    """SBC quadrants, assigning equality to the high-ADI/high-CV² side."""
    if adi < ADI_CUTOFF:
        return "smooth" if cv2 < CV2_CUTOFF else "erratic"
    return "intermittent" if cv2 < CV2_CUTOFF else "lumpy"


def intermittency_statistics(panel: Panel) -> pd.DataFrame:
    """ADI=T/N+; CV² uses sample variance of positive demand only (ddof=1)."""
    history = panel.common_train
    rows = []
    for sku in history.columns:
        values = history[sku].to_numpy()
        positive = values[values > 0]
        adi = len(values) / len(positive) if len(positive) else np.nan
        cv2 = positive.var(ddof=1) / positive.mean() ** 2 if len(positive) >= 2 else np.nan
        category = (
            "no_demand" if not len(positive) else
            "insufficient_history" if len(positive) == 1 else
            classify_demand(adi, cv2)
        )
        rows.append({
            "panel": panel.name, "panel_label": panel.label, "id": sku,
            "train_start": history.index[0], "train_end": history.index[-1],
            "n_months": len(values), "positive_count": len(positive),
            "zero_rate": float((values == 0).mean()),
            "adi": adi, "cv2_positive": cv2, "category": category,
        })
    return pd.DataFrame(rows)


def aggregate_statistics(panel: Panel, scale: str) -> pd.DataFrame:
    """Use a fixed training baseline for all plotted months, including test."""
    demand = panel.full.sum(axis=1)
    if scale == "index":
        baseline = float(panel.common_train.sum(axis=1).mean())
        if baseline <= 0:
            raise ValueError(f"Treino sem demanda em {panel.name}; use --scale absolute.")
        demand = 100 * demand / baseline
    frame = pd.DataFrame({"panel": panel.name, "date": panel.full.index, "demand": demand.to_numpy(), "scale": scale})
    for horizon, start in panel.test_starts.items():
        frame[f"is_test_h{horizon}"] = frame["date"] >= start
    return frame


def draw_aggregate(ax, panel: Panel, values: pd.DataFrame, scale: str) -> None:
    end = panel.full.index[-1] + pd.offsets.MonthBegin(1)
    for horizon, start in sorted(panel.test_starts.items(), reverse=True):
        color = HORIZON_COLORS.get(horizon, "#555555")
        ax.axvspan(start, end, color=color, alpha=0.10, linewidth=0)
        ax.axvline(start, color=color, linestyle="--", linewidth=1.1)
    ax.plot(values["date"], values["demand"], color="#263238", linewidth=1.35)
    if scale == "index":
        ax.axhline(100, color="#999999", linewidth=0.7, linestyle=":")
    ax.set_xlabel(f"Year\n{panel.label}")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(panel.full.index[0], end)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.18)


def draw_intermittency(ax, stats: pd.DataFrame, limits: tuple[float, float]) -> None:
    for category, color in CATEGORY_COLORS.items():
        points = stats.loc[stats.category == category]
        ax.scatter(points.adi, points.cv2_positive, s=13, color=color, alpha=0.60, edgecolors="none", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlim(0.95, limits[0])
    ax.set_ylim(0, limits[1])
    ticks = [value for value in (1, 2, 5, 10, 20, 50) if value < limits[0]]
    ax.set_xticks(ticks, labels=[str(value) for value in ticks])
    ax.set_yticks([value for value in (0, 0.1, 0.49, 1, 3, 10, 30) if value < limits[1]])
    ax.axvline(ADI_CUTOFF, color="#777777", linestyle="--", linewidth=0.8)
    ax.axhline(CV2_CUTOFF, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("ADI (log scale)")
    counts = stats.category.value_counts()
    ax.text(0.98, 0.98, f"No demand: {counts.get('no_demand', 0)}\nSingle event: {counts.get('insufficient_history', 0)}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.grid(alpha=0.12)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_overview(panels: list[Panel], output_dir: str, scale: str, plots: list[str]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregates = [aggregate_statistics(panel, scale) for panel in panels]
    statistics = [intermittency_statistics(panel) for panel in panels]
    pd.concat(aggregates, ignore_index=True).to_csv(output / "dataset_aggregate_demand.csv", index=False)
    sku_stats = pd.concat(statistics, ignore_index=True)
    sku_stats.to_csv(output / "dataset_adi_cv2.csv", index=False)
    counts = sku_stats.groupby(["panel", "category"]).size().unstack(fill_value=0)
    counts.to_csv(output / "dataset_demand_category_counts.csv")
    classified = sku_stats.loc[sku_stats.category.isin(CATEGORY_COLORS)]
    limits = (max(2, classified.adi.max() * 1.1) if len(classified) else 2,
              max(1, classified.cv2_positive.max() * 1.2) if len(classified) else 1)
    horizon_legend = [Line2D([0], [0], color=HORIZON_COLORS.get(h, "#555555"), linestyle="--", label=f"H{h} test start") for h in sorted({h for p in panels for h in p.test_starts}, reverse=True)]
    category_legend = [Line2D([0], [0], color=color, marker="o", linestyle="none", label=name.title()) for name, color in CATEGORY_COLORS.items()]
    ylabel = "Demand index" if scale == "index" else "Monthly demand"
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        for plot in plots:
            combined = plot == "overview"
            rows = 2 if combined else 1
            fig, axes = plt.subplots(rows, len(panels), figsize=(3.15 * len(panels), 7.1 if combined else 3.9), squeeze=False)
            for index, panel in enumerate(panels):
                if plot in {"aggregate", "overview"}:
                    draw_aggregate(axes[0, index], panel, aggregates[index], scale)
                if plot in {"adi_cv2", "overview"}:
                    ax = axes[-1, index]
                    draw_intermittency(ax, statistics[index], limits)
                    ax.set_xlabel(f"ADI (log scale)\n{panel.label}")
            if plot in {"aggregate", "overview"}:
                axes[0, 0].set_ylabel(ylabel)
                if scale == "index":
                    upper = max(values.demand.max() for values in aggregates) * 1.08
                    for ax in axes[0]:
                        ax.set_ylim(0, upper)
            if plot in {"adi_cv2", "overview"}:
                axes[-1, 0].set_ylabel("CV² of positive demand (symlog)")
            legend_handles = (
                horizon_legend + category_legend if combined else
                horizon_legend if plot == "aggregate" else category_legend
            )
            legend_columns = min(len(legend_handles), 2 * len(panels))
            legend_rows = (len(legend_handles) + legend_columns - 1) // legend_columns
            fig.legend(handles=legend_handles, loc="lower center", ncol=legend_columns, frameon=False, bbox_to_anchor=(0.5, 0.01))
            fig.tight_layout(rect=(0, 0.07 * legend_rows, 1, 1), h_pad=2.7)
            save_figure(fig, output / f"dataset_{plot if plot != 'aggregate' else 'aggregate_demand'}")
    print(f"✓ {len(panels)} painéis; figuras PDF e tabelas CSV em {output}")
    print(counts.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--output-dir", default="data/analysis/dataset/overview")
    parser.add_argument("--panels", nargs="+", default=None, help="Nomes sem _h3/_h6; ex.: Filtros Mecanismos.")
    parser.add_argument("--scale", choices=["index", "absolute"], default="index")
    parser.add_argument("--plots", nargs="+", choices=["aggregate", "adi_cv2", "overview"], default=["aggregate", "adi_cv2", "overview"])
    args = parser.parse_args()
    plot_overview(load_panels(args.data_dir, args.panels), args.output_dir, args.scale, args.plots)


if __name__ == "__main__":
    main()
