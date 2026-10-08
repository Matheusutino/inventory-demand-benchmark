"""Show every framework sensitivity comparison and model rank in two complete PDF views."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.inventory_framework import COMPONENTS, COMPONENT_LABELS
from src.analysis.model_style import FAMILY_COLORS, FAMILY_LABELS, FAMILY_ORDER
from src.analysis.plot_item_sum_mae import PANEL_ORDER, panel_name


PARAMETER_ORDER = ("asym", "cap", "scaled", "zero", "sum", "delta")
PANEL_CODES = dict(zip(PANEL_ORDER, ("HC", "F", "Me", "FM", "CB")))


def setting_label(row: pd.Series, include_component: bool = False) -> str:
    """Keep plot labels short, with original settings marked by an asterisk."""
    if row.kind == "leave_panel_out":
        label = PANEL_CODES[panel_name(row.excluded_panel)]
    elif row.component == "asym":
        label = f"τ={row.setting}"
    elif row.component == "cap":
        label = f"κ={row.setting}"
    elif row.component == "scaled":
        label = {"lag1": "Lag 1", "lag12": "Lag 12", "mean": "Historical mean"}[row.setting]
    elif row.component == "zero":
        label = {"mean": "Historical mean", "max": "Historical max", "none": "Unnormalized"}[row.setting]
    elif row.component in ("sum", "delta"):
        label = {"squared": "Squared", "absolute": "Absolute"}[row.setting]
    else:
        label = "Original"
    if row.is_original:
        label += "*"
    if include_component:
        label = f"{COMPONENT_LABELS[row.component]}: {label}"
    return label


def compute_rank_changes(means: pd.DataFrame) -> pd.DataFrame:
    """Compare every model's mean rank with its same-component ten-task original."""
    if means.duplicated(["component", "variation_id", "model"]).any():
        raise ValueError("Ranks médios duplicados.")
    reference = means.loc[means.is_original, ["component", "model", "mean_rank"]].rename(
        columns={"mean_rank": "original_mean_rank"})
    result = means.merge(reference, on=["component", "model"], how="left", validate="many_to_one")
    if not np.isfinite(result[["mean_rank", "original_mean_rank"]].to_numpy()).all():
        raise ValueError("Ranks ausentes ou não finitos na comparação com o original.")
    result["rank_change"] = result.mean_rank - result.original_mean_rank
    return result


def plot_stability_overview(summary: pd.DataFrame, output: Path) -> None:
    """Show all 14 parameter alternatives and 45 panel omissions, including all requested stability metrics."""
    params = summary.loc[(summary.kind == "parameter") & ~summary.is_original].copy()
    params["order"] = params.component.map({c: i for i, c in enumerate(PARAMETER_ORDER)})
    params["setting_order"] = params.setting.map(lambda s: float(s) if s.replace(".", "", 1).isdigit() else 0)
    params = params.sort_values(["order", "setting_order", "setting"]).reset_index(drop=True)
    omitted = summary.loc[summary.kind == "leave_panel_out"].copy()
    omitted["panel_label"] = omitted.excluded_panel.map(panel_name)
    rho = omitted.pivot(index="component", columns="panel_label", values="spearman").reindex(
        index=COMPONENTS, columns=PANEL_ORDER)
    minimum = min(.8, np.floor(summary.spearman.min() * 10) / 10)
    norm, cmap = Normalize(minimum, 1), plt.get_cmap("YlGnBu")
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1, 1.6]})
        y = np.arange(len(params))
        left.hlines(y, minimum, params.spearman, color="#cccccc", linewidth=1.4)
        for changed, marker in ((False, "o"), (True, "D")):
            mask = params.leader_changed == changed
            left.scatter(params.loc[mask, "spearman"], y[mask], c=params.loc[mask, "spearman"],
                         cmap=cmap, norm=norm, marker=marker, s=65, edgecolor="#333333", linewidth=.7, zorder=3)
        for pos, row in params.iterrows():
            star = "*" if row.original_top5_boundary_tie or row.variant_top5_boundary_tie else ""
            left.text(1.04, pos, f"{row.top5_overlap}/5{star}", transform=left.get_yaxis_transform(), va="center")
            if pos and row.component != params.component.iloc[pos - 1]:
                left.axhline(pos - .5, color="#dddddd", linewidth=.8)
        left.text(1.04, 1.02, "Top-5", transform=left.transAxes, fontsize=10)
        left.set_yticks(y, labels=[setting_label(row, True) for _, row in params.iterrows()])
        left.set_ylim(len(params) - .5, -.5)
        left.set_xlim(minimum - .005, 1.005)
        left.set_xticks(np.arange(minimum, 1.001, .05))
        left.grid(axis="x", alpha=.2)
        left.set_xlabel("Spearman vs original mean ranks")
        left.set_ylabel("Parameter variations")
        im = right.imshow(rho.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        right.set_yticks(range(len(COMPONENTS)), labels=[COMPONENT_LABELS[c] for c in COMPONENTS])
        right.set_xticks(range(len(PANEL_ORDER)), labels=PANEL_ORDER, rotation=35, ha="right")
        right.set_xlabel("Panel omitted (both H3 and H6)")
        right.set_ylabel("Component")
        lookup = omitted.set_index(["component", "panel_label"])
        for i, component in enumerate(COMPONENTS):
            for j, panel in enumerate(PANEL_ORDER):
                row = lookup.loc[(component, panel)]
                star = "*" if row.original_top5_boundary_tie or row.variant_top5_boundary_tie else ""
                color = "white" if norm(row.spearman) > .6 else "black"
                right.text(j, i, f"{row.spearman:.3f}\n{row.top5_overlap}/5{star}",
                           ha="center", va="center", fontsize=9, color=color)
                if row.leader_changed:
                    right.scatter(j + .35, i - .3, marker="D", s=25, facecolor="#333333",
                                  edgecolor="white", linewidth=.7)
        right.set_xticks(np.arange(-.5, len(PANEL_ORDER)), minor=True)
        right.set_yticks(np.arange(-.5, len(COMPONENTS)), minor=True)
        right.grid(which="minor", color="white", linewidth=1.5)
        right.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(im, ax=right, fraction=.035, pad=.025).set_label("Spearman vs original mean ranks")
        fig.legend(handles=[
            Line2D([], [], linestyle="none", marker="o", color="#555555", markerfacecolor="white", label="Leader set unchanged"),
            Line2D([], [], linestyle="none", marker="D", color="#555555", label="Leader set changed"),
        ], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .04))
        fig.text(.5, .015, "Cells: Spearman / original top-5 retained. * Tie at the top-5 cutoff; "
                 "five-name membership uses the documented alphabetical tiebreak.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .115, 1, 1), w_pad=4)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _ordered_scenarios(changes: pd.DataFrame, kind: str) -> pd.DataFrame:
    scenarios = changes.loc[changes.kind == kind].drop_duplicates(["component", "variation_id"]).copy()
    scenarios["component_order"] = scenarios.component.map({c: i for i, c in enumerate(COMPONENTS)})
    if kind == "leave_panel_out":
        scenarios["setting_order"] = scenarios.excluded_panel.map(panel_name).map({p: i for i, p in enumerate(PANEL_ORDER)})
    else:
        def order(row):
            if row.component in ("asym", "cap"):
                return float(row.setting)
            settings = {"scaled": ["lag1", "lag12", "mean"], "zero": ["mean", "max", "none"],
                        "sum": ["squared", "absolute"], "delta": ["squared", "absolute"]}
            return settings.get(row.component, ["original"]).index(row.setting)
        scenarios["setting_order"] = scenarios.apply(order, axis=1)
    return scenarios.sort_values(["component_order", "setting_order"]).reset_index(drop=True)


def plot_all_model_ranks(changes: pd.DataFrame, output: Path) -> None:
    """Show all 1360 mean ranks, with a shared diverging scale for change from the original."""
    models = changes[["model", "pretty_model", "family"]].drop_duplicates().copy()
    models["family_order"] = models.family.map({f: i for i, f in enumerate(FAMILY_ORDER)})
    models = models.sort_values(["family_order", "pretty_model", "model"]).reset_index(drop=True)
    limit = max(1., np.ceil(changes.rank_change.abs().max()))
    norm, cmap = Normalize(-limit, limit), plt.get_cmap("RdBu_r")
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9}):
        fig, axes = plt.subplots(2, 1, figsize=(27, 13))
        for ax, kind in zip(axes, ("parameter", "leave_panel_out")):
            scenarios = _ordered_scenarios(changes, kind)
            columns = pd.MultiIndex.from_frame(scenarios[["component", "variation_id"]])
            subset = changes.loc[changes.kind == kind]
            deltas = subset.pivot(index="model", columns=["component", "variation_id"], values="rank_change").reindex(index=models.model, columns=columns)
            ranks = subset.pivot(index="model", columns=["component", "variation_id"], values="mean_rank").reindex(index=models.model, columns=columns)
            if deltas.isna().any().any() or ranks.isna().any().any():
                raise ValueError("Cobertura incompleta na figura de ranks.")
            im = ax.imshow(deltas.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
            ax.set_yticks(range(len(models)), labels=models.pretty_model)
            ax.set_xticks(range(len(scenarios)), labels=[setting_label(row) for _, row in scenarios.iterrows()],
                          rotation=40 if kind == "parameter" else 0, ha="right" if kind == "parameter" else "center")
            ax.set_ylabel("Model")
            ax.set_xlabel("Parameter settings (* original)" if kind == "parameter" else "Panel omitted (both horizons)")
            for i, family in enumerate(models.family):
                ax.get_yticklabels()[i].set_color(FAMILY_COLORS[family])
                if i and family != models.family.iloc[i - 1]:
                    ax.axhline(i - .5, color="white", linewidth=2)
            centers, labels = [], []
            for component, positions in scenarios.groupby("component", sort=False).groups.items():
                centers.append(np.mean(list(positions)))
                labels.append(COMPONENT_LABELS[component])
                ax.axvline(max(positions) + .5, color="#555555", linewidth=1)
            header = ax.secondary_xaxis("top")
            header.set_xticks(centers, labels=labels)
            header.tick_params(length=0, pad=6, labelsize=10)
            for i in range(len(models)):
                for j in range(len(scenarios)):
                    red, green, blue, _ = cmap(norm(deltas.iloc[i, j]))
                    luminance = .2126 * red + .7152 * green + .0722 * blue
                    ax.text(j, i, f"{ranks.iloc[i, j]:.1f}", ha="center", va="center",
                            color="white" if luminance < .5 else "black", fontsize=7)
        fig.subplots_adjust(left=.09, right=.94, top=.95, bottom=.15, hspace=.6)
        fig.colorbar(im, ax=list(axes), fraction=.015, pad=.015).set_label(
            "Mean-rank change vs original (blue = improved; red = worsened)")
        fig.legend(handles=[Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_LABELS[f]) for f in FAMILY_ORDER],
                   loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(.5, .07))
        fig.text(.5, .043, "Cell numbers: mean task ranks (1 = best). Cell colours: change from the original "
                 "of the same component. Top: 23 parameter settings, including 9 originals. Bottom: all 45 panel omissions.",
                 ha="center", fontsize=9)
        fig.text(.5, .02, "Panel codes: HC = Hermetic Compressors; F = Filters; Me = Mechanisms; "
                 "FM = Fan Motors; CB = Control Boards. Original averages use 10 tasks; panel omissions use 8 tasks.",
                 ha="center", fontsize=9)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def save_overview_figures(means: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    """Generate both complete views and their auditable rank-change table without recalculating component values."""
    output.mkdir(parents=True, exist_ok=True)
    changes = compute_rank_changes(means)
    changes.to_csv(output / "rank_changes.csv", index=False)
    plot_stability_overview(summary, output / "sensitivity_overview")
    plot_all_model_ranks(changes, output / "sensitivity_all_model_ranks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/analysis/framework/sensitivity")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    source = Path(args.input_dir)
    save_overview_figures(pd.read_csv(source / "mean_ranks.csv"),
                          pd.read_csv(source / "sensitivity_summary.csv"), Path(args.output_dir or source))
    print("PDFs gerados: sensitivity_overview.pdf e sensitivity_all_model_ranks.pdf")


if __name__ == "__main__":
    main()
