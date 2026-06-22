import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from src.losses.loss import InventoryDemandLoss
from src.utils.data_loader import DataLoader, discover_datasets


DEFAULT_LOSS_CONFIG = {
    "lambda_item": 1.0,
    "lambda_sum": 1.0,
    "lambda_share": 1.0,
    "lambda_alloc": 1.0,
    "lambda_weighted": 1.0,
    "lambda_rel": 1.0,
    "lambda_neg": 0.0,
    "lambda_cap": 1.0,
    "lambda_asym": 1.0,
    "lambda_sparse": 1.0,
    "lambda_delta": 1.0,
    "lambda_tv": 0.0,
    "lambda_robust": 1.0,
    "lambda_int": 0.0,
    "base_kind": "huber",
    "huber_delta": 1.0,
    "rel_eps": 1e-6,
    "sum_relative": True,
    "share_divergence": "js",
    "share_mode": "relu",
    "share_eps": 1e-8,
    "alloc_relative": False,
    "quantile_tau": 0.7,
    "sparse_pos_weight": 2.0,
    "sparse_spike_weight": 1.0,
    "spike_temp": 10.0,
    "delta_kind": "mse",
    "tv_kind": "l1",
    "robust_clip": 10.0,
    "cap_mode": "factor_true_max",
    "cap_value": None,
    "cap_factor": 2.0,
    "int_mode": "cos",
    "components_axis": -1,
    "auto_normalize": False,
}

IGNORED_COMPONENTS = {"int", "neg", "total", "tv"}

PRETTY_MODEL_NAMES = {
    "darts_naive_moving_average": "Moving Average",
    "darts_arima": "ARIMA",
    "darts_prophet": "Prophet",
    "darts_randomforest": "Random Forest",
    "darts_lightgbm": "LightGBM",
    "darts_linear": "Linear Regression",
    "darts_global_naive_aggregate": "Global Naive Aggregate",
    "darts_global_naive_drift": "Global Naive Drift",
    "darts_global_naive_seasonal": "Global Naive Seasonal",
    "chronos-2": "Chronos-2",
    "chronos-2-finetuned-lora": "Chronos-2 Fine-tuned (LoRA)",
    "moirai2-small": "Moirai2-Small",
    "sundial-base-128m": "Sundial Base 128M",
}

MODEL_GROUPS = {
    "darts_naive_moving_average": "industry",
    "darts_arima": "industry",
    "darts_prophet": "industry",
    "darts_linear": "classic_ml",
    "darts_randomforest": "classic_ml",
    "darts_lightgbm": "classic_ml",
    "darts_global_naive_aggregate": "classic_ml",
    "darts_global_naive_drift": "classic_ml",
    "darts_global_naive_seasonal": "classic_ml",
    "chronos-2": "foundation",
    "chronos-2-finetuned-lora": "foundation",
    "moirai2-small": "foundation",
    "sundial-base-128m": "foundation",
}

GROUP_COLORS = {
    "industry": "#4C78A8",
    "classic_ml": "#F58518",
    "foundation": "#54A24B",
}

GROUP_LABELS = {
    "industry": "Indústria",
    "classic_ml": "Clássicos IA",
    "foundation": "Foundation Models",
}

GROUP_ORDER = ["classic_ml", "industry", "foundation"]


def pretty_model_name(model_name: str) -> str:
    return PRETTY_MODEL_NAMES.get(model_name, model_name.replace("darts_", "").replace("_", " ").title())


def model_group(model_name: str) -> str:
    return MODEL_GROUPS.get(model_name, "classic_ml")


def model_color(model_name: str) -> str:
    return GROUP_COLORS[model_group(model_name)]


def load_truth_by_dataset(data_dir: str) -> dict[str, pd.DataFrame]:
    datasets = discover_datasets(data_dir)
    truth = {}
    for dataset_name, paths in datasets.items():
        _, df_test_long, _ = DataLoader.load_dataset(paths["train"], paths["test"])
        df_test_long = df_test_long.copy()
        df_test_long["id"] = df_test_long["id"].astype(str)
        df_test_long["timestamp"] = pd.to_datetime(df_test_long["timestamp"])
        truth[dataset_name] = df_test_long
    return truth


def list_prediction_files(predictions_dir: str, models: list[str] | None) -> list[Path]:
    pred_dir = Path(predictions_dir)
    if models:
        paths = []
        for model_name in models:
            path = pred_dir / f"{model_name}_all_datasets_predictions.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Arquivo de predição não encontrado: {path}")
            paths.append(path)
        return paths

    files = sorted(pred_dir.glob("*_all_datasets_predictions.parquet"))
    if not files:
        raise ValueError(f"Nenhum arquivo de previsão encontrado em '{predictions_dir}'.")
    return files


def model_name_from_path(path: Path) -> str:
    return path.stem.replace("_all_datasets_predictions", "")


def pivot_matrix(df: pd.DataFrame, value_col: str, ids_order: list[str] | None = None) -> pd.DataFrame:
    wide = df.pivot(index="timestamp", columns="id", values=value_col).sort_index()
    if ids_order is not None:
        wide = wide.reindex(columns=ids_order)
    return wide


def aligned_tensors(df_true: pd.DataFrame, df_pred: pd.DataFrame):
    merged = df_true.merge(
        df_pred[["id", "timestamp", "q50"]],
        on=["id", "timestamp"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Nenhuma linha casou entre truth e prediction.")

    ids_order = sorted(merged["id"].unique())
    true_wide = pivot_matrix(merged, "target", ids_order=ids_order)
    pred_wide = pivot_matrix(merged, "q50", ids_order=ids_order)

    if true_wide.shape != pred_wide.shape:
        raise ValueError(f"Shapes desalinhados: {true_wide.shape} vs {pred_wide.shape}")

    y = torch.tensor(true_wide.to_numpy(), dtype=torch.float32).unsqueeze(0)
    y_hat = torch.tensor(pred_wide.to_numpy(), dtype=torch.float32).unsqueeze(0)
    return y_hat, y, len(merged), len(ids_order), true_wide.shape[0]


def compute_loss_terms(y_hat: torch.Tensor, y: torch.Tensor, loss_config: dict) -> dict[str, float]:
    loss_fn = InventoryDemandLoss(**loss_config)
    yh, yt = loss_fn._to_btd(y_hat, y)

    if loss_fn.auto_normalize:
        yh, yt, _ = loss_fn._normalize(yh, yt)

    term_fns = {
        "item": lambda: loss_fn._term_item(yh, yt),
        "sum": lambda: loss_fn._term_sum(yh, yt),
        "share": lambda: loss_fn._term_share(yh, yt),
        "alloc": lambda: loss_fn._term_alloc(yh, yt),
        "weighted_item": lambda: loss_fn._term_weighted_item(yh, yt),
        "rel": lambda: loss_fn._term_relative(yh, yt),
        "neg": lambda: loss_fn._term_neg(yh),
        "cap": lambda: loss_fn._term_cap(yh, yt),
        "asym": lambda: loss_fn._term_asym(yh, yt),
        "sparse": lambda: loss_fn._term_sparse(yh, yt),
        "delta": lambda: loss_fn._term_delta(yh, yt) if yh.shape[1] >= 2 else torch.zeros((), device=yh.device, dtype=yh.dtype),
        "tv": lambda: loss_fn._term_tv(yh) if yh.shape[1] >= 2 else torch.zeros((), device=yh.device, dtype=yh.dtype),
        "robust": lambda: loss_fn._term_robust(yh, yt),
        "int": lambda: loss_fn._term_int(yh),
    }

    terms = {name: float(fn().detach().cpu().item()) for name, fn in term_fns.items()}
    total_loss = loss_fn(y_hat, y)
    terms["total"] = float(total_loss.detach().cpu().item())
    return {"total": terms["total"], **{k: v for k, v in terms.items() if k != "total"}}


def evaluate_models(truth_by_dataset: dict[str, pd.DataFrame], prediction_files: list[Path], loss_config: dict):
    rows = []

    for pred_path in prediction_files:
        model_name = model_name_from_path(pred_path)
        pred_df = pd.read_parquet(pred_path)
        pred_df = pred_df.copy()
        pred_df["id"] = pred_df["id"].astype(str)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        for dataset_name, df_true in truth_by_dataset.items():
            df_pred = pred_df[pred_df["dataset"] == dataset_name].copy()
            if df_pred.empty:
                continue

            y_hat, y, matched_rows, n_series, horizon = aligned_tensors(df_true, df_pred)
            terms = compute_loss_terms(y_hat, y, loss_config)

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "matched_rows": matched_rows,
                "n_series": n_series,
                "horizon": horizon,
            }
            row.update(terms)
            rows.append(row)

    if not rows:
        raise ValueError("Nenhuma combinação válida de dataset e modelo foi avaliada.")

    return pd.DataFrame(rows)


def active_components(loss_results: pd.DataFrame) -> list[str]:
    reserved = {"dataset", "model", "matched_rows", "n_series", "horizon"}
    return [col for col in loss_results.columns if col not in reserved and col not in IGNORED_COMPONENTS]


def compute_winners(loss_results: pd.DataFrame, tol: float = 1e-8):
    components = active_components(loss_results)
    winner_rows = []
    count_rows = []

    for component in components:
        counts = {}
        for dataset_name, group in loss_results.groupby("dataset", sort=True):
            min_value = group[component].min()
            winners = group[np.isclose(group[component], min_value, atol=tol, rtol=tol)]

            for _, row in winners.iterrows():
                winner_rows.append(
                    {
                        "dataset": dataset_name,
                        "component": component,
                        "model": row["model"],
                        "value": row[component],
                    }
                )
                counts[row["model"]] = counts.get(row["model"], 0) + 1

        for model, wins in sorted(counts.items()):
            count_rows.append({"component": component, "model": model, "wins": wins})

    winners_df = pd.DataFrame(winner_rows)
    counts_df = pd.DataFrame(count_rows)
    models = sorted(loss_results["model"].unique())
    if counts_df.empty:
        summary_df = pd.DataFrame({"model": models})
        for component in components:
            summary_df[component] = 0
        summary_df["total_wins"] = 0
        return winners_df, counts_df, summary_df

    summary_df = (
        counts_df.pivot(index="model", columns="component", values="wins")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    summary_df = (
        pd.DataFrame({"model": models})
        .merge(summary_df, on="model", how="left")
        .fillna(0)
    )
    for component in components:
        if component not in summary_df.columns:
            summary_df[component] = 0
        summary_df[component] = summary_df[component].astype(int)
    summary_df["total_wins"] = summary_df.drop(columns=["model"]).sum(axis=1)
    summary_df = summary_df.sort_values(["total_wins", "model"], ascending=[False, True])
    return winners_df, counts_df, summary_df


def save_outputs(loss_results: pd.DataFrame, winners_df: pd.DataFrame, counts_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loss_path = output_path / "loss_component_values.csv"
    winners_path = output_path / "loss_component_winners.csv"
    counts_path = output_path / "loss_component_win_counts_long.csv"
    summary_path = output_path / "loss_component_win_counts_summary.csv"

    loss_results.to_csv(loss_path, index=False)
    winners_df.to_csv(winners_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return loss_path, winners_path, counts_path, summary_path


def prepare_summary_table(summary_df: pd.DataFrame):
    component_cols = [col for col in summary_df.columns if col not in {"model", "total_wins"}]
    ordered_cols = component_cols + (["total_wins"] if "total_wins" in summary_df.columns else [])

    plot_df = summary_df.copy()
    plot_df["pretty_model"] = plot_df["model"].map(pretty_model_name)

    if "total_wins" in plot_df.columns:
        plot_df = plot_df.sort_values("total_wins", ascending=False)

    return plot_df[["model", "pretty_model", *ordered_cols]], ordered_cols


def save_summary_figure(summary_df: pd.DataFrame, output_path: str):
    plot_df, value_cols = prepare_summary_table(summary_df)
    values = plot_df[value_cols].to_numpy(dtype=float)

    fig_width = max(10, 1.3 * len(value_cols) + 6)
    fig_height = max(4, 0.55 * len(plot_df) + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["pretty_model"])

    for tick_label, model_name in zip(ax.get_yticklabels(), plot_df["model"]):
        tick_label.set_color(model_color(model_name))

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, int(values[i, j]), ha="center", va="center", color="black", fontsize=9)

    ax.set_title("Vitórias por Componente da Loss")
    ax.set_xlabel("Componentes")
    ax.set_ylabel("Modelos")

    legend_handles = [
        Patch(facecolor=GROUP_COLORS[group], label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Conta vitórias por componente da InventoryDemandLoss.")
    parser.add_argument("--data-dir", type=str, default="data/datasets", help="Diretório com os datasets reais.")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions", help="Diretório com os parquets de predição.")
    parser.add_argument("--models", nargs="+", default=None, help="Modelos específicos a avaliar.")
    parser.add_argument("--output-dir", type=str, default="data/analysis/loss_wins", help="Diretório de saída.")
    parser.add_argument("--plots-dir", type=str, default="data/analysis/plots", help="Diretório para salvar a figura resumo.")
    parser.add_argument("--all-components", action="store_true", help="Ativa todos os componentes da loss com peso 1.0.")
    args = parser.parse_args()

    loss_config = DEFAULT_LOSS_CONFIG.copy()
    if args.all_components:
        for key in list(loss_config):
            if key.startswith("lambda_"):
                loss_config[key] = 1.0

    truth_by_dataset = load_truth_by_dataset(args.data_dir)
    prediction_files = list_prediction_files(args.predictions_dir, args.models)
    loss_results = evaluate_models(truth_by_dataset, prediction_files, loss_config)
    winners_df, counts_df, summary_df = compute_winners(loss_results)
    paths = save_outputs(loss_results, winners_df, counts_df, summary_df, args.output_dir)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure_path = save_summary_figure(summary_df, str(plots_dir / "loss_component_win_counts_summary.png"))

    print("Arquivos salvos:")
    for path in paths:
        print(f"  {path}")
    print(f"  {figure_path}")

    print("\nResumo de vitórias:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
