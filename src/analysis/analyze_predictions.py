import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from src.utils.data_loader import DataLoader, discover_datasets


def load_ground_truth(data_dir: str) -> pd.DataFrame:
    """Carrega todos os targets de teste e anexa a coluna dataset."""
    datasets = discover_datasets(data_dir)
    truth_frames = []

    for dataset_name, paths in datasets.items():
        _, df_test_long, _ = DataLoader.load_dataset(paths["train"], paths["test"])
        df_test_long = df_test_long.copy()
        df_test_long["dataset"] = dataset_name
        truth_frames.append(df_test_long[["dataset", "id", "timestamp", "target"]])

    if not truth_frames:
        raise ValueError(f"Nenhum dataset encontrado em '{data_dir}'.")

    truth_df = pd.concat(truth_frames, ignore_index=True)
    truth_df["id"] = truth_df["id"].astype(str)
    truth_df["timestamp"] = pd.to_datetime(truth_df["timestamp"])
    return truth_df


def evaluate_merged(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas por dataset e série."""
    rows = []

    for (dataset, series_id), group in merged_df.groupby(["dataset", "id"], sort=True):
        y_true = group["target"]
        y_pred = group["q50"]

        rows.append(
            {
                "dataset": dataset,
                "id": series_id,
                "n_points": len(group),
                "mse": float(mean_squared_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            }
        )

    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por dataset."""
    rows = []

    for dataset, group in metrics_df.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "series_count": group["id"].nunique(),
                "mse_mean": group["mse"].mean(),
                "mse_median": group["mse"].median(),
                "mae_mean": group["mae"].mean(),
                "mae_median": group["mae"].median(),
                "mape_mean": group["mape"].mean(),
                "mape_median": group["mape"].median(),
            }
        )

    overall = {
        "dataset": "ALL_DATASETS",
        "series_count": metrics_df["id"].nunique(),
        "mse_mean": metrics_df["mse"].mean(),
        "mse_median": metrics_df["mse"].median(),
        "mae_mean": metrics_df["mae"].mean(),
        "mae_median": metrics_df["mae"].median(),
        "mape_mean": metrics_df["mape"].mean(),
        "mape_median": metrics_df["mape"].median(),
    }
    rows.append(overall)

    return pd.DataFrame(rows)


def evaluate_prediction_file(pred_path: Path, truth_df: pd.DataFrame):
    """Avalia um parquet de previsões contra os targets reais."""
    pred_df = pd.read_parquet(pred_path).copy()

    required_cols = {"dataset", "id", "timestamp", "q50"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        raise ValueError(f"Arquivo {pred_path.name} sem colunas obrigatórias: {sorted(missing_cols)}")

    pred_df["id"] = pred_df["id"].astype(str)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

    merged_df = truth_df.merge(
        pred_df[["dataset", "id", "timestamp", "q50"]],
        on=["dataset", "id", "timestamp"],
        how="inner",
    )

    if merged_df.empty:
        raise ValueError(f"Nenhuma linha casou entre truth e {pred_path.name}.")

    finite_mask = np.isfinite(merged_df["target"]) & np.isfinite(merged_df["q50"])
    skipped_rows = int((~finite_mask).sum())
    if skipped_rows:
        print(
            f"Aviso: {pred_path.name} ignorou {skipped_rows} linhas com target/q50 não finito."
        )
        merged_df = merged_df.loc[finite_mask].copy()

    if merged_df.empty:
        raise ValueError(f"Nenhuma linha finita para avaliar em {pred_path.name}.")

    expected_rows = len(pred_df)
    matched_rows = len(merged_df)
    if matched_rows != expected_rows:
        print(
            f"Aviso: {pred_path.name} teve {matched_rows}/{expected_rows} linhas casadas "
            "com o ground truth."
        )

    metrics_df = evaluate_merged(merged_df)
    agg_df = aggregate_metrics(metrics_df)
    return metrics_df, agg_df


def prediction_files(predictions_dir: str, file_name: str | None):
    """Lista arquivos parquet de previsão a processar."""
    if file_name is not None:
        pred_path = Path(predictions_dir) / file_name
        if not pred_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {pred_path}")
        return [pred_path]

    pred_dir = Path(predictions_dir)
    files = sorted(pred_dir.glob("*_predictions.parquet"))
    if not files:
        raise ValueError(f"Nenhum arquivo *_predictions.parquet encontrado em '{predictions_dir}'.")
    return files


def save_outputs(metrics_df: pd.DataFrame, agg_df: pd.DataFrame, pred_path: Path, output_dir: str):
    """Salva CSVs de métricas por série e agregadas."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = pred_path.stem.replace("_predictions", "")
    metrics_path = output_path / f"{stem}_metrics.csv"
    agg_path = output_path / f"{stem}_metrics_agg.csv"

    metrics_df.to_csv(metrics_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    return metrics_path, agg_path


def main():
    parser = argparse.ArgumentParser(description="Avalia arquivos parquet de previsões.")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions", help="Diretório com os parquets.")
    parser.add_argument("--data-dir", type=str, default="data/datasets", help="Diretório com os datasets.")
    parser.add_argument("--file", type=str, default=None, help="Arquivo específico de previsão dentro de predictions-dir.")
    parser.add_argument("--output-dir", type=str, default="data/analysis/metrics", help="Diretório para salvar métricas.")
    args = parser.parse_args()

    truth_df = load_ground_truth(args.data_dir)
    pred_files = prediction_files(args.predictions_dir, args.file)

    for pred_path in pred_files:
        print(f"{'=' * 70}")
        print(f"Arquivo: {pred_path.name}")
        print(f"{'=' * 70}")

        metrics_df, agg_df = evaluate_prediction_file(pred_path, truth_df)
        metrics_path, agg_path = save_outputs(metrics_df, agg_df, pred_path, args.output_dir)

        print("Resumo:")
        for _, row in agg_df.iterrows():
            print(
                f"  {row['dataset']}: "
                f"MSE={row['mse_mean']:.4f} | "
                f"MAE={row['mae_mean']:.4f} | "
                f"MAPE={row['mape_mean']:.4f}"
            )

        print(f"Salvo: {metrics_path}")
        print(f"Salvo: {agg_path}")
        print()


if __name__ == "__main__":
    main()
