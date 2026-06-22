import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.data_loader import DataLoader, discover_datasets


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total == 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * values) / (n * total)) - ((n + 1.0) / n))


def longest_zero_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if value == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if mean == 0.0:
        return 0.0
    return float(np.std(values) / mean)


def infer_panel_name(dataset_name: str) -> str:
    if "_h" in dataset_name:
        return dataset_name.rsplit("_h", 1)[0]
    return dataset_name


def infer_horizon_name(dataset_name: str, fallback_horizon: int) -> str:
    if "_h" in dataset_name:
        suffix = dataset_name.rsplit("_h", 1)[1]
        if suffix.isdigit():
            return f"h{suffix}"
    return f"h{fallback_horizon}"


def series_statistics(dataset_name: str, split_name: str, df_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_id, group in df_long.groupby("id", sort=True):
        values = group.sort_values("timestamp")["target"].astype(float).to_numpy()
        positive = values[values > 0]
        total = float(values.sum())
        rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "id": str(series_id),
                "n_points": int(values.size),
                "total_demand": total,
                "mean_demand": float(values.mean()) if values.size else 0.0,
                "median_demand": float(np.median(values)) if values.size else 0.0,
                "std_demand": float(values.std()) if values.size else 0.0,
                "cv_demand": coefficient_of_variation(values),
                "max_demand": float(values.max()) if values.size else 0.0,
                "min_demand": float(values.min()) if values.size else 0.0,
                "zero_count": int((values == 0).sum()),
                "zero_rate": float((values == 0).mean()) if values.size else 0.0,
                "positive_count": int((values > 0).sum()),
                "positive_rate": float((values > 0).mean()) if values.size else 0.0,
                "positive_mean_demand": float(positive.mean()) if positive.size else 0.0,
                "longest_zero_run": longest_zero_run(values),
            }
        )
    return pd.DataFrame(rows)


def dataset_statistics(dataset_name: str, train_long: pd.DataFrame, test_long: pd.DataFrame, horizon: int) -> dict:
    full_long = pd.concat(
        [
            train_long.assign(split="train"),
            test_long.assign(split="test"),
        ],
        ignore_index=True,
    )
    series_totals = full_long.groupby("id")["target"].sum().astype(float)
    sorted_totals = series_totals.sort_values(ascending=False)
    total_demand = float(series_totals.sum())
    top_10_count = min(10, len(sorted_totals))
    top_20pct_count = max(1, int(np.ceil(0.2 * len(sorted_totals)))) if len(sorted_totals) else 0

    train_wide = train_long.pivot(index="timestamp", columns="id", values="target")
    test_wide = test_long.pivot(index="timestamp", columns="id", values="target")
    full_wide = full_long.pivot(index="timestamp", columns="id", values="target")

    values = full_long["target"].astype(float).to_numpy()
    train_values = train_long["target"].astype(float).to_numpy()
    test_values = test_long["target"].astype(float).to_numpy()

    return {
        "dataset": dataset_name,
        "panel": infer_panel_name(dataset_name),
        "horizon_label": infer_horizon_name(dataset_name, horizon),
        "n_skus": int(full_long["id"].nunique()),
        "train_length": int(train_wide.shape[0]),
        "test_length": int(test_wide.shape[0]),
        "horizon": int(horizon),
        "total_timesteps": int(full_wide.shape[0]),
        "train_start": str(train_long["timestamp"].min().date()),
        "train_end": str(train_long["timestamp"].max().date()),
        "test_start": str(test_long["timestamp"].min().date()),
        "test_end": str(test_long["timestamp"].max().date()),
        "total_observations": int(values.size),
        "total_demand": float(total_demand),
        "train_total_demand": float(train_values.sum()),
        "test_total_demand": float(test_values.sum()),
        "mean_demand": float(values.mean()) if values.size else 0.0,
        "median_demand": float(np.median(values)) if values.size else 0.0,
        "std_demand": float(values.std()) if values.size else 0.0,
        "cv_demand": coefficient_of_variation(values),
        "zero_count": int((values == 0).sum()),
        "zero_rate": float((values == 0).mean()) if values.size else 0.0,
        "train_zero_rate": float((train_values == 0).mean()) if train_values.size else 0.0,
        "test_zero_rate": float((test_values == 0).mean()) if test_values.size else 0.0,
        "positive_rate": float((values > 0).mean()) if values.size else 0.0,
        "sku_total_gini": gini(series_totals.to_numpy()),
        "top_1_sku_share": safe_divide(float(sorted_totals.iloc[0]) if len(sorted_totals) else 0.0, total_demand),
        "top_5_sku_share": safe_divide(float(sorted_totals.iloc[:5].sum()), total_demand),
        "top_10_sku_share": safe_divide(float(sorted_totals.iloc[:top_10_count].sum()), total_demand),
        "top_20pct_sku_share": safe_divide(float(sorted_totals.iloc[:top_20pct_count].sum()), total_demand),
        "zero_demand_skus": int((series_totals == 0).sum()),
        "single_positive_skus": int((full_long.groupby("id")["target"].apply(lambda s: (s > 0).sum()) == 1).sum()),
    }


def describe_datasets(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = discover_datasets(data_dir)
    if not datasets:
        raise ValueError(f"Nenhum dataset encontrado em '{data_dir}'.")

    dataset_rows = []
    sku_frames = []

    for dataset_name, paths in datasets.items():
        train_long, test_long, horizon = DataLoader.load_dataset(paths["train"], paths["test"])
        train_long = train_long.copy()
        test_long = test_long.copy()
        train_long["id"] = train_long["id"].astype(str)
        test_long["id"] = test_long["id"].astype(str)

        dataset_rows.append(dataset_statistics(dataset_name, train_long, test_long, horizon))
        sku_frames.append(series_statistics(dataset_name, "train", train_long))
        sku_frames.append(series_statistics(dataset_name, "test", test_long))
        sku_frames.append(series_statistics(dataset_name, "full", pd.concat([train_long, test_long], ignore_index=True)))

    dataset_summary = pd.DataFrame(dataset_rows)
    sku_summary = pd.concat(sku_frames, ignore_index=True)
    panel_summary = (
        dataset_summary.groupby("panel", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            horizons=("horizon_label", lambda s: ",".join(sorted(s.unique()))),
            max_skus=("n_skus", "max"),
            total_demand=("total_demand", "sum"),
            mean_zero_rate=("zero_rate", "mean"),
            mean_sku_total_gini=("sku_total_gini", "mean"),
            mean_top_10_sku_share=("top_10_sku_share", "mean"),
        )
        .sort_values("panel")
    )

    return dataset_summary, sku_summary, panel_summary


def save_outputs(
    dataset_summary: pd.DataFrame,
    sku_summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
    output_dir: str,
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path = output_path / "dataset_summary.csv"
    sku_path = output_path / "dataset_sku_summary.csv"
    panel_path = output_path / "dataset_panel_summary.csv"

    dataset_summary.to_csv(dataset_path, index=False)
    sku_summary.to_csv(sku_path, index=False)
    panel_summary.to_csv(panel_path, index=False)

    return dataset_path, sku_path, panel_path


def print_summary(dataset_summary: pd.DataFrame, panel_summary: pd.DataFrame) -> None:
    cols = [
        "dataset",
        "n_skus",
        "train_length",
        "test_length",
        "total_demand",
        "zero_rate",
        "sku_total_gini",
        "top_10_sku_share",
    ]
    print("\nResumo por dataset:")
    print(dataset_summary[cols].to_string(index=False))

    print("\nResumo por painel:")
    print(panel_summary.to_string(index=False))

    print("\nResumo global:")
    print(f"  Datasets: {dataset_summary['dataset'].nunique()}")
    print(f"  Painéis: {dataset_summary['panel'].nunique()}")
    print(f"  SKUs máximos em um painel: {dataset_summary['n_skus'].max()}")
    print(f"  Observações totais: {dataset_summary['total_observations'].sum()}")
    print(f"  Demanda total: {dataset_summary['total_demand'].sum():.0f}")
    print(f"  Zero rate médio: {dataset_summary['zero_rate'].mean():.2%}")
    print(f"  Gini médio por demanda SKU: {dataset_summary['sku_total_gini'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Descreve os datasets reais de demanda.")
    parser.add_argument("--data-dir", type=str, default="data/datasets", help="Diretório com *_train.parquet/*_test.parquet.")
    parser.add_argument("--output-dir", type=str, default="data/analysis/dataset", help="Diretório para salvar os CSVs.")
    args = parser.parse_args()

    dataset_summary, sku_summary, panel_summary = describe_datasets(args.data_dir)
    paths = save_outputs(dataset_summary, sku_summary, panel_summary, args.output_dir)
    print_summary(dataset_summary, panel_summary)

    print("\nArquivos salvos:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
