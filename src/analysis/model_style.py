"""Shared model families, colors and display names for every benchmark analysis.

The visual reference is framework/ranks/inventory_component_rank_heatmap.pdf.
"""

FAMILY_ORDER = ("naive", "statistical", "intermittent", "machine_learning", "foundation")
FAMILY_LABELS = {
    "naive": "Naive baselines",
    "statistical": "Statistical models",
    "intermittent": "Intermittent-demand methods",
    "machine_learning": "Machine learning models",
    "foundation": "Foundation models",
}
FAMILY_COLORS = {
    "naive": "#7F7F7F",
    "statistical": "#4C78A8",
    "intermittent": "#B279A2",
    "machine_learning": "#F58518",
    "foundation": "#54A24B",
}

PRETTY_MODEL_NAMES = {
    "TiRex": "TiRex",
    "chronos-2": "Chronos-2",
    "chronos-2-finetuned-lora": "Chronos-2 Fine-tuned (LoRA)",
    "darts_arima": "ARIMA",
    "darts_global_naive_aggregate": "Naive Aggregate",
    "darts_global_naive_drift": "Naive Drift",
    "darts_global_naive_seasonal": "Naive Seasonal",
    "darts_lightgbm": "LightGBM",
    "darts_lightgbm_topdown": "LightGBM Top-Down",
    "darts_linear": "Linear Regression",
    "darts_linear_topdown": "Linear Regression Top-Down",
    "darts_naive_moving_average": "Moving Average",
    "darts_prophet": "Prophet",
    "darts_randomforest": "Random Forest",
    "darts_randomforest_topdown": "Random Forest Top-Down",
    "darts_xgboost": "XGBoost",
    "darts_xgboost_topdown": "XGBoost Top-Down",
    "granite-timeseries-ttm-r2": "Granite TTM",
    "moirai2-small": "Moirai2-Small",
    "statsforecast_croston": "Croston",
    "statsforecast_sba": "SBA",
    "statsforecast_tsb": "TSB",
    "sundial-base-128m": "Sundial Base 128M",
    "tabpfn-ts-local": "TabPFN-TS",
    "timer-base-84m": "Timer Base 84M",
    "timesfm-2.5-200m-pytorch": "TimesFM 2.5",
}


def pretty_model_name(model_name: str) -> str:
    """Return one consistent display name for a model across figures and tables."""
    if model_name in PRETTY_MODEL_NAMES:
        return PRETTY_MODEL_NAMES[model_name]
    normalized = model_name.lower()
    for token, label in (
        ("timesfm", "TimesFM"), ("tabpfn", "TabPFN-TS"), ("timer", "Timer"),
        ("tirex", "TiRex"), ("granite", "Granite TTM"), ("moirai", "Moirai"),
        ("sundial", "Sundial"), ("chronos", "Chronos-2"),
    ):
        if token in normalized:
            return label
    return model_name.replace("darts_", "").replace("_", " ").title()


def model_family(model_name: str) -> str:
    """Classify models using the five families of the reference heatmap."""
    normalized = model_name.lower()
    if normalized.startswith("darts_global_naive_"):
        return "naive"
    if normalized in {"statsforecast_croston", "statsforecast_sba", "statsforecast_tsb"}:
        return "intermittent"
    if any(token in normalized for token in ("moving_average", "arima", "prophet")):
        return "statistical"
    if any(token in normalized for token in (
        "chronos", "moirai", "sundial", "timesfm", "tabpfn", "tirex", "granite", "timer",
    )):
        return "foundation"
    return "machine_learning"


def model_color(model_name: str) -> str:
    """Return the reference color of a model's family."""
    return FAMILY_COLORS[model_family(model_name)]


MODEL_MARKERS = {
    "statsforecast_croston": "o",
    "statsforecast_sba": "s",
    "statsforecast_tsb": "^",
    "darts_naive_moving_average": "o",
    "darts_arima": "s",
    "darts_prophet": "^",
    "darts_randomforest": "D",
    "darts_lightgbm": "P",
    "darts_linear": "X",
    "darts_xgboost": "o",
    "darts_global_naive_aggregate": "v",
    "darts_global_naive_drift": "<",
    "darts_global_naive_seasonal": ">",
    "chronos-2": "h",
    "chronos-2-finetuned-lora": "H",
    "moirai2-small": "*",
    "sundial-base-128m": "8",
    "granite-timeseries-ttm-r2": "s",
    "tabpfn-ts-local": "P",
    "timer-base-84m": "v",
    "timesfm-2.5-200m-pytorch": "^",
    "TiRex": "D",
}


def model_marker(model_name: str) -> str:
    """Distinguish models that share a family color in line plots."""
    return MODEL_MARKERS.get(model_name, "o")
