"""
Script simples para modelos clássicos do Darts.

Modelos suportados:
- LinearRegression
- RandomForest
- LightGBM
- XGBoost
- GlobalNaiveAggregate
- GlobalNaiveDrift
- GlobalNaiveSeasonal

Usage:
    python -m src.models.darts_classic --model randomforest
    python -m src.models.darts_classic --model global_naive_aggregate --datasets Mecanismos_h6
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from darts import TimeSeries
from darts.models import (
    LinearRegressionModel,
    RandomForestModel,
    LightGBMModel,
    XGBModel,
    GlobalNaiveAggregate,
    GlobalNaiveDrift,
    GlobalNaiveSeasonal,
)

from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


def create_model(
    model_name: str,
    lags: int = None,
    use_aggregate_covariate: bool = False,
    output_chunk_length: int = 1
):
    """Cria modelo Darts baseado no nome (sem hiperparâmetros fixos para gridsearch)."""

    use_past_covariates = use_aggregate_covariate and model_name in {'linear', 'randomforest', 'lightgbm', 'xgboost'}

    models = {
        'linear': LinearRegressionModel(
            lags=lags,
            lags_past_covariates=lags if use_past_covariates else None,
            output_chunk_length=output_chunk_length
        ),
        'randomforest': RandomForestModel(
            lags=lags,
            lags_past_covariates=lags if use_past_covariates else None,
            output_chunk_length=output_chunk_length,
            random_state=42
        ),
        'lightgbm': LightGBMModel(
            lags=lags,
            lags_past_covariates=lags if use_past_covariates else None,
            output_chunk_length=output_chunk_length,
            random_state=42,
            verbose=-1
        ),
        'xgboost': XGBModel(
            lags=lags,
            lags_past_covariates=lags if use_past_covariates else None,
            output_chunk_length=output_chunk_length,
            random_state=42,
            verbosity=0
        ),
        'global_naive_aggregate': GlobalNaiveAggregate(
            input_chunk_length=lags,
            output_chunk_length=output_chunk_length,
            agg_fn='mean'
        ),
        'global_naive_drift': GlobalNaiveDrift(
            input_chunk_length=lags,
            output_chunk_length=output_chunk_length
        ),
        'global_naive_seasonal': GlobalNaiveSeasonal(
            input_chunk_length=lags,
            output_chunk_length=output_chunk_length
        )
    }

    if model_name not in models:
        raise ValueError(f"Modelo '{model_name}' não suportado. Escolha: {list(models.keys())}")

    return models[model_name]


def get_hyperparameter_grid(model_name: str):
    """Define grid de hiperparâmetros para cada modelo."""

    grids = {
        'linear': {
            'lags': [3, 6, 12]
        },
        'randomforest': {
            'lags': [3, 6, 12],
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        },
        'lightgbm': {
            'lags': [3, 6, 12],
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        },
        'xgboost': {
            'lags': [3, 6, 12],
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        },
        'global_naive_aggregate': {
            'lags': [3, 6, 12]
        },
        'global_naive_drift': {
            'lags': [3, 6, 12]
        },
        'global_naive_seasonal': {
            'lags': [3, 6, 12]
        }
    }

    return grids.get(model_name, {})


def df_to_multivariate_timeseries(df: pd.DataFrame) -> TimeSeries:
    """
    Converte DataFrame long para uma única TimeSeries multivariada.

    Cada `id` vira um componente da série multivariada.
    """
    df_wide = (
        df.pivot(index='timestamp', columns='id', values='target')
        .sort_index()
        .reset_index()
    )

    value_cols = [col for col in df_wide.columns if col != 'timestamp']
    return TimeSeries.from_dataframe(
        df_wide,
        time_col='timestamp',
        value_cols=value_cols,
        freq='MS'
    )


def df_to_aggregate_timeseries(df: pd.DataFrame) -> TimeSeries:
    """Converte DataFrame long para uma única série agregada total."""
    df_total = (
        df.groupby('timestamp', as_index=False)['target']
        .sum()
        .sort_values('timestamp')
    )
    return TimeSeries.from_dataframe(
        df_total,
        time_col='timestamp',
        value_cols='target',
        freq='MS'
    )


def compute_recent_shares(df_train_long: pd.DataFrame, window: int = 12) -> pd.Series:
    """Calcula shares recentes para decomposição top-down."""
    df_sorted = df_train_long.sort_values(['timestamp', 'id']).copy()
    unique_timestamps = sorted(df_sorted['timestamp'].unique())
    effective_window = min(window, len(unique_timestamps))
    recent_timestamps = unique_timestamps[-effective_window:]

    df_recent = df_sorted[df_sorted['timestamp'].isin(recent_timestamps)].copy()
    totals = df_recent.groupby('timestamp')['target'].sum().rename('total')
    df_recent = df_recent.merge(totals, on='timestamp', how='left')
    df_recent['share'] = np.where(df_recent['total'] != 0, df_recent['target'] / df_recent['total'], 0.0)

    shares = df_recent.groupby('id')['share'].mean().fillna(0.0)
    if shares.sum() <= 0:
        counts = df_recent['id'].value_counts().sort_index()
        shares = pd.Series(1.0 / len(counts), index=counts.index)
    else:
        shares = shares / shares.sum()

    shares.index = shares.index.astype(str)
    return shares


def decompose_forecast(aggregate_forecast: TimeSeries, shares: pd.Series) -> pd.DataFrame:
    """Decompõe previsão agregada em previsões por série individual."""
    forecast_values = aggregate_forecast.values().flatten()
    timestamps = pd.to_datetime(aggregate_forecast.time_index)
    pred_frames = []

    for series_id, share in shares.items():
        pred_frames.append(
            pd.DataFrame(
                {
                    'timestamp': timestamps,
                    'q50': forecast_values * float(share),
                    'id': str(series_id),
                }
            )
        )

    return pd.concat(pred_frames, ignore_index=True)


def split_train_val_multivariate(series: TimeSeries, horizon: int):
    """Divide uma TimeSeries multivariada em treino e validação."""
    return series[:-horizon], series[-horizon:]


def split_train_val_univariate(series: TimeSeries, horizon: int):
    """Divide uma TimeSeries univariada em treino e validação."""
    return series[:-horizon], series[-horizon:]


def validation_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric == "mape":
        denom = np.where(y_true == 0, np.nan, y_true)
        return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    raise ValueError(f"Métrica de validação não suportada: {metric}")


def train_with_gridsearch(model_name: str, train_series: TimeSeries, val_series: TimeSeries,
                          param_grid: dict, horizon: int, validation_metric: str,
                          aggregate_covariate: TimeSeries | None = None):
    """
    Treina modelos usando uma série multivariada com validação.
    """
    from itertools import product

    print("  Grid search multivariado...")
    print(f"  Grid: {param_grid}")

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_score = float('inf')
    best_params = None
    best_model = None

    total_combinations = np.prod([len(v) for v in param_values])
    print(f"  Testando {int(total_combinations)} combinações...")

    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        current_model = create_model(
            model_name,
            lags=params.get('lags', 6),
            use_aggregate_covariate=aggregate_covariate is not None,
            output_chunk_length=horizon
        )
        fit_kwargs = {}
        predict_kwargs = {}
        if aggregate_covariate is not None:
            fit_kwargs['past_covariates'] = aggregate_covariate
            predict_kwargs['past_covariates'] = aggregate_covariate

        current_model.fit(train_series, **fit_kwargs)

        pred = current_model.predict(n=horizon, series=train_series, **predict_kwargs)
        score = validation_score(val_series.values(), pred.values(), validation_metric)

        if score < best_score:
            best_score = score
            best_params = params
            best_model = current_model
            print(
                f"    [{i+1}/{int(total_combinations)}] Nova melhor: "
                f"{params} -> {validation_metric.upper()}={score:.4f}"
            )
        else:
            print(
                f"    [{i+1}/{int(total_combinations)}] "
                f"{params} -> {validation_metric.upper()}={score:.4f}"
            )

    print(f"  ✓ Melhores parâmetros: {best_params}")
    print(f"  ✓ Melhor {validation_metric.upper()} (validação): {best_score:.4f}")

    return best_model, best_params


def predict_multivariate_series(model, train_series: TimeSeries, horizon: int,
                                aggregate_covariate: TimeSeries | None = None) -> pd.DataFrame:
    """
    Gera previsões de uma série multivariada no formato longo do pipeline.
    """
    predict_kwargs = {}
    if aggregate_covariate is not None:
        predict_kwargs['past_covariates'] = aggregate_covariate

    forecast = model.predict(n=horizon, series=train_series, **predict_kwargs)
    forecast_df = forecast.to_dataframe().reset_index()
    timestamp_col = forecast_df.columns[0]
    forecast_df = forecast_df.rename(columns={timestamp_col: 'timestamp'})
    pred_long = forecast_df.melt(id_vars=['timestamp'], var_name='id', value_name='q50')
    pred_long['timestamp'] = pd.to_datetime(pred_long['timestamp'])
    pred_long['id'] = pred_long['id'].astype(str)
    return pred_long[['id', 'timestamp', 'q50']]


def predict_top_down(model, train_series: TimeSeries, shares: pd.Series, horizon: int) -> pd.DataFrame:
    """Gera previsão agregada e decompõe por shares."""
    forecast = model.predict(n=horizon, series=train_series)
    return decompose_forecast(forecast, shares)


def supports_aggregate_covariate(model_name: str) -> bool:
    return model_name in {'linear', 'randomforest', 'lightgbm', 'xgboost'}


def main(model_name, data_dir, output_dir, lags, datasets_filter, validation_metric, top_down, use_aggregate_covariate):

    # ==================== DESCOBRIR DATASETS ====================
    datasets = discover_datasets(data_dir)

    # Filtrar datasets se especificado
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets")
    print(f"Modelo: {model_name}")
    if top_down:
        print("Usando GridSearch no agregado com decomposição top-down\n")
    elif use_aggregate_covariate:
        if not supports_aggregate_covariate(model_name):
            print("Aviso: este modelo não usa a soma total como covariável. A flag será ignorada.\n")
        else:
            print("Usando GridSearch multivariado com soma total como past covariate\n")
    else:
        print("Usando GridSearch multivariado\n")

    # Listas para acumular resultados
    all_predictions = []  # Predições no conjunto de teste
    all_best_params = []  # Melhores hiperparâmetros por dataset

    # ==================== LOOP POR DATASETS ====================
    for dataset_name, paths in datasets.items():
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        try:
            # Carregar dados
            df_train_long, df_test_long, horizon = DataLoader.load_dataset(
                paths['train'],
                paths['test']
            )

            info = DataLoader.get_dataset_info(df_train_long, df_test_long, horizon)
            print(f"Séries: {info['n_series']} | Train: {info['train_length']} | Horizon: {horizon}")

            # Verificar se há dados suficientes (max lags do grid=12 + validação)
            min_required = 12 + horizon + 1
            if info['train_length'] < min_required:
                print(f"⚠️  PULADO: Train muito curto ({info['train_length']} < {min_required})")
                continue

            if top_down:
                print("Convertendo para série agregada...")
                train_series_full = df_to_aggregate_timeseries(df_train_long)
                shares = compute_recent_shares(df_train_long)
                aggregate_covariate_full = None
            else:
                print("Convertendo para série multivariada...")
                train_series_full = df_to_multivariate_timeseries(df_train_long)
                if use_aggregate_covariate and supports_aggregate_covariate(model_name):
                    print("Construindo covariável com soma total das séries...")
                    aggregate_covariate_full = df_to_aggregate_timeseries(df_train_long)
                else:
                    aggregate_covariate_full = None

            print("Dividindo train/val...")
            train_series, val_series = split_train_val_multivariate(train_series_full, horizon)
            if aggregate_covariate_full is not None:
                aggregate_cov_train, _ = split_train_val_univariate(aggregate_covariate_full, horizon)
            else:
                aggregate_cov_train = None

            print("Grid Search com validação...")
            param_grid = get_hyperparameter_grid(model_name)
            best_model, best_params = train_with_gridsearch(
                model_name,
                train_series,
                val_series,
                param_grid,
                horizon,
                validation_metric,
                aggregate_covariate=aggregate_cov_train,
            )

            best_params['dataset'] = dataset_name
            all_best_params.append(best_params)

            if top_down:
                print("Retreinando no agregado e decompondo para predição no teste...")
            else:
                print("Retreinando com dados completos para predição no teste...")
            final_model = create_model(
                model_name,
                lags=best_params.get('lags'),
                use_aggregate_covariate=aggregate_covariate_full is not None,
                output_chunk_length=horizon
            )
            fit_kwargs = {}
            if aggregate_covariate_full is not None:
                fit_kwargs['past_covariates'] = aggregate_covariate_full
            final_model.fit(train_series_full, **fit_kwargs)
            if top_down:
                pred_test_df = predict_top_down(final_model, train_series_full, shares, horizon)
            else:
                pred_test_df = predict_multivariate_series(
                    final_model, train_series_full, horizon, aggregate_covariate=aggregate_covariate_full
                )

            pred_test_df['dataset'] = dataset_name

            # Acumular resultados
            all_predictions.append(pred_test_df)

            print(f"✓ Dataset processado\n")

        except Exception as e:
            print(f"✗ Erro: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    # ==================== SALVAR TUDO ====================
    if all_predictions:
        model_suffix = "_topdown" if top_down else ""
        model_label = f"darts_{model_name}{model_suffix}"

        # Concatenar resultados de TESTE
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_best_params = pd.DataFrame(all_best_params)

        # Reordenar colunas
        pred_cols = ['dataset', 'id', 'timestamp', 'q50']
        final_predictions = final_predictions[pred_cols]

        # Salvar resultados de TESTE
        pred_path = save_predictions(final_predictions, 'all_datasets', model_label, output_dir)

        # Salvar melhores hiperparâmetros
        import json
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        params_path = output_dir_path / f"{model_label}_best_hyperparameters.json"

        # Converter para dict e salvar
        params_dict = final_best_params.to_dict('records')
        with open(params_path, 'w') as f:
            json.dump(params_dict, f, indent=2)

        print(f"\n{'='*70}")
        print("SALVAMENTO FINAL")
        print(f"{'='*70}")
        print(f"✓ Previsões (teste): {pred_path}")
        print(f"✓ Melhores hiperparâmetros: {params_path}")
        print(f"\nTotal de datasets: {len(all_predictions)}")
    else:
        print("\n✗ Nenhum dataset processado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Darts Classic Models Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=[
            'linear',
            'randomforest',
            'lightgbm',
            'xgboost',
            'global_naive_aggregate',
            'global_naive_drift',
            'global_naive_seasonal',
            'all'
        ],
        help='Modelo a usar (all = todos os modelos)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/datasets',
        help='Diretório dos datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/predictions',
        help='Diretório de saída'
    )
    parser.add_argument(
        '--lags',
        type=int,
        default=6,
        help='Número de lags (valores passados) para usar'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Datasets específicos (vazio = todos)'
    )
    parser.add_argument(
        '--validation-metric',
        type=str,
        default='mae',
        choices=['mae', 'mse', 'rmse', 'mape'],
        help='Métrica usada apenas no grid search de validação'
    )
    parser.add_argument(
        '--top-down',
        action='store_true',
        help='Treina na série agregada do dataset e decompõe a previsão para os ids'
    )
    parser.add_argument(
        '--use-aggregate-covariate',
        action='store_true',
        help='Passa a soma total das séries como past covariate para modelos que suportam covariáveis'
    )

    args = parser.parse_args()

    # Se 'all', rodar todos os modelos
    if args.model == 'all':
        models_to_run = [
            'linear',
            'randomforest',
            # 'lightgbm',
            'xgboost',
            'global_naive_aggregate',
            'global_naive_drift',
            'global_naive_seasonal'
        ]
        for model_name in models_to_run:
            print(f"\n{'#'*80}")
            print(f"# RODANDO MODELO: {model_name.upper()}")
            print(f"{'#'*80}\n")
            main(
                model_name=model_name,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                lags=args.lags,
                datasets_filter=args.datasets,
                validation_metric=args.validation_metric,
                top_down=args.top_down,
                use_aggregate_covariate=args.use_aggregate_covariate
            )
    else:
        main(
            model_name=args.model,
            data_dir=args.data_dir,
        output_dir=args.output_dir,
        lags=args.lags,
        datasets_filter=args.datasets,
        validation_metric=args.validation_metric,
        top_down=args.top_down,
        use_aggregate_covariate=args.use_aggregate_covariate
    )
