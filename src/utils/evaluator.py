import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(
    df_test_long: pd.DataFrame,
    pred_df: pd.DataFrame,
    metrics: List[str] = ['mse']
) -> pd.DataFrame:
    """
    Avalia previsões usando métricas customizáveis.
    
    Args:
        df_test_long: Ground truth em formato long
        pred_df: Previsões (deve ter colunas: 'id', 'timestamp', 'forecast')
        metrics: Lista de métricas a calcular ['mse', 'mae', 'rmse', 'mape']
        
    Returns:
        DataFrame com métricas por série
    """
    results = []
    
    for serie_id in df_test_long['id'].unique():
        y_true = df_test_long[df_test_long['id'] == serie_id]['target'].values
        y_pred = pred_df[pred_df['id'] == serie_id]['forecast'].values
        
        serie_metrics = {'id': serie_id}
        
        # Calcular métricas solicitadas
        if 'mse' in metrics:
            serie_metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        if 'mae' in metrics:
            serie_metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'rmse' in metrics:
            serie_metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'mape' in metrics:
            serie_metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        results.append(serie_metrics)
    
    return pd.DataFrame(results)


def aggregate_metrics(metrics_df: pd.DataFrame) -> Dict:
    """
    Agrega métricas (média, mediana, std).
    
    Args:
        metrics_df: DataFrame retornado por evaluate_predictions
        
    Returns:
        Dict com estatísticas agregadas
    """
    agg = {}
    
    for col in metrics_df.columns:
        if col != 'id':
            agg[f'{col}_mean'] = metrics_df[col].mean()
            agg[f'{col}_median'] = metrics_df[col].median()
            agg[f'{col}_std'] = metrics_df[col].std()
    
    return agg