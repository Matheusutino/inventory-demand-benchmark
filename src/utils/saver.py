import pandas as pd
from pathlib import Path
from typing import Dict, Union

def save_predictions(
    pred_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    output_dir: str = 'data/predictions'
) -> str:
    """
    Salva previsões em formato parquet.
    
    Args:
        pred_df: DataFrame com previsões
        dataset_name: Nome do dataset
        model_name: Nome do modelo
        output_dir: Diretório de saída
        
    Returns:
        Caminho do arquivo salvo
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"{model_name}_{dataset_name}_predictions.parquet"
    filepath = output_path / filename
    
    pred_df.to_parquet(filepath, index=False)
    
    return str(filepath)


def save_metrics(
    metrics_df: pd.DataFrame,
    agg_metrics: Union[Dict, pd.DataFrame],
    dataset_name: str,
    model_name: str,
    output_dir: str = 'data/predictions'
) -> str:
    """
    Salva métricas em formato CSV.
    
    Args:
        metrics_df: DataFrame com métricas por série
        agg_metrics: Dict ou DataFrame com métricas agregadas
        dataset_name: Nome do dataset
        model_name: Nome do modelo
        output_dir: Diretório de saída
        
    Returns:
        Caminho do arquivo salvo
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Salvar métricas por série
    filename = f"{model_name}_{dataset_name}_metrics.csv"
    filepath = output_path / filename
    metrics_df.to_csv(filepath, index=False)
    
    # Salvar métricas agregadas
    agg_filename = f"{model_name}_{dataset_name}_metrics_agg.csv"
    agg_filepath = output_path / agg_filename
    
    # Se for dict, converter para DataFrame
    if isinstance(agg_metrics, dict):
        pd.DataFrame([agg_metrics]).to_csv(agg_filepath, index=False)
    else:
        # Já é DataFrame
        agg_metrics.to_csv(agg_filepath, index=False)
    
    return str(filepath)