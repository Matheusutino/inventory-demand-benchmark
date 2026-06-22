import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

def discover_datasets(data_dir: str = 'data/datasets') -> Dict[str, Dict[str, str]]:
    """
    Auto-descobre todos os datasets na pasta.
    
    Returns:
        Dict com formato: {'dataset_name': {'train': 'path/to/train.parquet', 'test': 'path/to/test.parquet'}}
    """
    data_path = Path(data_dir)
    datasets = {}
    
    for train_file in sorted(data_path.glob('*_train.parquet')):
        dataset_name = train_file.stem.replace('_train', '')
        test_file = train_file.parent / f"{dataset_name}_test.parquet"
        
        if test_file.exists():
            datasets[dataset_name] = {
                'train': str(train_file),
                'test': str(test_file),
            }
    
    return datasets


class DataLoader:
    """Carrega e prepara datasets para forecasting"""
    
    @staticmethod
    def load_dataset(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Carrega dataset e converte para formato long.
        
        Args:
            train_path: Caminho para arquivo train.parquet
            test_path: Caminho para arquivo test.parquet
            
        Returns:
            df_train_long: DataFrame train em formato long
            df_test_long: DataFrame test em formato long
            horizon: Número de timesteps a prever
        """
        # Carregar
        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)
        
        # Converter para long format
        df_train_long = DataLoader._to_long_format(df_train)
        df_test_long = DataLoader._to_long_format(df_test)
        
        horizon = len(df_test)
        
        return df_train_long, df_test_long, horizon
    
    @staticmethod
    def _to_long_format(df: pd.DataFrame) -> pd.DataFrame:
        """Converte DataFrame wide para long format"""
        df_long = df.melt(
            id_vars=['date'],
            var_name='id',
            value_name='target'
        )
        df_long = df_long.rename(columns={'date': 'timestamp'})
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])
        df_long = df_long.sort_values(['id', 'timestamp']).reset_index(drop=True)
        
        return df_long
    
    @staticmethod
    def get_dataset_info(df_train_long: pd.DataFrame, df_test_long: pd.DataFrame, horizon: int) -> Dict:
        """Retorna informações sobre o dataset"""
        n_series = df_train_long['id'].nunique()
        train_length = len(df_train_long) // n_series

        return {
            'n_series': n_series,
            'train_length': train_length,
            'horizon': horizon,
            'date_range': (df_train_long['timestamp'].min(), df_train_long['timestamp'].max()),
            'series_ids': df_train_long['id'].unique().tolist()
        }

    @staticmethod
    def load_excel_aggregated(excel_path: str) -> pd.DataFrame:
        """
        Carrega arquivo Excel com dados de vendas por produto e agrega por subfamília.

        Formato esperado:
        - Colunas: codigo_produto, data, subfamilia, quantidade
        - Agrega quantidade por (data, subfamilia)
        - Converte para formato wide: date + uma coluna por subfamília

        Args:
            excel_path: Caminho para arquivo .xlsx

        Returns:
            DataFrame com colunas: date, subfamilia1, subfamilia2, ...
        """
        # Ler Excel da Sheet1
        df = pd.read_excel(excel_path, sheet_name='Sheet1')

        # Verificar colunas necessárias
        required_cols = ['data', 'subfamilia', 'quantidade']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel deve conter colunas: {required_cols}")

        # Converter data
        df['data'] = pd.to_datetime(df['data'])

        # Agregar por (data, subfamilia)
        df_agg = df.groupby(['data', 'subfamilia'])['quantidade'].sum().reset_index()

        # Converter para formato wide (pivot)
        df_wide = df_agg.pivot(index='data', columns='subfamilia', values='quantidade')

        # Preencher NaNs com 0 (meses sem vendas)
        df_wide = df_wide.fillna(0)

        # Resetar index para ter 'data' como coluna
        df_wide = df_wide.reset_index()

        # Renomear coluna 'data' para 'date'
        df_wide = df_wide.rename(columns={'data': 'date'})

        # Ordenar por data
        df_wide = df_wide.sort_values('date').reset_index(drop=True)

        return df_wide

    @staticmethod
    def load_excel_by_product(excel_path: str, subfamilia: str = None) -> pd.DataFrame:
        """
        Carrega arquivo Excel mantendo séries individuais por produto (SEM agregação).

        Formato esperado:
        - Colunas: codigo_produto, data, subfamilia, quantidade
        - Mantém quantidade por (data, codigo_produto)
        - Converte para formato wide: date + uma coluna por produto

        Args:
            excel_path: Caminho para arquivo .xlsx
            subfamilia: Se especificado, filtra apenas produtos desta subfamília

        Returns:
            DataFrame com colunas: date, produto1, produto2, ...
        """
        # Ler Excel da Sheet1
        df = pd.read_excel(excel_path, sheet_name='Sheet1')

        # Verificar colunas necessárias
        required_cols = ['codigo_produto', 'data', 'quantidade']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel deve conter colunas: {required_cols}")

        # Converter data
        df['data'] = pd.to_datetime(df['data'])

        # Filtrar por subfamília se especificado
        if subfamilia is not None:
            if 'subfamilia' not in df.columns:
                raise ValueError("Coluna 'subfamilia' não encontrada no Excel")
            df = df[df['subfamilia'] == subfamilia].copy()
            if len(df) == 0:
                raise ValueError(f"Nenhum produto encontrado para subfamília: {subfamilia}")

        # Agregar por (data, codigo_produto) - pode haver duplicatas
        df_agg = df.groupby(['data', 'codigo_produto'])['quantidade'].sum().reset_index()

        # Converter para formato wide (pivot)
        df_wide = df_agg.pivot(index='data', columns='codigo_produto', values='quantidade')

        # Preencher NaNs com 0 (meses sem vendas)
        df_wide = df_wide.fillna(0)

        # Resetar index para ter 'data' como coluna
        df_wide = df_wide.reset_index()

        # Renomear coluna 'data' para 'date'
        df_wide = df_wide.rename(columns={'data': 'date'})

        # Ordenar por data
        df_wide = df_wide.sort_values('date').reset_index(drop=True)

        return df_wide

    @staticmethod
    def load_excel_mixed(excel_path: str, subfamilia: str) -> pd.DataFrame:
        """
        Carrega arquivo Excel com séries individuais por produto + série agregada total.

        Formato esperado:
        - Colunas: codigo_produto, data, subfamilia, quantidade
        - Mantém séries individuais dos produtos
        - Adiciona coluna extra com total agregado da subfamília

        Args:
            excel_path: Caminho para arquivo .xlsx
            subfamilia: Subfamília para processar

        Returns:
            DataFrame com colunas: date, produto1, produto2, ..., TOTAL_subfamilia
        """
        # Carregar séries individuais
        df_products = DataLoader.load_excel_by_product(excel_path, subfamilia)

        # Calcular total agregado (soma de todos os produtos por data)
        product_cols = [col for col in df_products.columns if col != 'date']
        df_products[f'TOTAL_{subfamilia}'] = df_products[product_cols].sum(axis=1)

        return df_products