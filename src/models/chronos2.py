import argparse
import pandas as pd
from chronos import Chronos2Pipeline
from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions

def main(model_path, data_dir, output_dir, device, quantiles, datasets_filter):
    
    # Extrair model_name do model_path
    # Ex: "amazon/chronos-2" -> "chronos-2"
    model_name = model_path.split('/')[-1]
    
    # ==================== DESCOBRIR DATASETS ====================
    datasets = discover_datasets(data_dir)
    
    # Filtrar datasets se especificado
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}
    
    print(f"Encontrados {len(datasets)} datasets\n")
    
    # ==================== CARREGAR MODELO ====================
    print(f"Carregando {model_path}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_path,
        device_map=device
    )
    print("✓ Modelo carregado\n")
    
    # Listas para acumular todas as previsões
    all_predictions = []
    
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
            
            # ==================== PREVER ====================
            print("Gerando previsões...")
            pred_df = pipeline.predict_df(
                df_train_long,
                prediction_length=horizon,
                quantile_levels=quantiles,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            
            # Renomear colunas dos quantis para formato limpo
            quantile_cols = {str(q): f'q{int(q*100)}' for q in quantiles}
            pred_df = pred_df.rename(columns=quantile_cols)
            
            # Adicionar coluna de dataset
            pred_df['dataset'] = dataset_name
            
            # Acumular resultados
            all_predictions.append(pred_df)
            
            print(f"✓ Dataset processado\n")
            
        except Exception as e:
            print(f"✗ Erro: {e}\n")
            continue
    
    # ==================== SALVAR TUDO ====================
    if all_predictions:
        # Concatenar todas as previsões
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Reordenar colunas (dataset primeiro)
        pred_cols = ['dataset', 'id', 'timestamp'] + [col for col in final_predictions.columns if col.startswith('q')]
        final_predictions = final_predictions[pred_cols]
        
        # Salvar arquivos únicos
        pred_path = save_predictions(final_predictions, 'all_datasets', model_name, output_dir)
        
        print(f"\n{'='*70}")
        print("SALVAMENTO FINAL")
        print(f"{'='*70}")
        print(f"✓ Todas as previsões salvas: {pred_path}")
        print(f"\nTotal de datasets processados: {len(all_predictions)}")
    else:
        print("\n✗ Nenhum dataset foi processado com sucesso")


if __name__ == "__main__":
    # ==================== ARGUMENTOS ====================
    parser = argparse.ArgumentParser(description='Chronos-2 Forecasting')
    parser.add_argument('--model-path', type=str, default='amazon/chronos-2',
                        help='Caminho do modelo no HuggingFace')
    parser.add_argument('--data-dir', type=str, default='data/datasets', 
                        help='Diretório dos datasets')
    parser.add_argument('--output-dir', type=str, default='data/predictions',
                        help='Diretório para salvar previsões')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device para rodar o modelo')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.5, 0.9],
                        help='Quantis para previsão probabilística')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets específicos (deixe vazio para todos)')

    args = parser.parse_args()
    
    # Chamar main com argumentos
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        quantiles=args.quantiles,
        datasets_filter=args.datasets
    )
