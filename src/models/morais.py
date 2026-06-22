import argparse
import pandas as pd
import numpy as np
import torch
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions

def main(model_type, model_size, data_dir, output_dir, context_length, patch_size, batch_size, num_samples, quantiles, datasets_filter):
    
    # Extrair model_name
    model_name = f"{model_type}-{model_size}"
    
    # ==================== DESCOBRIR DATASETS ====================
    datasets = discover_datasets(data_dir)
    
    # Filtrar datasets se especificado
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}
    
    print(f"Encontrados {len(datasets)} datasets\n")
    
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
            
            # ==================== CONVERTER PARA GLUONTS ====================
            ds = PandasDataset(
                {serie_id: df_train_long[df_train_long['id'] == serie_id].set_index('timestamp')['target']
                 for serie_id in df_train_long['id'].unique()},
                freq='M'
            )
            
            # ==================== CARREGAR MODELO ====================
            print(f"Carregando {model_type}-{model_size}...")
            
            if model_type == "moirai":
                model = MoiraiForecast(
                    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{model_size}"),
                    prediction_length=horizon,
                    context_length=context_length,
                    patch_size=patch_size,
                    num_samples=num_samples,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            elif model_type == "moirai-moe":
                model = MoiraiMoEForecast(
                    module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{model_size}"),
                    prediction_length=horizon,
                    context_length=context_length,
                    patch_size=16, 
                    num_samples=num_samples,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            elif model_type == "moirai2":
                model = Moirai2Forecast(
                    module=Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{model_size}"),
                    prediction_length=horizon,
                    context_length=context_length,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            
            print("✓ Modelo carregado")
            
            # ==================== PREVER ====================
            print("Gerando previsões...")
            predictor = model.create_predictor(batch_size=batch_size)
            forecasts = predictor.predict(ds)
            
            forecasts_list = []
            
            for serie_id, forecast in zip(df_train_long['id'].unique(), forecasts):
                # Extrair previsões (API diferente para Moirai2)
                if model_type == "moirai2":
                    forecast_values = {}
                    for q in quantiles:
                        forecast_values[f'q{int(q*100)}'] = forecast.quantile(q)
                else:
                    forecast_samples = forecast.samples  # (num_samples, horizon)
                    forecast_values = {}
                    for q in quantiles:
                        forecast_values[f'q{int(q*100)}'] = np.quantile(forecast_samples, q, axis=0)
                
                # Criar DataFrame
                last_date = df_train_long[df_train_long['id'] == serie_id]['timestamp'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=horizon,
                    freq='MS'
                )
                
                forecast_dict = {
                    'id': serie_id,
                    'timestamp': future_dates,
                }
                forecast_dict.update(forecast_values)
                
                forecasts_list.append(pd.DataFrame(forecast_dict))
            
            pred_df = pd.concat(forecasts_list, ignore_index=True)
            
            # Adicionar coluna de dataset
            pred_df['dataset'] = dataset_name
            
            # Acumular resultados
            all_predictions.append(pred_df)
            
            print(f"✓ Dataset processado\n")
            
            # Limpar memória
            del model, predictor
            torch.cuda.empty_cache()
            
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
    parser = argparse.ArgumentParser(description='Moirai Forecasting')
    parser.add_argument('--model-type', type=str, default='moirai2',
                        choices=['moirai', 'moirai-moe', 'moirai2'],
                        help='Tipo do modelo Moirai')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Tamanho do modelo')
    parser.add_argument('--data-dir', type=str, default='data/datasets',
                        help='Diretório dos datasets')
    parser.add_argument('--output-dir', type=str, default='data/predictions',
                        help='Diretório para salvar previsões')
    parser.add_argument('--context-length', type=int, default=200,
                        help='Context length')
    parser.add_argument('--patch-size', type=str, default='auto',
                        help='Patch size (auto, 8, 16, 32, 64, 128)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Número de samples para previsão probabilística')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.5, 0.9],
                        help='Quantis para extrair')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets específicos (deixe vazio para todos)')

    args = parser.parse_args()
    
    # Chamar main com argumentos
    main(
        model_type=args.model_type,
        model_size=args.model_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        context_length=args.context_length,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        quantiles=args.quantiles,
        datasets_filter=args.datasets
    )
