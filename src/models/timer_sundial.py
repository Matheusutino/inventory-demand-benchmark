import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions


def main(model_path, data_dir, output_dir, num_samples, quantiles, datasets_filter):
    
    # Extrair model_name do model_path
    model_name = model_path.split('/')[-1]
    
    # Determinar se é modelo probabilístico (Sundial) ou pontual (Timer)
    is_probabilistic = 'sundial' in model_path.lower()
    
    # ==================== DESCOBRIR DATASETS ====================
    datasets = discover_datasets(data_dir)
    
    # Filtrar datasets se especificado
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}
    
    print(f"Encontrados {len(datasets)} datasets\n")
    
    # ==================== CARREGAR MODELO ====================
    print(f"Carregando {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    ).cuda()
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
            forecasts_list = []
            
            for serie_id in df_train_long['id'].unique():
                serie_data = df_train_long[df_train_long['id'] == serie_id].sort_values('timestamp')
                
                # Preparar input
                seqs = torch.tensor(
                    serie_data['target'].values,
                    dtype=torch.float32
                ).unsqueeze(0).cuda()
                
                # Prever
                with torch.no_grad():
                    if is_probabilistic:
                        # Sundial: gera múltiplos samples
                        output = model.generate(
                            seqs,
                            max_new_tokens=horizon,
                            num_samples=num_samples
                        )
                        
                        # output shape: (num_samples, batch_size, forecast_length)
                        forecast_samples = output[:, 0, :].cpu().numpy()  # (num_samples, horizon)
                        
                        # Calcular quantis
                        forecast_values = {}
                        for q in quantiles:
                            forecast_values[f'q{int(q*100)}'] = np.quantile(forecast_samples, q, axis=0)
                        
                    else:
                        # Timer: gera apenas valores pontuais
                        output = model.generate(seqs, max_new_tokens=horizon)
                        forecast_point = output.squeeze().cpu().numpy()[-horizon:]
                        
                        # Criar dict com apenas q50 (forecast pontual)
                        forecast_values = {'q50': forecast_point}
                
                # Criar DataFrame
                last_date = serie_data['timestamp'].max()
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
            
        except Exception as e:
            print(f"✗ Erro: {e}\n")
            import traceback
            traceback.print_exc()
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
    
    # Limpar GPU
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # ==================== ARGUMENTOS ====================
    parser = argparse.ArgumentParser(description='Timer/Sundial Forecasting')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Caminho do modelo no HuggingFace (ex: thuml/timer-base-84m ou thuml/sundial-base-128m)')
    parser.add_argument('--data-dir', type=str, default='data/datasets',
                        help='Diretório dos datasets')
    parser.add_argument('--output-dir', type=str, default='data/predictions',
                        help='Diretório para salvar previsões')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Número de samples para modelos probabilísticos (apenas Sundial)')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.5, 0.9],
                        help='Quantis para extrair (apenas para Sundial)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets específicos (deixe vazio para todos)')

    args = parser.parse_args()
    
    # Chamar main com argumentos
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        quantiles=args.quantiles,
        datasets_filter=args.datasets
    )
