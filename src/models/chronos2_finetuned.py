import argparse
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from src.utils.data_loader import DataLoader, discover_datasets
from src.utils.saver import save_predictions

FINETUNE_MODE = "lora"


def main(
    model_path,
    data_dir,
    output_dir,
    device,
    quantiles,
    datasets_filter,
    num_steps=1000,
    learning_rate=1e-4,
    batch_size=32,
    validation_split=0.2,
    patience=50,
    min_delta=0.001,
):

    # Extrair model_name do model_path
    model_name = model_path.split('/')[-1]
    model_name = f"{model_name}-finetuned-{FINETUNE_MODE}"

    # ==================== DESCOBRIR DATASETS ====================
    datasets = discover_datasets(data_dir)

    # Filtrar datasets se especificado
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    print(f"Encontrados {len(datasets)} datasets\n")

    # ==================== CARREGAR MODELO BASE ====================
    print(f"Carregando {model_path}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_path,
        device_map=device
    )
    print("✓ Modelo carregado\n")
    print("Usando loss padrão do Chronos-2.\n")

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

            # ==================== FINE-TUNING ====================
            print(f"\nIniciando fine-tuning ({FINETUNE_MODE})...")

            # Preparar inputs para fine-tuning
            all_series = []
            for _, group in df_train_long.groupby("id"):
                all_series.append({
                    "target": torch.tensor(group["target"].values, dtype=torch.float32),
                })

            if validation_split > 0 and len(all_series) >= 5:
                n_val = max(1, int(len(all_series) * validation_split))
                n_val = min(n_val, len(all_series) - 1)
                train_inputs = all_series[:-n_val]
                val_inputs = all_series[-n_val:]
                print(f"Train/Val split: {len(train_inputs)}/{len(val_inputs)} series")
            else:
                if len(all_series) < 5:
                    print(f"⚠️  Dataset muito pequeno ({len(all_series)} series), pulando validation split")
                train_inputs = all_series
                val_inputs = None

            from transformers import EarlyStoppingCallback

            callbacks = []
            extra_args = {
                "logging_steps": 50,
            }

            if val_inputs is not None:
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=min_delta,
                ))
                extra_args.update({
                    "eval_strategy": "steps",
                    "eval_steps": 50,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                })
                print(f"Early stopping enabled (patience={patience}, min_delta={min_delta})")

            # Fine-tune usando validation nativo do Chronos-2
            # Ajustar min_past para séries curtas
            context_length = info['train_length'] - horizon
            min_past = max(1, min(horizon, context_length // 2))

            print(f"Context length: {context_length}, min_past: {min_past}")

            finetuned_pipeline = pipeline.fit(
                inputs=train_inputs,
                validation_inputs=val_inputs,
                prediction_length=horizon,
                num_steps=num_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                finetune_mode=FINETUNE_MODE,
                lora_config=None,
                callbacks=callbacks,
                min_past=min_past,
                **extra_args,
            )

            print("✓ Fine-tuning concluído")

            # Usar pipeline fine-tuned para predições
            current_pipeline = finetuned_pipeline

            # ==================== PREVER ====================
            print("\nGerando previsões...")
            pred_df = current_pipeline.predict_df(
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
            import traceback
            print(f"✗ Erro: {e}")
            traceback.print_exc()
            print()
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
    parser = argparse.ArgumentParser(description='Chronos-2 Fine-tuned Forecasting')

    # Model arguments
    parser.add_argument('--model-path', type=str, default='amazon/chronos-2',
                        help='Caminho do modelo no HuggingFace')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device para rodar o modelo')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/datasets',
                        help='Diretório dos datasets')
    parser.add_argument('--output-dir', type=str, default='data/predictions',
                        help='Diretório para salvar previsões')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets específicos (deixe vazio para todos)')

    # Prediction arguments
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.5, 0.9],
                        help='Quantis para previsão probabilística')
    # Fine-tuning arguments
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='Número de steps de treinamento')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (1e-4 recomendado para LoRA, 1e-6 para full)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size para treinamento')

    # Validation / early stopping
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of series to use for validation (0-1)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Minimum change in eval_loss to count as improvement')

    args = parser.parse_args()

    # Chamar main com argumentos
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        quantiles=args.quantiles,
        datasets_filter=args.datasets,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        patience=args.patience,
        min_delta=args.min_delta,
    )
