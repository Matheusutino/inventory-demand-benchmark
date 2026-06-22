# Analysis Scripts

Scripts para avaliar previsões, comparar modelos e gerar visualizações.

## Estrutura de Dados

Por padrão, o projeto usa:

```text
data/
  datasets/      # parquets de train/test e dados originais
  predictions/   # previsões geradas pelos modelos
  analysis/      # métricas, tabelas agregadas e plots
```

## Avaliar Previsões

```bash
python -m src.analysis.analyze_predictions
```

Defaults:

```bash
python -m src.analysis.analyze_predictions \
  --predictions-dir data/predictions \
  --data-dir data/datasets \
  --output-dir data/analysis/metrics
```

Arquivo específico:

```bash
python -m src.analysis.analyze_predictions \
  --predictions-dir data/predictions \
  --file darts_randomforest_all_datasets_predictions.parquet \
  --data-dir data/datasets \
  --output-dir data/analysis/metrics
```

Saídas:

```text
data/analysis/metrics/<modelo>_metrics.csv
data/analysis/metrics/<modelo>_metrics_agg.csv
```

## Comparar Métricas

```bash
python -m src.analysis.plot_aggregate_metrics
```

Defaults:

```bash
python -m src.analysis.plot_aggregate_metrics \
  --metrics-dir data/analysis/metrics \
  --predictions-dir data/predictions \
  --data-dir data/datasets
```

Saída:

```text
data/analysis/plots/comparison_all_datasets_and_panels_<metricas>.png
```

## Descrever Datasets

```bash
python -m src.analysis.describe_datasets
```

Saídas:

```text
data/analysis/dataset/dataset_summary.csv
data/analysis/dataset/dataset_sku_summary.csv
data/analysis/dataset/dataset_panel_summary.csv
```

## Plotar Previsões

```bash
python -m src.analysis.plot_dataset_predictions --dataset Mecanismos_h3
```

Com alguns modelos:

```bash
python -m src.analysis.plot_dataset_predictions \
  --dataset Mecanismos_h3 \
  --models darts_naive_moving_average darts_randomforest darts_global_naive_drift
```

Saídas:

```text
data/analysis/plots/<dataset>_aggregate_comparison.png
data/analysis/plots/<dataset>_individual_series.pdf
```

## Item MAE vs Sum MAE

```bash
python -m src.analysis.plot_item_sum_mae
```

Saídas:

```text
data/analysis/plots/item_sum_mae_by_panel.png
data/analysis/plots/item_sum_mae_overall.png
data/analysis/plots/item_sum_mae_by_dataset.csv
data/analysis/plots/item_sum_mae_by_panel.csv
data/analysis/plots/item_sum_mae_overall.csv
```

## CD Diagram

```bash
python -m src.analysis.cd_diagram
```

Usa a `InventoryDemandLoss`, normaliza cada componente por dataset entre 0 e 1,
ignora `neg`, `tv`, `int` e `total`, e gera um score de regressão menor-melhor
convertido para performance maior-melhor apenas para o teste/ranking do CD diagram.

Saídas:

```text
data/analysis/cd_diagram/inventory_loss_cd.png
data/analysis/cd_diagram/inventory_loss_cd_performance.csv
data/analysis/cd_diagram/inventory_loss_cd_components_normalized.csv
data/analysis/cd_diagram/inventory_loss_cd_scores.csv
data/analysis/cd_diagram/inventory_loss_cd_ranks.csv
```

Para gerar o CD diagram com MAE médio por SKU:

```bash
python -m src.analysis.cd_diagram --score mae --labels
```

Saídas:

```text
data/analysis/cd_diagram/mae_cd.png
data/analysis/cd_diagram/mae_cd_performance.csv
data/analysis/cd_diagram/mae_cd_series_mae.csv
data/analysis/cd_diagram/mae_cd_scores.csv
data/analysis/cd_diagram/mae_cd_ranks.csv
```

## Rodar Tudo

```bash
src/analysis/run_all_analysis.sh --dataset Mecanismos_h3
```
