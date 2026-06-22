# Enterprise Demand Forecasting Benchmark

Benchmark de modelos de forecasting para demanda empresarial mensal em múltiplos
SKUs. O projeto compara modelos estatísticos, machine learning clássico e
foundation models nos horizontes H3 e H6.

Além de métricas tradicionais, o benchmark utiliza a `InventoryDemandLoss`, uma
avaliação composta voltada a aspectos operacionais da demanda, como erro por
item, volume agregado, participação dos SKUs, alocação, esparsidade e variação
temporal.

## Estrutura

```text
.
├── data/
│   ├── datasets/       # Datasets train/test
│   ├── predictions/    # Previsões dos modelos
│   └── analysis/       # Métricas, tabelas e figuras
├── requirements/       # Dependências separadas por modelo
├── scripts/
│   ├── setup_venvs.sh  # Criação dos ambientes
│   └── run_models.py   # Runner dos modelos
└── src/
    ├── analysis/       # Métricas e visualizações
    ├── losses/         # InventoryDemandLoss
    ├── models/         # Wrappers dos modelos
    └── utils/          # Carregamento e salvamento
```

O diretório `data/` é ignorado pelo Git. Os dados e resultados precisam ser
disponibilizados localmente.

## Formato dos Datasets

Cada dataset é composto por dois arquivos:

```text
data/datasets/<dataset>_train.parquet
data/datasets/<dataset>_test.parquet
```

Os arquivos devem estar no formato wide:

```text
date | SKU_1 | SKU_2 | ... | SKU_N
```

- `date`: timestamp mensal;
- demais colunas: demanda de cada SKU;
- número de linhas no teste: horizonte de previsão.

Exemplo:

```text
data/datasets/Filtros_h3_train.parquet
data/datasets/Filtros_h3_test.parquet
data/datasets/Filtros_h6_train.parquet
data/datasets/Filtros_h6_test.parquet
```

## Instalação

O projeto usa um ambiente virtual separado por família de modelos para evitar
conflitos de dependências.

Criar todos os ambientes:

```bash
scripts/setup_venvs.sh
```

Criar apenas ambientes específicos:

```bash
scripts/setup_venvs.sh chronos darts analysis
```

Atualizar as dependências de ambientes existentes:

```bash
scripts/setup_venvs.sh --sync-existing chronos darts analysis
```

Os ambientes são criados em:

```text
.venvs/analysis
.venvs/chronos
.venvs/darts
.venvs/granite_ttm
.venvs/morais
.venvs/tabpfn_time_series
.venvs/timer_sundial
.venvs/timesfm
.venvs/tirex
```

O runner não precisa ativar os ambientes. Ele chama diretamente o interpretador
correto em `.venvs/<ambiente>/bin/python`.

## Modelos

Foundation models:

- Amazon Chronos-2;
- Salesforce Moirai 2.0 Small;
- Google TimesFM 2.5 200M;
- Prior Labs TabPFN-TS v3;
- NX-AI TiRex;
- IBM Granite TinyTimeMixer R2;
- THUML Sundial Base 128M;
- THUML Timer Base 84M.

Baselines e modelos clássicos via Darts:

- ARIMA;
- Prophet;
- Moving Average;
- Linear Regression;
- Random Forest;
- XGBoost;
- Global Naive Aggregate;
- Global Naive Drift;
- Global Naive Seasonal.

Os modelos Darts usam validação interna por dataset. Por padrão, a seleção é
feita por MAE.

## Executar Modelos

Listar modelos e datasets disponíveis:

```bash
scripts/run_models.py --list
```

Executar todos os modelos padrão em todos os datasets:

```bash
scripts/run_models.py --device cuda
```

Modelos que exigem treinamento ou licença externa não são incluídos
automaticamente.

Executar um modelo:

```bash
scripts/run_models.py --models timesfm
```

Executar modelos específicos:

```bash
scripts/run_models.py \
  --models chronos2 moirai2_small tirex \
  --device cuda
```

Executar em um dataset:

```bash
scripts/run_models.py \
  --models granite_ttm \
  --datasets Mecanismos_h3
```

Executar combinações específicas:

```bash
scripts/run_models.py \
  --models timesfm tabpfn_ts \
  --datasets Filtros_h3 "Placas de Controle_h6" \
  --include-gated
```

As previsões são salvas em:

```text
data/predictions/<modelo>_all_datasets_predictions.parquet
```

### TabPFN-TS

O TabPFN-TS local exige aceite de licença e uma API key da Prior Labs:

1. acesse `https://ux.priorlabs.ai`;
2. aceite a licença do TabPFN;
3. copie a API key;
4. exporte o token:

```bash
export TABPFN_TOKEN="<sua-api-key>"
```

Depois execute:

```bash
scripts/run_models.py \
  --models tabpfn_ts \
  --include-gated
```

## Executar Análises

Criar o ambiente de análise:

```bash
scripts/setup_venvs.sh analysis
```

Executar todo o pipeline:

```bash
src/analysis/run_all_analysis.sh \
  --python .venvs/analysis/bin/python
```

Gerar também os gráficos de previsão para um dataset:

```bash
src/analysis/run_all_analysis.sh \
  --python .venvs/analysis/bin/python \
  --dataset Mecanismos_h3
```

Pular análises específicas:

```bash
src/analysis/run_all_analysis.sh \
  --python .venvs/analysis/bin/python \
  --exclude cd_diagram.py plot_dataset_predictions.py
```

## Inventory Demand Loss

A implementação está em:

```text
src/losses/loss.py
```

Os componentes utilizados na análise atual são:

- `item`: erro médio por SKU;
- `sum`: erro no volume total;
- `share`: diferença na participação relativa dos SKUs;
- `alloc`: erro de alocação entre itens;
- `weighted_item`: erro ponderado por escala;
- `rel`: erro relativo;
- `cap`: penalização de previsões excessivas;
- `asym`: penalização assimétrica;
- `sparse`: comportamento em demanda esparsa;
- `delta`: erro na variação temporal;
- `robust`: erro com influência limitada de extremos.

Os componentes `neg`, `tv` e `int` não são usados na comparação principal.
Para o CD diagram e os heatmaps, cada componente é normalizado entre 0 e 1
dentro de cada dataset antes da agregação.

## Principais Saídas

Métricas tradicionais:

```text
data/analysis/metrics/
```

Caracterização dos datasets:

```text
data/analysis/dataset/dataset_summary.csv
data/analysis/dataset/dataset_sku_summary.csv
data/analysis/dataset/dataset_panel_summary.csv
```

Vitórias por componente da loss:

```text
data/analysis/loss_wins/loss_component_values.csv
data/analysis/loss_wins/loss_component_winners.csv
data/analysis/loss_wins/loss_component_win_counts_summary.csv
```

CD diagrams:

```text
data/analysis/cd_diagram/inventory_loss_cd.png
data/analysis/cd_diagram/mae_cd.png
```

Figuras principais:

```text
data/analysis/plots/item_sum_mae_overall.pdf
data/analysis/plots/item_mae_vs_horizon_degradation.pdf
data/analysis/plots/inventory_loss_panel_heatmap.pdf
data/analysis/plots/dataset_panel_inventory_rank_heatmap.pdf
```

Tabelas correspondentes são salvas em CSV no mesmo diretório.

## Execução em Background

Para rodar todos os modelos em background:

```bash
nohup scripts/run_models.py --device cuda \
  > run_models.log 2>&1 &
```

Acompanhar:

```bash
tail -f run_models.log
```

Para rodar apenas alguns modelos:

```bash
nohup scripts/run_models.py \
  --models granite_ttm timer_base timesfm \
  --device cuda \
  > run_selected_models.log 2>&1 &
```

## Reprodutibilidade

- Execute os comandos a partir da raiz do projeto.
- Use os arquivos em `requirements/` para reproduzir os ambientes.
- Preserve os nomes dos datasets entre os arquivos train/test.
- Para comparações finais, regenere as previsões antes de executar
  `run_all_analysis.sh`.
- Checkpoints e APIs externas podem exigir autenticação ou aceite de licença.

Mais detalhes sobre os scripts de análise estão em
[`src/analysis/README.md`](src/analysis/README.md).
