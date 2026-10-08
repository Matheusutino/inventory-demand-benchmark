# Analysis Scripts

Análises de **Inventory-Aware Evaluation of Forecasting Models for Sparse Multi-SKU Demand**, usando o
Inventory-Aware Evaluation Framework. Execute os
comandos na raiz do repositório. Figuras são exportadas em **PDF**, tabelas em
CSV e relatórios em Markdown.

Todas as análises usam as famílias, cores e nomes de modelos definidos em
`src/analysis/model_style.py`, seguindo a legenda de
`framework/ranks/inventory_component_rank_heatmap.pdf`:

| Família | Cor |
| --- | --- |
| Naive baselines | Cinza (`#7F7F7F`) |
| Statistical models | Azul (`#4C78A8`) |
| Intermittent-demand methods | Roxo (`#B279A2`) |
| Machine learning models | Laranja (`#F58518`) |
| Foundation models | Verde (`#54A24B`) |

As cores identificam famílias nas barras, nos nomes dos modelos em heatmaps,
nas curvas de previsão e no CD diagram. Nos CSVs, `group` e `family` usam os
mesmos identificadores: `naive`, `statistical`, `intermittent`,
`machine_learning` e `foundation`.
Marcadores distintos ajudam a identificar modelos da mesma família nas curvas.
Todas as figuras são exportadas sem título geral ou de subplot. Horizontes,
métricas e painéis continuam identificados nos rótulos dos eixos. Os nomes
exibidos dos baselines são `Naive Aggregate`, `Naive Drift` e `Naive Seasonal`,
definidos no mesmo módulo compartilhado para todos os gráficos e tabelas.

## Executar o Pipeline

```bash
src/analysis/run_all_analysis.sh --python .venvs/darts/bin/python
```

Também é possível usar `.venvs/analysis/bin/python`. `--analysis-dir DIR`
muda a raiz de toda a árvore; `--metrics-dir` sobrescreve a pasta de métricas.
`--plots-dir` é um alias de compatibilidade para `--analysis-dir`.
`--exclude SCRIPT...` pula etapas; `--fail-fast` interrompe na primeira falha.
`--dataset Mecanismos_h3` inclui previsões individuais.

## Organização das Saídas

```text
data/analysis/
  dataset/                       # Descrições de painéis e SKUs
    overview/                    # Demanda agregada e ADI × CV²
  accuracy/
    metrics/                     # Métricas tradicionais por modelo
    aggregate/                   # Comparações por dataset
    item_sum_mae/                # MAE individual e da demanda somada
    item_mae_by_horizon/         # Item (MAE) em H3 e H6, lado a lado
    sku_winners/                 # Vitórias por SKU/família e oráculo por painel
    cd_diagram/                  # CD diagram de MAE
  diagnostics/
    horizon_degradation/         # H3 versus H6
    mae_vs_horizon/               # MAE versus sensibilidade ao horizonte
  framework/
    components/                  # Valores brutos dos nove componentes
    ranks/                       # Rank médio por modelo e componente
    sensitivity/                 # Parâmetros, exclusão de painéis e violações do Cap
    concordance/                 # Correlações, dispersão e relatório
    wins/                        # Vitórias por componente (suplementar)
  predictions/<dataset>/         # Previsões opcionais
  archive/                       # Resultados anteriores à reorganização
```

Cada pasta reúne seus PDFs e tabelas. A geração atual não usa uma pasta genérica
`plots/`. Arquivos em `archive/` são históricos e podem conter sparse,
nomes anteriores e PNGs.

## Inventory-Aware Evaluation Framework

```bash
python -m src.analysis.evaluate_inventory_framework
python -m src.analysis.plot_inventory_component_rank_heatmap
python -m src.analysis.analyze_component_concordance
```

Componentes ativos: `item`, `sum`, `share`, `alloc`, `scaled`, `cap`, `asym`,
`delta`, `zero`. `weighted_item`, `rel`, `robust`, `sparse`, `neg`, `tv`, `int`
e `total` ficam fora da avaliação. Não há score global na análise do framework.
Nas figuras, os rótulos são **Scaled**, **Zero** e **Sum**, com inicial maiúscula,
conforme `COMPONENT_LABELS` em `inventory_framework.py`. MASE designa o método
de cálculo de `scaled`; “spurious” e “aggregate consistency” não são nomes
de componentes nos gráficos.
O avaliador está em `src/evaluation/inventory.py` e retorna os nove
componentes separadamente, sem gradientes ou objetivo agregado de treinamento.
Os helpers de alinhamento e avaliação ficam em
`src/evaluation/forecast_evaluation.py`.

- **Item** é MAE, igual à figura de acurácia individual e ao critério de validação.
- **Sum**, **Share** (JSD) e **Alloc** preservam suas fórmulas.
- **Scaled** (`scaled`) usa `mean(|y_hat - y| / q_i)`, com
  `q_i = mean(|y_train[t] - y_train[t-1]|)` (lag 1, sem sazonalidade).
- **Cap** é `mean(max(y_hat - c_i, 0)^2)`, com `c_i = 2 * max(y_train[:, i])`.
- **Asym** mantém a pinball com τ = 0,7.
- **Delta** mantém o MSE das diferenças dentro da janela prevista, sem ancorar
  o primeiro passo no último valor observado do treino.
- **Zero** usa `mean(y_hat / mean(y_train[:, i]))` sobre as observações do teste
  em que `y == 0`, com peso igual por observação elegível.

Scaled exclui SKUs com `q_i == 0`; Zero exclui SKUs com média histórica zero.
Um SKU constante com demanda positiva é excluído de Scaled, mas pode entrar
em Zero. Nenhum ε substitui esses denominadores. Cap mantém todos os SKUs,
inclusive os de histórico sem demanda, cujo limite histórico é zero.
Se não há observações elegíveis, o componente é indefinido (NaN), e os ranks
rejeitam a comparação em vez de atribuir valor zero.

O histórico é o treino de cada tarefa: 44 meses em H6 e 47 em H3. Os arquivos
`framework/components/training_scales_by_sku.csv` e `component_coverage.csv`
reportam escalas, limites, elegibilidade, contagens de SKUs excluídos e
observações efetivamente avaliadas. `component_coverage_report.md` resume as
exclusões por painel e horizonte, sem multiplicar as contagens pelos modelos.
O histórico de H6 corresponde ao treino comum usado na Tabela 1; H3 inclui
os três meses adicionais disponíveis na origem posterior da previsão.

As previsões salvas são usadas sem truncamento. A fórmula de Zero é assinada:
previsões negativas podem gerar valores negativos, que são ordenados como
os demais valores desse componente. Scaled e Zero **premiam prever zero**.
Quem contrapõe esse incentivo são Asym, que pune subprevisão, e Sum, que
pune subestimar o total. Zero explicita previsões em observações sem demanda.

O avaliador exige cobertura completa dos mesmos modelos e tarefas e exporta
`framework/components/component_values.csv`, com os valores e cobertura.
Em `framework/wins/`, salva `component_winners.csv`,
`component_win_counts_long.csv`, `component_win_counts_summary.csv` e
`component_win_counts_summary.pdf`. As vitórias permanecem separadas por
componente, sem total global. Empates nas vitórias usam `numpy.isclose` com
`atol=rtol=1e-8`; cada vencedor empatado recebe uma vitória.

## Heatmap Modelo × Componente

Entrada: `framework/components/component_values.csv`. Cada tarefa é um
painel × horizonte (cinco painéis × H3/H6 = dez tarefas). Os modelos são
ordenados pelo valor crescente de cada componente em cada tarefa; empates
exatos recebem rank médio. Cada célula mostra a média dos ranks nas tarefas,
com pesos iguais. Não há normalização, coluna Overall ou score agregado.

As linhas seguem as cinco famílias do artigo: naive baselines, statistical
models, intermittent-demand methods, machine learning models e foundation
models. A ordem dentro de cada família é alfabética.

Saídas em `framework/ranks/`: `inventory_component_rank_heatmap.pdf`,
`inventory_component_mean_ranks.csv` e `inventory_component_task_ranks.csv`.
`--models`, `--datasets` e `--components` selecionam a comparação antes
dos ranks; retirar modelos altera o conjunto de competidores. Resultados
ausentes, não finitos ou com cobertura inconsistente são rejeitados.

## Sensibilidade do Framework

```bash
python -m src.analysis.analyze_framework_sensitivity
```

Usa os 20 modelos e dez tarefas das previsões salvas, sem retreinamento.
Varia um fator por vez: Asym τ em {0.5, 0.6, 0.7, 0.8, 0.9}; Cap κ em
{1.25, 1.5, 2, 3, 5}; Scaled com escala de lag 1, lag 12 ou média histórica;
Zero normalizado pela média, máximo histórico ou sem normalização; Sum e
Delta com erro quadrático ou absoluto. Sum preserva o denominador relativo
original e Delta continua sem ancoragem no treino. Zero mantém a mesma
elegibilidade nas três versões, excluindo SKUs sem demanda no histórico.
Scaled exclui escalas nulas em cada versão e reporta a cobertura resultante.

Também recalcula os ranks dos nove componentes originais deixando um painel
de fora de cada vez (H3 e H6 juntos, oito tarefas restantes). Em cada tarefa,
empates exatos recebem rank médio; a média dos ranks dá peso igual às
tarefas. O ranking usa o CSV exportado, como o heatmap da RQ2, preservando os
empates da representação numérica persistida. Cada cenário é comparado com
o original do mesmo componente nas dez tarefas, usando Spearman,
sobreposição do top-5 e mudança no conjunto de líderes.

O top-5 contém cinco nomes, resolvendo empates no corte pelo nome exibido
em ordem alfabética, depois pelo identificador. Os ranks permanecem
empatados. A tabela marca cortes ambíguos com `*`; o CSV também reporta
sobreposição e tamanhos incluindo todos os empatados. A liderança inclui
todos os co-líderes. Spearman de vetores constantes é indefinido (NaN).

Saídas em `framework/sensitivity/`:

- `mean_ranks.csv`: ranks médios de todos os modelos nos 68 cenários
  (nove originais, 14 alternativas paramétricas, 45 exclusões de painel);
- `sensitivity_summary.csv` e `sensitivity_report.md`: tabela-resumo,
  metodologia, cobertura e interpretação dos empates;
- `asym_cap_sensitivity.pdf`: duas curvas de sensibilidade, uma linha por
  modelo, com destaques escolhidos separadamente para cada componente;
- `sensitivity_overview.pdf`: todas as 59 variações efetivas em uma figura,
  com Spearman, retenção do top-5 e losangos indicando mudança de líder;
- `sensitivity_all_model_ranks.pdf`: os 1.360 ranks médios dos vinte modelos
  nos 68 cenários; números mostram ranks e cores mostram mudanças frente
  ao original do mesmo componente (azul = melhora; vermelho = piora);
- `rank_changes.csv`: ranks originais e mudanças de rank por modelo/cenário;
- `highlighted_models.csv`: modelos destacados e empates na escolha do
  melhor original de cada família;
- `component_values.csv` e `task_ranks.csv`: valores e ranks por tarefa;
- `training_scales.csv` e `component_coverage.csv`: escalas históricas,
  exclusões e número de termos avaliados;
- `cap_violations_by_task.csv` e `cap_violations_by_model.csv`: contagens e
  frações das previsões estritamente acima do limite para cada κ.

Violações do Cap são pares SKU/mês por tarefa, incluindo os limites zero
dos históricos sem demanda. A soma das dez tarefas conta separadamente as
previsões H3/H6, cujas janelas são aninhadas. `--data-dir`,
`--predictions-dir` e `--output-dir` alteram os caminhos. Esta etapa também
faz parte de `run_all_analysis.sh`.

Para refazer apenas as figuras gerais usando os CSVs já calculados:

```bash
python -m src.analysis.plot_framework_sensitivity_overview
```

## Concordância entre Componentes

A análise principal calcula Spearman entre modelos **em cada tarefa** e depois
tira a média dos coeficientes, com pesos iguais. As células mostram média ±
desvio-padrão amostral entre tarefas (`ddof=1`); o CSV também exporta mediana,
mínimo e máximo. A dispersão não é intervalo de confiança.

A análise complementar correlaciona os ranks médios, descrevendo o padrão
estável entre tarefas. Kendall tau-b fornece outra verificação. As figuras
mostram somente o triângulo inferior, sem diagonal; os CSVs contêm matrizes
completas. Rankings constantes têm correlação indefinida e contagens de
tarefas válidas são exportadas.

Saídas em `framework/concordance/`:

- `component_concordance_spearman.pdf/csv`: análise principal;
- `component_concordance_spearman_task_mean.csv` e
  `component_concordance_spearman_task_std.csv`: média e dispersão;
- `component_concordance_spearman_mean_ranks.pdf/csv`: complementar;
- `component_concordance_kendall_mean_ranks.pdf/csv`: complementar;
- `component_concordance_by_task.csv` e `component_concordance_pairs.csv`;
- `component_concordance_task_ranks.csv`,
  `component_concordance_mean_ranks.csv` e
  `component_concordance_spearman_valid_tasks.csv`;
- `component_concordance_model_rank_differences.csv`: diferenças versus
  rank médio de `item`;
- `component_concordance_report.md`: metodologia e diagnóstico.

O limiar `--similarity-threshold 0.9` é descritivo. O relatório só substitui
um componente por um representante quando o limiar é atingido em todas as
tarefas. Isso não comprova independência. H3/H6 têm janelas aninhadas; as dez
tarefas não são automaticamente observações independentes para inferência.
`item` usa MAE; `scaled` usa MASE de lag 1. Escalas de MASE, médias de Zero
e limites de Cap usam apenas o treino pareado da tarefa.

## Datasets

```bash
python -m src.analysis.describe_datasets
python -m src.analysis.plot_dataset_overview
```

Em `dataset/overview/`, gera `dataset_aggregate_demand.pdf`,
`dataset_adi_cv2.pdf` e `dataset_overview.pdf`, com tabelas correspondentes.
A demanda é indexada para média do treino comum = 100; `--scale absolute`
usa unidades originais. As janelas H6/H3 são aninhadas.

ADI e CV² usam somente o treino comum (44 meses nos dados atuais).
ADI = meses / meses com demanda positiva; CV² = variância amostral dos
tamanhos positivos / média positiva². Os limiares são 1.32 e 0.49.
SKUs sem demanda ou com um único evento ficam fora da dispersão e são
contabilizados. O histórico é reconstruído uma vez por painel, validando
concordância entre horizontes. `--plots` e `--panels` selecionam figuras e
painéis.

## Acurácia, Diagnósticos e Previsões

```bash
python -m src.analysis.analyze_predictions
python -m src.analysis.plot_aggregate_metrics
python -m src.analysis.plot_item_sum_mae
python -m src.analysis.plot_item_mae_by_horizon
python -m src.analysis.analyze_sku_winners
python -m src.analysis.plot_horizon_degradation
python -m src.analysis.plot_item_mae_vs_degradation
python -m src.analysis.cd_diagram --score mae --labels
python -m src.analysis.plot_dataset_predictions --dataset Mecanismos_h3 --max-series 10
```

`accuracy/item_mae_by_horizon/item_mae_h3_h6_overall.pdf` compara o componente
Item (MAE) em H3 e H6, com a mesma ordem de modelos e escala linear nos dois
painéis. Calcula-se o MAE médio por SKU em cada tarefa e depois a média dos
painéis de negócio, com pesos iguais, separadamente para H3 e H6. Os valores
absolutos em unidades de demanda também são exportados em CSV. A legenda
das famílias fica inteira em uma linha horizontal.

`accuracy/sku_winners/sku_family_wins_and_oracle.pdf` mostra, por painel e
horizonte, as frações de SKUs vencidas por família e a redução de MAE do
oráculo frente ao melhor modelo único daquele painel/horizonte. O MAE de
cada SKU usa toda a janela prevista: o oráculo escolhe um modelo fixo por
SKU em retrospecto, sem alternar modelos entre meses. Todos os modelos
devem cobrir exatamente os mesmos SKUs e timestamps; dados incompletos,
duplicados ou não finitos são rejeitados.

Empates exatos dividem uma vitória igualmente entre os modelos vencedores.
Nas barras, o crédito é dividido igualmente entre as famílias vencedoras,
independentemente de quantos modelos empataram em cada família. Cada barra
soma 100%. `model_win_shares.csv` também distingue crédito fracionado,
vitórias exclusivas e presença entre co-vencedores. `sku_winners.csv` e
`sku_family_winners.csv` identificam os vencedores de cada SKU.

`oracle_by_panel.csv` compara o oráculo e o melhor modelo único em cada
tarefa; `oracle_overall.csv` usa peso igual por painel nos MAEs e peso igual
por par painel/SKU nas frações globais de vitórias. `sku_model_mae.csv`
preserva todos os erros e `sku_summary.csv` conta empates e SKUs sem demanda
no teste. `model_wins_by_category.csv` cruza as vitórias com as categorias
SBC calculadas no treino comum, sem usar a demanda do teste na classificação.
`sku_winners_report.md` explica os critérios e resume os resultados, incluindo
o TSB. O ganho do oráculo é retrospectivo; não demonstra que uma política de
seleção treinada conseguiria reproduzi-lo. `--models` restringe os competidores
antes de calcular vitórias e oráculo.

`sku_family_wins_by_count_and_volume.pdf` compara, lado a lado, win shares por
contagem de SKUs e por volume de demanda observado no teste, por painel e
horizonte. A última linha reúne todos os painéis. Os vencedores continuam
sendo os de menor MAE por SKU; apenas a ponderação muda: para cada SKU,
`V_i = sum(y_test[:, i])`, e `volume_win_share = sum(V_i * credit_i) / sum(V_i)`.
Empates dividem o volume entre modelos ou famílias com os mesmos créditos
usados para contagem. SKUs sem demanda no teste têm peso zero na versão por
volume; um painel inteiro sem demanda tem participação indefinida (NaN).
A figura explicita essa definição no rótulo dos eixos da direita e na nota
abaixo da legenda, incluindo o peso zero dos SKUs sem demanda no teste.

`model_win_shares.csv` e `family_win_shares.csv` incluem `won_test_volume`,
`total_test_volume` e `volume_win_share`. Os arquivos
`model_win_shares_overall.csv` e `family_win_shares_overall.csv` agregam
separadamente H3/H6: somam volumes entre painéis antes de normalizar, dando
mais peso aos painéis de maior demanda. Também incluem as médias das
participações com pesos iguais por painel (`equal_panel_win_share` e
`equal_panel_volume_win_share`); nesta última, painéis sem volume ficam fora,
e `n_panels_with_volume` informa a cobertura. Os CSVs de vencedores por SKU
incluem `model_volume_credit` ou `family_volume_credit`. O relatório compara
as lideranças por contagem e volume, tanto globalmente quanto por painel.

MAE permanece uma métrica tradicional em `accuracy/`; não é um componente
adicional na concordância do framework. O pipeline gera apenas o CD diagram
de MAE. As análises legadas de componentes normalizados (`cd_diagram --score
inventory_score`, `plot_inventory_score_panel_heatmap` e
`plot_dataset_rank_heatmap`) usam a coluna `normalized_score`, ficam fora da
execução padrão e não representam o framework atual. O CD diagram de MAE
usa a coluna `mae` em seus CSVs.

Previsões são salvas em `predictions/<dataset>/`, em PDF. `--models` e
`--series` selecionam modelos e SKUs; `--per-model` inclui PDFs separados.
