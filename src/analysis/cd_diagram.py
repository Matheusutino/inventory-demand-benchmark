# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare

from src.analysis.analyze_loss_wins import (
    DEFAULT_LOSS_CONFIG,
    IGNORED_COMPONENTS,
    active_components,
    evaluate_models,
    list_prediction_files,
    load_truth_by_dataset,
    model_name_from_path,
)

def nemenyi_posthoc(alpha=0.05, df_perf=None, verbose=False):
    """
    Applies the Nemenyi post-hoc test after Friedman test.
    Requires scikit-posthocs: pip install scikit-posthocs
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        raise ImportError("Nemenyi posthoc requires scikit-posthocs. Install with: pip install scikit-posthocs")

    if verbose:
        print(pd.unique(df_perf['classifier_name']))
    df_counts = pd.DataFrame({'count': df_perf.groupby(['classifier_name']).size()}).reset_index()
    max_nb_datasets = df_counts['count'].max()
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]['classifier_name'])

    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        if verbose:
            print('the null hypothesis over the entire classifiers cannot be rejected')
        return None

    m = len(classifiers)
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers),
                            columns=np.unique(sorted_df_perf['dataset_name']))

    # posthoc_nemenyi_friedman expects (datasets x classifiers)
    nemenyi_pvalues = sp.posthoc_nemenyi_friedman(df_ranks.T)

    p_values = []
    sorted_classifiers = np.sort(classifiers)
    for i in range(m - 1):
        classifier_1 = sorted_classifiers[i]
        for j in range(i + 1, m):
            classifier_2 = sorted_classifiers[j]
            p_value = float(nemenyi_pvalues.iloc[i, j])
            is_significant = bool(p_value <= alpha)
            p_values.append((classifier_1, classifier_2, p_value, is_significant))

    dfff = df_ranks.rank(ascending=False)
    if verbose:
        print(dfff[dfff == 1.0].sum(axis=1))

    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    return p_values, average_ranks, max_nb_datasets


def draw_cd_diagram(output_path, df_perf=None, alpha=0.05, title=None, labels=False, posthoc='nemenyi', verbose=False):
    """
    Mantém as suas chamadas originais de cálculo, mas usa o scikit-posthocs 
    para gerar o desenho final sem sobreposição de barras e textos.
    """
    # 1. Mantém os seus cálculos originais exatamente como eram
    if posthoc == 'nemenyi':
        posthoc_result = nemenyi_posthoc(alpha=alpha, df_perf=df_perf, verbose=verbose)
    else:
        posthoc_result = wilcoxon_holm(alpha=alpha, df_perf=df_perf, verbose=verbose)

    if posthoc_result is None:
        average_ranks = compute_average_ranks(df_perf)
        classifiers = average_ranks.keys()
        p_values = [
            (clf1, clf2, 1.0, False)
            for i, clf1 in enumerate(classifiers)
            for clf2 in list(classifiers)[i + 1 :]
        ]
    else:
        p_values, average_ranks, _ = posthoc_result

    if verbose:
        print(average_ranks)
        for p in p_values:
            print(p)

    # --- NOVA LÓGICA DE DESENHO ---
    try:
        import scikit_posthocs as sp
    except ImportError:
        raise ImportError("O novo desenho requer o pacote. Instale com: pip install scikit-posthocs")

    # 2. Converte a sua lista de tuplas de p_values em uma matriz quadrada (necessário para o desenho)
    classifiers = average_ranks.keys()
    p_val_matrix = pd.DataFrame(1.0, index=classifiers, columns=classifiers)

    for clf1, clf2, p_val, is_sign in p_values:
        p_val_matrix.loc[clf1, clf2] = p_val
        p_val_matrix.loc[clf2, clf1] = p_val

    # 3. Configura a figura
    plt.figure(figsize=(9, max(4, len(classifiers) * 0.4))) # Altura dinâmica baseada na qtd de classificadores

    if title:
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 22}
        plt.title(title, fontdict=font, pad=20)

    # 4. Define se vai mostrar os números (labels=True) ou não, controlando o espaçamento
    if labels:
        # Mostra o Nome do algoritmo e o (Rank)
        fmt_left = '{label} ({rank:.3f})'
        fmt_right = '({rank:.3f}) {label}'
    else:
        # Mostra apenas o Nome
        fmt_left = '{label}'
        fmt_right = '{label}'

    fig, ax = plt.subplots(figsize=(10, 2))

    # 5. Gera o gráfico limpo
    sp.critical_difference_diagram(
        ranks=average_ranks,
        sig_matrix=p_val_matrix,
        alpha=alpha,
        ax=ax,
        label_fmt_left=fmt_left,
        label_fmt_right=fmt_right,
        
        # --- CUSTOMIZAÇÕES AQUI ---
        
        # 1. crossbar_props: Linhas horizontais grossas (que indicam empate estatístico)
        crossbar_props={
            'color': 'black',  # Cor da linha (ex: vermelho, pode usar hexadecimal)
            'linewidth': 5       # Grossura da linha
        },
        
        # 2. elbow_props: Linhas que saem do eixo principal e vão até os nomes
        elbow_props={
            'color': 'black',     # Cor da linha 
            'linewidth': 1.5,    # Grossura da linha
            #'linestyle': '--'    # Estilo da linha (Tracejado, opcional)
        },
        
        # 3. marker_props (Bônus): As bolinhas no eixo principal
        marker_props={
            'color': 'black',    # Cor da bolinha
            's': 30              # Tamanho da bolinha (size)
        }
    )

    # 6. Salva o arquivo final
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def compute_average_ranks(df_perf: pd.DataFrame) -> pd.Series:
    df_counts = pd.DataFrame({'count': df_perf.groupby(['classifier_name']).size()}).reset_index()
    max_nb_datasets = df_counts['count'].max()
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]['classifier_name'])
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)].sort_values(
        ['classifier_name', 'dataset_name']
    )
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(len(classifiers), max_nb_datasets)
    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(classifiers),
        columns=np.unique(sorted_df_perf['dataset_name']),
    )
    return df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)

def wilcoxon_holm(alpha=0.05, df_perf=None, verbose=False):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    if verbose:
        print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        if verbose:
            print('the null hypothesis over the entire classifiers cannot be rejected')
        return None
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]
                              ['accuracy'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    if verbose:
        print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets


def minmax_normalize_components(
    loss_results: pd.DataFrame,
    components: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset_name, dataset_group in loss_results.groupby("dataset", sort=True):
        for component in components:
            values = dataset_group[component].astype(float)
            min_value = values.min()
            max_value = values.max()
            denom = max_value - min_value

            for _, row in dataset_group.iterrows():
                raw_value = float(row[component])
                normalized = 0.0 if denom == 0 else (raw_value - min_value) / denom
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": row["model"],
                        "component": component,
                        "raw_value": raw_value,
                        "normalized_loss": float(normalized),
                    }
                )

    component_df = pd.DataFrame(rows)
    score_df = (
        component_df.groupby(["dataset", "model"], as_index=False)
        .agg(normalized_loss=("normalized_loss", "mean"))
    )
    score_df["performance"] = 1.0 - score_df["normalized_loss"]
    return component_df, score_df


def build_cd_performance(
    data_dir: str,
    predictions_dir: str,
    models: list[str] | None,
    ignored_components: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loss_config = DEFAULT_LOSS_CONFIG.copy()
    truth_by_dataset = load_truth_by_dataset(data_dir)
    prediction_files = list_prediction_files(predictions_dir, models)
    loss_results = evaluate_models(truth_by_dataset, prediction_files, loss_config)

    components = [
        component
        for component in active_components(loss_results)
        if component not in ignored_components
    ]
    if not components:
        raise ValueError("Nenhum componente ativo para montar o CD diagram.")

    component_df, score_df = minmax_normalize_components(loss_results, components)
    finite_by_model = score_df.groupby("model")["normalized_loss"].apply(lambda s: np.isfinite(s).all())
    dropped_models = sorted(finite_by_model[~finite_by_model].index.tolist())
    if dropped_models:
        print(
            "Aviso: removendo modelos com InventoryDemandLoss não finita no CD diagram: "
            + ", ".join(dropped_models)
        )
        valid_models = set(finite_by_model[finite_by_model].index)
        component_df = component_df[component_df["model"].isin(valid_models)].copy()
        score_df = score_df[score_df["model"].isin(valid_models)].copy()

    df_perf = score_df.rename(
        columns={
            "dataset": "dataset_name",
            "model": "classifier_name",
            "performance": "accuracy",
        }
    )[["classifier_name", "dataset_name", "accuracy", "normalized_loss"]]

    ranks = (
        df_perf.assign(rank=df_perf.groupby("dataset_name")["accuracy"].rank(ascending=False, method="average"))
        .groupby("classifier_name", as_index=False)
        .agg(avg_rank=("rank", "mean"), mean_normalized_loss=("normalized_loss", "mean"))
        .sort_values("avg_rank")
    )
    return df_perf, component_df, score_df, ranks


def build_mae_cd_performance(
    data_dir: str,
    predictions_dir: str,
    models: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth_by_dataset = load_truth_by_dataset(data_dir)
    prediction_files = list_prediction_files(predictions_dir, models)

    series_rows = []
    score_rows = []

    for pred_path in prediction_files:
        model_name = model_name_from_path(pred_path)
        pred_df = pd.read_parquet(pred_path).copy()
        required_cols = {"dataset", "id", "timestamp", "q50"}
        missing_cols = required_cols - set(pred_df.columns)
        if missing_cols:
            print(f"Aviso: ignorando {model_name}; colunas ausentes: {sorted(missing_cols)}")
            continue

        pred_df["id"] = pred_df["id"].astype(str)
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        for dataset_name, df_true in truth_by_dataset.items():
            df_pred = pred_df[pred_df["dataset"] == dataset_name].copy()
            if df_pred.empty:
                continue

            merged = df_true.merge(
                df_pred[["id", "timestamp", "q50"]],
                on=["id", "timestamp"],
                how="inner",
            )
            if merged.empty:
                continue

            merged["abs_error"] = (merged["target"] - merged["q50"]).abs()
            series_mae = (
                merged.groupby("id", as_index=False)
                .agg(mae=("abs_error", "mean"), n_points=("abs_error", "size"))
            )
            series_mae["dataset"] = dataset_name
            series_mae["model"] = model_name
            series_rows.append(series_mae[["dataset", "model", "id", "n_points", "mae"]])

            score_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "mae": float(series_mae["mae"].mean()),
                    "matched_rows": int(len(merged)),
                    "n_series": int(series_mae["id"].nunique()),
                }
            )

    if not score_rows:
        raise ValueError("Nenhuma combinação válida de dataset e modelo foi avaliada com MAE.")

    series_df = pd.concat(series_rows, ignore_index=True) if series_rows else pd.DataFrame()
    score_df = pd.DataFrame(score_rows)

    finite_by_model = score_df.groupby("model")["mae"].apply(lambda s: np.isfinite(s).all())
    dropped_models = sorted(finite_by_model[~finite_by_model].index.tolist())
    if dropped_models:
        print(
            "Aviso: removendo modelos com MAE não finito no CD diagram: "
            + ", ".join(dropped_models)
        )
        valid_models = set(finite_by_model[finite_by_model].index)
        series_df = series_df[series_df["model"].isin(valid_models)].copy()
        score_df = score_df[score_df["model"].isin(valid_models)].copy()

    score_df["performance"] = -score_df["mae"]
    df_perf = score_df.rename(
        columns={
            "dataset": "dataset_name",
            "model": "classifier_name",
            "performance": "accuracy",
            "mae": "normalized_loss",
        }
    )[["classifier_name", "dataset_name", "accuracy", "normalized_loss"]]

    ranks = (
        df_perf.assign(rank=df_perf.groupby("dataset_name")["accuracy"].rank(ascending=False, method="average"))
        .groupby("classifier_name", as_index=False)
        .agg(avg_rank=("rank", "mean"), mean_mae=("normalized_loss", "mean"))
        .sort_values("avg_rank")
    )
    score_df = score_df.rename(columns={"mae": "normalized_loss"})
    return df_perf, series_df, score_df, ranks


def save_cd_tables(
    output_dir: str,
    df_perf: pd.DataFrame,
    component_df: pd.DataFrame,
    score_df: pd.DataFrame,
    ranks: pd.DataFrame,
    prefix: str,
    detail_suffix: str = "components_normalized",
) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    perf_path = output_path / f"{prefix}_performance.csv"
    component_path = output_path / f"{prefix}_{detail_suffix}.csv"
    score_path = output_path / f"{prefix}_scores.csv"
    ranks_path = output_path / f"{prefix}_ranks.csv"

    df_perf.to_csv(perf_path, index=False)
    component_df.to_csv(component_path, index=False)
    score_df.to_csv(score_path, index=False)
    ranks.to_csv(ranks_path, index=False)
    return perf_path, component_path, score_path, ranks_path


def split_csv_or_space(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out or None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gera CD diagram para regressão usando InventoryDemandLoss com "
            "componentes normalizados por dataset entre 0 e 1."
        )
    )
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--predictions-dir", type=str, default="data/predictions")
    parser.add_argument("--output-dir", type=str, default="data/analysis/cd_diagram")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--posthoc", choices=["nemenyi", "wilcoxon"], default="nemenyi")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--score",
        choices=["inventory_loss", "mae"],
        default="inventory_loss",
        help="Score usado no CD diagram. inventory_loss normaliza componentes; mae usa MAE médio por SKU.",
    )
    parser.add_argument("--prefix", type=str, default="inventory_loss_cd")
    parser.add_argument(
        "--ignore-components",
        nargs="+",
        default=sorted(IGNORED_COMPONENTS),
        help="Componentes ignorados no score normalizado. Default: int neg total tv.",
    )
    args = parser.parse_args()

    models = split_csv_or_space(args.models)
    ignored_components = set(split_csv_or_space(args.ignore_components) or [])
    ignored_components.add("total")

    prefix = args.prefix
    if args.score == "mae" and prefix == "inventory_loss_cd":
        prefix = "mae_cd"

    if args.score == "mae":
        df_perf, component_df, score_df, ranks = build_mae_cd_performance(
            data_dir=args.data_dir,
            predictions_dir=args.predictions_dir,
            models=models,
        )
        title = "MAE CD Diagram"
    else:
        df_perf, component_df, score_df, ranks = build_cd_performance(
            data_dir=args.data_dir,
            predictions_dir=args.predictions_dir,
            models=models,
            ignored_components=ignored_components,
        )
        title = "InventoryDemandLoss CD Diagram"

    detail_suffix = "series_mae" if args.score == "mae" else "components_normalized"
    table_paths = save_cd_tables(
        args.output_dir,
        df_perf,
        component_df,
        score_df,
        ranks,
        prefix,
        detail_suffix=detail_suffix,
    )

    output_path = Path(args.output_dir) / f"{prefix}.png"
    draw_cd_diagram(
        output_path=str(output_path),
        df_perf=df_perf,
        alpha=args.alpha,
        title=title,
        labels=args.labels,
        posthoc=args.posthoc,
        verbose=args.verbose,
    )

    if args.score == "mae":
        print("Score usado: MAE médio por SKU em cada dataset")
    else:
        print("Componentes usados:")
        print(", ".join(sorted(component_df["component"].unique())))
    print("\nRanks médios:")
    print(ranks.to_string(index=False))
    print("\nArquivos salvos:")
    for path in table_paths:
        print(f"  {path}")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()


# df_perf = pd.read_csv('example.csv', index_col=False)

# draw_cd_diagram(df_perf=df_perf, title='Accuracy', labels=True)
