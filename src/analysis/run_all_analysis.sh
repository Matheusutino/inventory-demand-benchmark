#!/usr/bin/env bash

set -euo pipefail

DATASET=""
DATA_DIR="data/datasets"
PREDICTIONS_DIR="data/predictions"
ANALYSIS_DIR="data/analysis"
METRICS_DIR=""
PYTHON_BIN="${PYTHON_BIN:-}"
FAIL_FAST=0
EXCLUDES=()

usage() {
  cat <<'EOF'
Usage:
  src/analysis/run_all_analysis.sh [options]

Options:
  --dataset NAME              Dataset para plot_dataset_predictions.py.
  --data-dir DIR             Diretório dos datasets reais. Default: data/datasets
  --predictions-dir DIR      Diretório das predições parquet. Default: data/predictions
  --analysis-dir DIR         Diretório das métricas/análises. Default: data/analysis
  --metrics-dir DIR          Default: <analysis-dir>/accuracy/metrics
  --plots-dir DIR            Alias de compatibilidade para --analysis-dir.
  --python BIN               Interpretador Python. Default: python
  --fail-fast                Para na primeira falha.
  --exclude SCRIPT...        Pula scripts pelo nome, ex: evaluate_inventory_framework.py
  -h, --help                 Mostra esta ajuda.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:-}"
      shift 2
      ;;
    --predictions-dir)
      PREDICTIONS_DIR="${2:-}"
      shift 2
      ;;
    --analysis-dir)
      ANALYSIS_DIR="${2:-}"
      shift 2
      ;;
    --metrics-dir)
      METRICS_DIR="${2:-}"
      shift 2
      ;;
    --plots-dir)
      ANALYSIS_DIR="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --exclude)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EXCLUDES+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Argumento desconhecido: $1"
      usage
      exit 1
      ;;
  esac
done

METRICS_DIR="${METRICS_DIR:-$ANALYSIS_DIR/accuracy/metrics}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Erro: nenhum interpretador Python encontrado. Use --python /caminho/para/python."
    exit 127
  fi
fi

should_skip() {
  local script_name="$1"
  for excluded in "${EXCLUDES[@]}"; do
    if [[ "$excluded" == "$script_name" ]]; then
      return 0
    fi
  done
  return 1
}

run_cmd() {
  local label="$1"
  shift

  echo
  echo "================================================================================"
  echo "RODANDO: $label"
  echo "================================================================================"

  if "$@"; then
    echo "✓ OK: $label"
  else
    local exit_code=$?
    echo "✗ Falhou: $label (exit code $exit_code)"
    if [[ $FAIL_FAST -eq 1 ]]; then
      exit "$exit_code"
    fi
    FAILURES+=("$label:$exit_code")
  fi
}

FAILURES=()

if ! should_skip "analyze_predictions.py"; then
  run_cmd "src.analysis.analyze_predictions" \
    "$PYTHON_BIN" -m src.analysis.analyze_predictions \
      --predictions-dir "$PREDICTIONS_DIR" \
      --data-dir "$DATA_DIR" \
      --output-dir "$METRICS_DIR"
fi

if ! should_skip "describe_datasets.py"; then
  run_cmd "src.analysis.describe_datasets" \
    "$PYTHON_BIN" -m src.analysis.describe_datasets \
      --data-dir "$DATA_DIR" \
      --output-dir "$ANALYSIS_DIR/dataset"
fi

if ! should_skip "plot_dataset_overview.py"; then
  run_cmd "src.analysis.plot_dataset_overview" \
    "$PYTHON_BIN" -m src.analysis.plot_dataset_overview \
      --data-dir "$DATA_DIR" \
      --output-dir "$ANALYSIS_DIR/dataset/overview"
fi

if ! should_skip "plot_aggregate_metrics.py"; then
  run_cmd "src.analysis.plot_aggregate_metrics" \
    "$PYTHON_BIN" -m src.analysis.plot_aggregate_metrics \
      --metrics-dir "$METRICS_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --data-dir "$DATA_DIR" \
      --output-dir "$ANALYSIS_DIR/accuracy/aggregate"
fi

if ! should_skip "plot_item_sum_mae.py"; then
  run_cmd "src.analysis.plot_item_sum_mae" \
    "$PYTHON_BIN" -m src.analysis.plot_item_sum_mae \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/accuracy/item_sum_mae"
fi

if ! should_skip "plot_item_mae_by_horizon.py"; then
  run_cmd "src.analysis.plot_item_mae_by_horizon" \
    "$PYTHON_BIN" -m src.analysis.plot_item_mae_by_horizon \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/accuracy/item_mae_by_horizon"
fi

if ! should_skip "analyze_sku_winners.py"; then
  run_cmd "src.analysis.analyze_sku_winners" \
    "$PYTHON_BIN" -m src.analysis.analyze_sku_winners \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/accuracy/sku_winners"
fi

if ! should_skip "plot_horizon_degradation.py"; then
  run_cmd "src.analysis.plot_horizon_degradation" \
    "$PYTHON_BIN" -m src.analysis.plot_horizon_degradation \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/diagnostics/horizon_degradation"
fi

if ! should_skip "plot_item_mae_vs_degradation.py"; then
  run_cmd "src.analysis.plot_item_mae_vs_degradation" \
    "$PYTHON_BIN" -m src.analysis.plot_item_mae_vs_degradation \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/diagnostics/mae_vs_horizon"
fi

if ! should_skip "evaluate_inventory_framework.py"; then
  run_cmd "src.analysis.evaluate_inventory_framework" \
    "$PYTHON_BIN" -m src.analysis.evaluate_inventory_framework \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/framework/components" \
      --wins-dir "$ANALYSIS_DIR/framework/wins"
fi

if ! should_skip "plot_inventory_component_rank_heatmap.py"; then
  run_cmd "src.analysis.plot_inventory_component_rank_heatmap" \
    "$PYTHON_BIN" -m src.analysis.plot_inventory_component_rank_heatmap \
      --input "$ANALYSIS_DIR/framework/components/component_values.csv" \
      --output-dir "$ANALYSIS_DIR/framework/ranks"
fi

if ! should_skip "analyze_framework_sensitivity.py"; then
  run_cmd "src.analysis.analyze_framework_sensitivity" \
    "$PYTHON_BIN" -m src.analysis.analyze_framework_sensitivity \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/framework/sensitivity"
fi

if ! should_skip "analyze_component_concordance.py"; then
  run_cmd "src.analysis.analyze_component_concordance" \
    "$PYTHON_BIN" -m src.analysis.analyze_component_concordance \
      --input "$ANALYSIS_DIR/framework/components/component_values.csv" \
      --output-dir "$ANALYSIS_DIR/framework/concordance"
fi

if ! should_skip "cd_diagram.py"; then
  run_cmd "src.analysis.cd_diagram_mae" \
    "$PYTHON_BIN" -m src.analysis.cd_diagram \
      --score mae \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/accuracy/cd_diagram" \
      --labels
fi

if [[ -n "$DATASET" ]] && ! should_skip "plot_dataset_predictions.py"; then
  run_cmd "src.analysis.plot_dataset_predictions" \
    "$PYTHON_BIN" -m src.analysis.plot_dataset_predictions \
      --dataset "$DATASET" \
      --data-dir "$DATA_DIR" \
      --predictions-dir "$PREDICTIONS_DIR" \
      --output-dir "$ANALYSIS_DIR/predictions"
elif [[ -z "$DATASET" ]] && ! should_skip "plot_dataset_predictions.py"; then
  echo "- Pulando plot_dataset_predictions.py: requer --dataset"
fi

echo
echo "================================================================================"
echo "RESUMO FINAL"
echo "================================================================================"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  for failure in "${FAILURES[@]}"; do
    echo "✗ $failure"
  done
  exit 1
fi

echo "✓ Todos os scripts executados com sucesso."
