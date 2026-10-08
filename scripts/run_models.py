#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    venv: str
    module: str
    base_args: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    training: bool = False
    gated: bool = False
    supports_device: bool = False
    description: str = ""
    extra_by_name: dict[str, tuple[str, ...]] = field(default_factory=dict)


MODEL_SPECS = [
    ModelSpec(
        name="chronos2",
        venv="chronos",
        module="src.models.chronos2",
        aliases=("chronos",),
        supports_device=True,
        description="Amazon Chronos-2 zero-shot.",
    ),
    ModelSpec(
        name="chronos2_finetuned",
        venv="chronos",
        module="src.models.chronos2_finetuned",
        aliases=("chronos_finetuned", "chronos-ft"),
        training=True,
        supports_device=True,
        description="Chronos-2 com fine-tuning LoRA.",
    ),
    ModelSpec(
        name="granite_ttm",
        venv="granite_ttm",
        module="src.models.granite_ttm",
        aliases=("granite", "ttm"),
        supports_device=True,
        description="IBM Granite TinyTimeMixer.",
    ),
    ModelSpec(
        name="moirai2_small",
        venv="morais",
        module="src.models.morais",
        base_args=("--model-type", "moirai2", "--model-size", "small"),
        aliases=("moirai", "morais", "moirai2"),
        description="Salesforce Moirai2 small.",
    ),
    ModelSpec(
        name="timesfm",
        venv="timesfm",
        module="src.models.timesfm",
        supports_device=True,
        description="Google TimesFM.",
    ),
    ModelSpec(
        name="tabpfn_ts",
        venv="tabpfn_time_series",
        module="src.models.tabpfn_time_series",
        aliases=("tabpfn", "tabpfn_time_series"),
        gated=True,
        description="Prior Labs TabPFN-TS.",
    ),
    ModelSpec(
        name="tirex",
        venv="tirex",
        module="src.models.tirex",
        aliases=("tirex_ts",),
        supports_device=True,
        description="NX-AI TiRex zero-shot.",
    ),
    ModelSpec(
        name="sundial_base",
        venv="timer_sundial",
        module="src.models.timer_sundial",
        base_args=("--model-path", "thuml/sundial-base-128m"),
        aliases=("sundial", "timer_sundial"),
        description="THUML Sundial base 128M.",
    ),
    ModelSpec(
        name="timer_base",
        venv="timer_sundial",
        module="src.models.timer_xl",
        base_args=("--model-path", "thuml/timer-base-84m"),
        aliases=("timer", "timer_xl"),
        supports_device=True,
        description="THUML Timer base 84M.",
    ),
    ModelSpec(
        name="statsforecast_intermittent",
        venv="darts",
        module="src.models.intermittent_baselines",
        base_args=("--model", "all"),
        aliases=("intermittent",),
        description="StatsForecast Croston, SBA e TSB por SKU.",
        extra_by_name={
            "statsforecast_croston": ("--model", "croston"),
            "statsforecast_sba": ("--model", "sba"),
            "statsforecast_tsb": ("--model", "tsb"),
            "croston": ("--model", "croston"),
            "sba": ("--model", "sba"),
            "tsb": ("--model", "tsb"),
        },
    ),
    ModelSpec(
        name="darts_classic",
        venv="darts",
        module="src.models.darts_classic",
        base_args=("--model", "all"),
        aliases=("darts", "darts_global"),
        description="Darts modelos globais/classicos.",
        extra_by_name={
            "darts_linear": ("--model", "linear"),
            "darts_randomforest": ("--model", "randomforest"),
            "darts_lightgbm": ("--model", "lightgbm"),
            "darts_xgboost": ("--model", "xgboost"),
            "darts_global_naive_aggregate": ("--model", "global_naive_aggregate"),
            "darts_global_naive_drift": ("--model", "global_naive_drift"),
            "darts_global_naive_seasonal": ("--model", "global_naive_seasonal"),
        },
    ),
    ModelSpec(
        name="darts_univariate",
        venv="darts",
        module="src.models.darts_univariate_baselines",
        base_args=("--model", "all"),
        aliases=("darts_uni", "darts_baselines"),
        description="Darts baselines univariados.",
        extra_by_name={
            "darts_naive_moving_average": ("--model", "naive_moving_average"),
            "darts_arima": ("--model", "arima"),
            "darts_prophet": ("--model", "prophet"),
        },
    ),
]


def build_lookup() -> dict[str, tuple[ModelSpec, tuple[str, ...]]]:
    lookup: dict[str, tuple[ModelSpec, tuple[str, ...]]] = {}
    for spec in MODEL_SPECS:
        keys = (spec.name, *spec.aliases)
        for key in keys:
            lookup[key] = (spec, ())
        for key, extra_args in spec.extra_by_name.items():
            lookup[key] = (spec, extra_args)
    return lookup


MODEL_LOOKUP = build_lookup()


def discover_datasets(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted(
        train_file.name.removesuffix("_train.parquet")
        for train_file in data_dir.glob("*_train.parquet")
        if train_file.with_name(train_file.name.replace("_train.parquet", "_test.parquet")).exists()
    )


def split_csv_or_space(values: list[str] | None) -> list[str]:
    if not values:
        return []

    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def selected_models(
    model_args: list[str],
    include_training: bool,
    include_gated: bool,
) -> list[tuple[ModelSpec, tuple[str, ...], str]]:
    requested = split_csv_or_space(model_args)
    if not requested or requested == ["all"]:
        requested = [
            spec.name
            for spec in MODEL_SPECS
            if (include_training or not spec.training) and (include_gated or not spec.gated)
        ]

    selected: list[tuple[ModelSpec, tuple[str, ...], str]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unknown: list[str] = []
    skipped_training: list[str] = []

    for name in requested:
        if name == "all":
            for spec in MODEL_SPECS:
                if spec.training and not include_training:
                    continue
                if spec.gated and not include_gated:
                    continue
                key = (spec.name, spec.base_args)
                if key not in seen:
                    selected.append((spec, (), spec.name))
                    seen.add(key)
            continue

        item = MODEL_LOOKUP.get(name)
        if item is None:
            unknown.append(name)
            continue

        spec, extra_args = item
        if spec.training and not include_training:
            skipped_training.append(name)
            continue

        key = (spec.name, extra_args)
        if key not in seen:
            selected.append((spec, extra_args, name))
            seen.add(key)

    if unknown:
        raise SystemExit(f"Modelo(s) desconhecido(s): {', '.join(unknown)}")
    if skipped_training:
        raise SystemExit(
            "Modelo(s) de treinamento exigem --include-training: "
            + ", ".join(skipped_training)
        )
    return selected


def resolve_python(spec: ModelSpec) -> Path:
    python_path = ROOT_DIR / ".venvs" / spec.venv / "bin" / "python"
    if not python_path.exists():
        raise SystemExit(
            f"Venv ausente para {spec.name}: {python_path}\n"
            f"Crie com: scripts/setup_venvs.sh {spec.venv}"
        )
    return python_path


def build_command(
    spec: ModelSpec,
    extra_args: tuple[str, ...],
    args: argparse.Namespace,
) -> list[str]:
    model_args = extra_args if extra_args else spec.base_args
    cmd = [
        str(resolve_python(spec)),
        "-m",
        spec.module,
        *model_args,
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
    ]

    if args.device and spec.supports_device:
        cmd.extend(["--device", args.device])

    if args.datasets:
        cmd.append("--datasets")
        cmd.extend(args.datasets)

    if spec.venv == "darts" and args.darts_validation_metric:
        cmd.extend(["--validation-metric", args.darts_validation_metric])

    cmd.extend(args.extra_args)
    return cmd


def print_available(data_dir: Path) -> None:
    print("Modelos:")
    for spec in MODEL_SPECS:
        markers = []
        if spec.training:
            markers.append("training")
        if spec.gated:
            markers.append("gated")
        marker = f" [{' '.join(markers)}]" if markers else ""
        aliases = f" aliases: {', '.join(spec.aliases)}" if spec.aliases else ""
        print(f"  {spec.name}{marker} - {spec.description}{aliases}")
        for name in spec.extra_by_name:
            print(f"    {name}")

    datasets = discover_datasets(data_dir)
    print("\nDatasets:")
    if datasets:
        for dataset in datasets:
            print(f"  {dataset}")
    else:
        print(f"  Nenhum *_train.parquet/*_test.parquet encontrado em {data_dir}")


def run(args: argparse.Namespace) -> int:
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    args.datasets = split_csv_or_space(args.datasets)
    args.extra_args = args.extra_args or []

    if args.list:
        print_available(args.data_dir)
        return 0

    models = selected_models(args.models, args.include_training, args.include_gated)
    if args.datasets:
        available = set(discover_datasets(args.data_dir))
        missing = [dataset for dataset in args.datasets if dataset not in available]
        if missing:
            raise SystemExit(
                "Dataset(s) não encontrado(s): "
                + ", ".join(missing)
                + f"\nUse --list para ver os nomes disponíveis em {args.data_dir}."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []
    for index, (spec, extra_args, requested_name) in enumerate(models, start=1):
        label = requested_name if requested_name != spec.name else spec.name
        if spec.gated and "TABPFN_TOKEN" not in os.environ:
            print(
                f"\nAviso: {label} requer TABPFN_TOKEN/licença aceita. "
                "Se falhar, abra https://ux.priorlabs.ai, aceite a licença e exporte TABPFN_TOKEN.",
                flush=True,
            )
        cmd = build_command(spec, extra_args, args)
        print("\n" + "#" * 80)
        print(f"# [{index}/{len(models)}] {label}")
        print("# " + shlex.join(cmd))
        print("#" * 80 + "\n")
        sys.stdout.flush()

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=ROOT_DIR, check=False)
        if completed.returncode != 0:
            failures.append((label, completed.returncode))
            if args.stop_on_error:
                break

    if failures:
        print("\nFalhas:")
        for label, code in failures:
            print(f"  {label}: exit code {code}")
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa modelos de forecasting em todos ou em datasets específicos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemplos:
  scripts/run_models.py --list
  scripts/run_models.py
  scripts/run_models.py --models granite_ttm --datasets Filtros_h3
  scripts/run_models.py --models granite_ttm,timesfm --datasets "Filtros_h3" "Placas de Controle_h6"
  scripts/run_models.py --models darts_linear --datasets Filtros_h3 -- --lags 12
  scripts/run_models.py --models darts_linear --datasets Filtros_h3 --darts-validation-metric rmse
  scripts/run_models.py --models tabpfn_ts --include-gated --datasets Filtros_h3
  scripts/run_models.py --models chronos2_finetuned --include-training --datasets Filtros_h3 -- --num-steps 200
""",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Modelos/aliases separados por espaço ou vírgula. Use --list para ver opções.",
    )
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets específicos.")
    parser.add_argument("--data-dir", default="data/datasets", help="Diretório dos parquet splits.")
    parser.add_argument("--output-dir", default="data/predictions", help="Diretório de saída.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--darts-validation-metric",
        choices=["mae", "mse", "rmse", "mape"],
        default="mae",
        help="Métrica usada apenas no grid search de validação dos modelos Darts.",
    )
    parser.add_argument("--include-training", action="store_true")
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Inclui modelos que exigem aceite/token externo, como TabPFN-TS.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="Lista modelos e datasets.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Argumentos extras após -- são repassados para cada script executado.",
    )

    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
