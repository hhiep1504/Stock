#!/usr/bin/env python3
"""Run common-period feature-set benchmarks for the deep models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking import (
    _common_contiguous_periods,
    _restrict_prepared_to_periods,
    run_common_period_feature_benchmark,
)
from src.config import ExperimentConfig
from src.pipeline import prepare_dataset

DEFAULT_CONFIGS = [
    PROJECT_ROOT / "configs" / "feature_benchmark_baseline4.json",
    PROJECT_ROOT / "configs" / "feature_benchmark_baseline_plus_sma12.json",
    PROJECT_ROOT / "configs" / "feature_benchmark_screened_with_hurst.json",
]


def _load_configs(paths: list[Path]) -> list[ExperimentConfig]:
    return [ExperimentConfig.from_json(path) for path in paths]


def _apply_overrides(configs: list[ExperimentConfig], args: argparse.Namespace) -> None:
    for config in configs:
        if args.device is not None:
            config.benchmark.device = args.device
            config.runtime.device = args.device
        if args.gnn_epochs is not None:
            config.benchmark.gnn_epochs = args.gnn_epochs
        if args.baseline_epochs is not None:
            config.benchmark.baseline_epochs = args.baseline_epochs
        if args.benchmark_runs is not None:
            config.benchmark.n_runs = args.benchmark_runs
        config.benchmark.protocol = "walk_forward"


def _dry_run(configs: list[ExperimentConfig]) -> None:
    prepared_items = [(config, prepare_dataset(config)) for config in configs]
    common_periods = _common_contiguous_periods([prepared for _, prepared in prepared_items])
    print("Common-period dry run")
    print("=" * 80)
    print(f"common_periods={len(common_periods)}")
    print(f"common_start={common_periods[0] if common_periods else None}")
    print(f"common_end={common_periods[-1] if common_periods else None}")
    print()
    for config, prepared in prepared_items:
        restricted = _restrict_prepared_to_periods(prepared, common_periods, config.data.window_size)
        print(
            f"{config.data.feature_set}: "
            f"periods={len(restricted.valid_indices_map)} | "
            f"sequences={len(restricted.sequence_templates)} | "
            f"features={len(restricted.feature_names)} | "
            f"names={','.join(restricted.feature_names)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common-period feature benchmark runner")
    parser.add_argument("--configs", nargs="+", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "feature_benchmarks_common")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--gnn-epochs", type=int, default=None)
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--benchmark-runs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = _load_configs(args.configs)
    if args.quick:
        args.gnn_epochs = 10 if args.gnn_epochs is None else args.gnn_epochs
        args.baseline_epochs = 10 if args.baseline_epochs is None else args.baseline_epochs
        args.benchmark_runs = 1 if args.benchmark_runs is None else args.benchmark_runs

    _apply_overrides(configs, args)

    if args.dry_run:
        _dry_run(configs)
        return 0

    result = run_common_period_feature_benchmark(configs, output_dir=args.output_dir)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
