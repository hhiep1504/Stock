#!/usr/bin/env python3
"""Run final walk-forward benchmarks with Optuna-tuned model parameters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking import run_tuned_walk_forward_benchmark
from src.config import ExperimentConfig


DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "tuned_baseline4_benchmark_manifest.json"


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.device is not None:
        config.runtime.device = args.device
        config.benchmark.device = args.device
    if args.gnn_epochs is not None:
        config.benchmark.gnn_epochs = args.gnn_epochs
    if args.baseline_epochs is not None:
        config.benchmark.baseline_epochs = args.baseline_epochs
    if args.benchmark_runs is not None:
        config.benchmark.n_runs = args.benchmark_runs
    config.benchmark.protocol = "walk_forward"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tuned walk-forward benchmarks")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--gnn-epochs", type=int, default=None)
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--benchmark-runs", type=int, default=None)
    parser.add_argument("--only-run", action="append", default=None, help="Run only matching manifest run name(s).")
    parser.add_argument("--only-model", action="append", default=None, help="Run only matching model name(s).")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.gnn_epochs = 10 if args.gnn_epochs is None else args.gnn_epochs
        args.baseline_epochs = 10 if args.baseline_epochs is None else args.baseline_epochs
        args.benchmark_runs = 1 if args.benchmark_runs is None else args.benchmark_runs

    manifest = _load_manifest(_resolve_path(args.manifest))
    output_root = _resolve_path(args.output_dir or manifest.get("output_dir", "logs/tuned_feature_benchmarks"))
    output_root.mkdir(parents=True, exist_ok=True)

    only_runs = set(args.only_run or [])
    only_models = set(args.only_model or [])
    combined_rows = []
    for run_spec in manifest["runs"]:
        run_name = run_spec["name"]
        if only_runs and run_name not in only_runs:
            continue
        config = ExperimentConfig.from_json(_resolve_path(run_spec["config"]))
        _apply_overrides(config, args)
        model_specs = run_spec["models"]
        if only_models:
            model_specs = {
                model_name: params_path
                for model_name, params_path in model_specs.items()
                if model_name in only_models
            }
            if not model_specs:
                continue
        tuned_files = {
            model_name: _resolve_path(params_path)
            for model_name, params_path in model_specs.items()
        }
        run_dir = output_root / run_name
        print(f"[tuned-benchmark] run={run_name}")
        print(f"[tuned-benchmark] config={_resolve_path(run_spec['config'])}")
        print(f"[tuned-benchmark] output={run_dir}")
        result = run_tuned_walk_forward_benchmark(
            config=config,
            tuned_param_files=tuned_files,
            run_dir=run_dir,
            protocol_label="tuned_walk_forward",
        )
        result.insert(0, "run", run_name)
        combined_rows.append(result)
        print(result.to_string(index=False))

    if not combined_rows:
        raise ValueError("No manifest runs/models matched the requested filters.")

    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = output_root / "combined_tuned_benchmark_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[tuned-benchmark] combined={combined_path}")
    print(combined.sort_values(by="mae_interval_mean").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
