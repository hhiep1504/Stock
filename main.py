"""Command-line entrypoint for training, tuning, and benchmarking."""

from __future__ import annotations

import argparse
from typing import Any

from src.config import ExperimentConfig, default_experiment_config


def _load_config(path: str | None) -> ExperimentConfig:
    return ExperimentConfig.from_json(path) if path else default_experiment_config()


def _print_tuning_result(result: Any) -> None:
    from src.tuning import MultiTuningResult

    if isinstance(result, MultiTuningResult):
        print("Tuning summary")
        print("=" * 70)
        for family, family_result in result.runs.items():
            print(f"{family}: best_value={family_result.best_value:.6f} | trial={family_result.best_trial} | study={family_result.study_name}")
            print(f"  artifact_dir={family_result.artifact_dir}")
        return

    print("Tuning summary")
    print("=" * 70)
    print(f"study={result.study_name}")
    print(f"best_value={result.best_value:.6f}")
    print(f"best_trial={result.best_trial}")
    print(f"artifact_dir={result.artifact_dir}")


def _print_tsne_result(result: Any) -> None:
    print("t-SNE inspection")
    print("=" * 70)
    print(f"output_dir={result.output_dir}")
    print(f"image_path={result.image_path}")
    print(f"csv_path={result.csv_path}")
    print(f"num_sequences={result.num_sequences}")
    print(f"perplexity={result.perplexity:.2f}")


def _print_graph_result(result: Any) -> None:
    print("Graph inspection")
    print("=" * 70)
    print(f"output_dir={result.output_dir}")
    print(f"summary_csv={result.summary_csv_path}")
    print(f"summary_json={result.summary_json_path}")
    print(f"node_summary_csv={result.node_summary_csv_path}")
    print(f"snapshot_indices={result.snapshot_indices}")
    for image_path in result.image_paths:
        print(f"image_path={image_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAT-LSTM training and benchmark entrypoint")
    parser.add_argument("--config", type=str, default=None, help="Path to an experiment JSON config")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices={
            "train",
            "tune",
            "benchmark",
            "benchmark-cv",
            "benchmark-walkforward",
            "visualize-tsne",
            "visualize-graph",
            "final-baseline",
        },
        help="Execution mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare data and report counts only")
    parser.add_argument("--trials", type=int, default=None, help="Override number of Optuna trials")
    parser.add_argument(
        "--max-epochs-per-trial",
        type=int,
        default=None,
        help="Cap the number of epochs per Optuna trial",
    )
    parser.add_argument(
        "--prune-after-epochs",
        type=int,
        default=None,
        help="Start pruning checks after this many epochs",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="all",
        help="Comma-separated model families to tune, or 'all'",
    )
    parser.add_argument(
        "--tuned-params",
        type=str,
        default=None,
        help="Path to best_params.json for final baseline training",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Optional window_size override for this run",
    )
    parser.add_argument("--gnn-epochs", type=int, default=None, help="Override benchmark GNN epochs")
    parser.add_argument("--baseline-epochs", type=int, default=None, help="Override benchmark baseline epochs")
    parser.add_argument("--benchmark-runs", type=int, default=None, help="Override benchmark runs per fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    if args.window_size is not None:
        config.data.window_size = args.window_size
    if args.gnn_epochs is not None:
        config.benchmark.gnn_epochs = args.gnn_epochs
    if args.baseline_epochs is not None:
        config.benchmark.baseline_epochs = args.baseline_epochs
    if args.benchmark_runs is not None:
        config.benchmark.n_runs = args.benchmark_runs
    config.ensure_output_directories()

    if args.mode == "train":
        from src.pipeline import run_training_experiment

        result = run_training_experiment(config, dry_run=args.dry_run)
        print(result)
        return

    if args.mode == "tune":
        from src.tuning import run_optuna_tuning

        families = ["all"] if args.families.strip().lower() == "all" else [item.strip() for item in args.families.split(",") if item.strip()]
        result = run_optuna_tuning(
            config,
            n_trials=args.trials,
            max_epochs_per_trial=args.max_epochs_per_trial,
            prune_after_epochs=args.prune_after_epochs,
            families=families,
        )
        _print_tuning_result(result)
        return

    if args.mode == "benchmark":
        from src.benchmarking import run_standard_benchmark

        result = run_standard_benchmark(config)
        print(result.to_string(index=False))
        return

    if args.mode == "visualize-tsne":
        from src.inspection import run_pretraining_tsne

        result = run_pretraining_tsne(config)
        _print_tsne_result(result)
        return

    if args.mode == "visualize-graph":
        from src.inspection import run_graph_visualization

        result = run_graph_visualization(config)
        _print_graph_result(result)
        return

    if args.mode == "final-baseline":
        from src.benchmarking import run_final_neural_baseline

        families = [item.strip() for item in args.families.split(",") if item.strip()]
        if len(families) != 1:
            raise ValueError("final-baseline mode expects exactly one family in --families")
        if not args.tuned_params:
            raise ValueError("final-baseline mode requires --tuned-params")
        result = run_final_neural_baseline(
            config=config,
            family=families[0],
            tuned_params_file=args.tuned_params,
        )
        print(result)
        return

    if args.mode == "benchmark-walkforward" or config.benchmark.protocol == "walk_forward":
        from src.benchmarking import run_walk_forward_benchmark

        result = run_walk_forward_benchmark(config)
    else:
        from src.benchmarking import run_cross_validation_benchmark

        result = run_cross_validation_benchmark(config)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
