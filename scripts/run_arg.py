#!/usr/bin/env python3
"""Reproducible experiment launcher for final baselines and graph-model runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default_experiment.json"
RUNTIME_CONFIG_DIR = PROJECT_ROOT / "outputs" / "runtime_configs"

FINAL_BASELINE_PRESETS: dict[str, dict[str, Any]] = {
    "lstm": {
        "window_size": 4,
        "graph_mode": "static",
        "tuned_params": PROJECT_ROOT / "logs" / "optuna_lstm_20260428_213706_w4" / "best_params.json",
    },
    "gru": {
        "window_size": 4,
        "graph_mode": "static",
        "tuned_params": PROJECT_ROOT / "logs" / "optuna_gru_20260428_213102_w4" / "best_params.json",
    },
    "cnn_lstm": {
        "window_size": 8,
        "graph_mode": "static",
        "tuned_params": PROJECT_ROOT / "logs" / "optuna_cnn_lstm_20260428_001712_w8" / "best_params.json",
    },
}

GRAPH_PRESETS: dict[str, dict[str, Any]] = {
    "hybrid_fixed": {
        "data": {"window_size": 8},
        "graph": {
            "graph_mode": "hybrid",
            "top_k": 4,
            "similarity_metric": "pearson",
            "corr_threshold": 0.7,
        },
        "optuna": {"study_name": "graph_hybrid_fixed"},
    },
    "dual_graph_fixed": {
        "data": {"window_size": 8},
        "graph": {
            "graph_mode": "dual_graph",
            "top_k": 4,
            "similarity_metric": "pearson",
            "corr_threshold": 0.7,
        },
        "optuna": {"study_name": "graph_dual_fixed"},
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _python_bin() -> str:
    env_python = os.environ.get("PYTHON_BIN")
    if env_python:
        return env_python
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    windows_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if windows_python.exists():
        return str(windows_python)
    return sys.executable


def _write_runtime_config(config: dict[str, Any], stem: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_dirs = [
        RUNTIME_CONFIG_DIR,
        Path(tempfile.gettempdir()) / "gat_lstm_runtime_configs",
    ]
    payload = json.dumps(config, indent=2, ensure_ascii=False)

    last_error: Exception | None = None
    for directory in candidate_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{stamp}_{stem}.json"
            path.write_text(payload, encoding="utf-8")
            return path
        except OSError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to materialise runtime config.")


def _run_command(command: list[str], print_only: bool) -> int:
    print("[run]", " ".join(command))
    if print_only:
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def _build_final_baseline_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    family = args.family.strip().lower()
    if family not in FINAL_BASELINE_PRESETS:
        raise ValueError(f"Unsupported final-baseline family: {family}")

    config = _load_json(Path(args.config) if args.config else DEFAULT_CONFIG)
    preset = FINAL_BASELINE_PRESETS[family]
    tuned_params = Path(args.tuned_params) if args.tuned_params else Path(preset["tuned_params"])
    window_size = int(args.window_size) if args.window_size is not None else int(preset["window_size"])
    baseline_epochs = int(args.baseline_epochs) if args.baseline_epochs is not None else None

    _deep_update(
        config,
        {
            "data": {"window_size": window_size},
            "graph": {"graph_mode": preset["graph_mode"]},
        },
    )
    if baseline_epochs is not None:
        _deep_update(config, {"benchmark": {"baseline_epochs": baseline_epochs}})

    runtime_config = _write_runtime_config(config, f"final_{family}_w{window_size}")
    command = [
        _python_bin(),
        "main.py",
        "--mode",
        "final-baseline",
        "--config",
        str(runtime_config),
        "--families",
        family,
        "--window-size",
        str(window_size),
        "--tuned-params",
        str(tuned_params),
    ]
    return command, runtime_config


def _build_tune_graph_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    preset_name = args.preset.strip().lower()
    if preset_name not in GRAPH_PRESETS:
        raise ValueError(f"Unsupported graph preset: {preset_name}")

    family = args.family.strip().lower()
    config = _load_json(Path(args.config) if args.config else DEFAULT_CONFIG)
    preset = GRAPH_PRESETS[preset_name]
    _deep_update(config, preset)

    if args.study_name:
        _deep_update(config, {"optuna": {"study_name": args.study_name}})

    runtime_config = _write_runtime_config(config, f"{preset_name}_{family}")
    command = [
        _python_bin(),
        "main.py",
        "--mode",
        "tune",
        "--config",
        str(runtime_config),
        "--families",
        family,
        "--trials",
        str(args.trials),
        "--max-epochs-per-trial",
        str(args.max_epochs_per_trial),
        "--prune-after-epochs",
        str(args.prune_after_epochs),
    ]
    return command, runtime_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible experiment launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    final_parser = subparsers.add_parser("final-baseline", help="Train/test one tuned neural baseline")
    final_parser.add_argument("--family", required=True, choices=sorted(FINAL_BASELINE_PRESETS.keys()))
    final_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    final_parser.add_argument("--tuned-params", default=None)
    final_parser.add_argument("--window-size", type=int, default=None)
    final_parser.add_argument("--baseline-epochs", type=int, default=None)
    final_parser.add_argument("--print-only", action="store_true")

    tune_parser = subparsers.add_parser("tune-graph", help="Tune one graph model under a fixed graph preset")
    tune_parser.add_argument("--preset", required=True, choices=sorted(GRAPH_PRESETS.keys()))
    tune_parser.add_argument("--family", required=True, choices=["gcn_lstm", "gat_lstm", "gat_dual"])
    tune_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    tune_parser.add_argument("--study-name", default=None)
    tune_parser.add_argument("--trials", type=int, default=100)
    tune_parser.add_argument("--max-epochs-per-trial", type=int, default=12)
    tune_parser.add_argument("--prune-after-epochs", type=int, default=8)
    tune_parser.add_argument("--print-only", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "final-baseline":
        command, runtime_config = _build_final_baseline_command(args)
    else:
        command, runtime_config = _build_tune_graph_command(args)

    print(f"[config] runtime_config={runtime_config}")
    return _run_command(command, print_only=args.print_only)


if __name__ == "__main__":
    raise SystemExit(main())
