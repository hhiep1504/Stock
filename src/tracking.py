"""Experiment logging and comparison utilities."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils import compute_file_hash, get_environment_snapshot, get_git_metadata

import torch

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


class ExperimentLogger:
    """Persist experiment configuration, metrics, and artefacts."""

    def __init__(self, log_dir: str | Path = "logs", experiment_name: str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"exp_{self.timestamp}"
        self.experiment_id = self.timestamp
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.exp_dir / "config.json"
        self.metrics_file = self.exp_dir / "metrics.json"
        self.training_log = self.exp_dir / "training_log.csv"
        self.summary_file = self.exp_dir / "summary.txt"

        self.config: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.epoch_logs: list[dict[str, Any]] = []
        self.system_info = {
            "environment": get_environment_snapshot(),
            "git": get_git_metadata(self.log_dir.parent),
        }
        self.use_mlflow = mlflow is not None
        self.mlflow_run = None

    def log_config(self, config_dict: dict[str, Any]) -> None:
        data_file = config_dict.get("data", {}).get("daily_file")
        dataset_hash = compute_file_hash(data_file) if data_file else None
        self.config = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": config_dict,
            "system_info": {
                **self.system_info,
                "dataset_hash": dataset_hash,
            },
        }
        with self.config_file.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, ensure_ascii=False)

        if self.use_mlflow:
            try:
                mlflow.set_experiment(self.experiment_name)
                self.mlflow_run = mlflow.start_run(run_name=self.experiment_name)
                mlflow.log_params(
                    {
                        "experiment_id": self.experiment_id,
                        "timestamp": self.timestamp,
                    }
                )
                mlflow.log_dict(self.config, "config.json")
            except Exception:
                self.mlflow_run = None
                self.use_mlflow = False

    def log_model_info(self, model: torch.nn.Module) -> None:
        model_info = {
            "model_name": model.__class__.__name__,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "model_structure": str(model),
        }
        self.config["model_info"] = model_info
        with self.config_file.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, ensure_ascii=False)
        if self.use_mlflow:
            try:
                mlflow.log_params(
                    {
                        "model_name": model_info["model_name"],
                        "total_parameters": model_info["total_parameters"],
                        "trainable_parameters": model_info["trainable_parameters"],
                    }
                )
            except Exception:
                pass

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **metrics,
        }
        self.epoch_logs.append(log_entry)
        file_exists = self.training_log.exists()
        with self.training_log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
        if self.use_mlflow:
            try:
                mlflow.log_metrics(metrics, step=epoch)
            except Exception:
                pass

    def log_loss_components(self, epoch: int, loss_dict: dict[str, float]) -> None:
        self.log_epoch(epoch, loss_dict)

    def log_final_results(self, results: dict[str, Any]) -> None:
        serialisable_results = {}
        for key, value in results.items():
            if key in {"predictions", "targets"}:
                continue
            if hasattr(value, "item"):
                serialisable_results[key] = value.item()
            else:
                serialisable_results[key] = value
        self.metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": serialisable_results,
        }
        with self.metrics_file.open("w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2, ensure_ascii=False)
        if self.use_mlflow:
            try:
                mlflow.log_dict(self.metrics, "metrics.json")
            except Exception:
                pass

    def create_summary(self, additional_notes: str = "") -> None:
        lines = [
            "=" * 70,
            f"EXPERIMENT SUMMARY: {self.experiment_name}",
            "=" * 70,
            f"Experiment ID: {self.experiment_id}",
            f"Timestamp: {self.timestamp}",
            "",
            "CONFIGURATION:",
            "-" * 70,
        ]

        for key, value in self.config.get("config", {}).items():
            lines.append(f"{key}: {value}")

        model_info = self.config.get("model_info")
        if model_info:
            lines.extend(
                [
                    "",
                    "MODEL INFORMATION:",
                    "-" * 70,
                    f"Model: {model_info.get('model_name', 'N/A')}",
                    f"Total Parameters: {model_info.get('total_parameters', 0):,}",
                    f"Trainable Parameters: {model_info.get('trainable_parameters', 0):,}",
                ]
            )

        if self.epoch_logs:
            first_loss = self.epoch_logs[0].get("loss", self.epoch_logs[0].get("total_loss", "N/A"))
            last_loss = self.epoch_logs[-1].get("loss", self.epoch_logs[-1].get("total_loss", "N/A"))
            lines.extend(
                [
                    "",
                    "TRAINING SUMMARY:",
                    "-" * 70,
                    f"Total Epochs: {len(self.epoch_logs)}",
                    f"Initial Loss: {first_loss}",
                    f"Final Loss: {last_loss}",
                ]
            )

        if self.metrics.get("results"):
            lines.extend(["", "FINAL RESULTS:", "-" * 70])
            for key, value in self.metrics["results"].items():
                lines.append(f"{key}: {value}")

        if additional_notes:
            lines.extend(["", "NOTES:", "-" * 70, additional_notes])

        system_info = self.config.get("system_info", {})
        if system_info:
            lines.extend(
                [
                    "",
                    "SYSTEM INFO:",
                    "-" * 70,
                    f"Environment: {system_info.get('environment', {})}",
                    f"Git: {system_info.get('git', {})}",
                    f"Dataset hash: {system_info.get('dataset_hash', 'N/A')}",
                ]
            )

        lines.append("=" * 70)
        with self.summary_file.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def save_checkpoint(self, model: torch.nn.Module, optimizer, epoch: int) -> Path:
        checkpoint_dir = self.exp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )
        if self.use_mlflow:
            try:
                mlflow.log_artifact(str(checkpoint_path))
            except Exception:
                pass
        return checkpoint_path

    def close(self) -> None:
        if self.use_mlflow and self.mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass
            self.mlflow_run = None

    def get_experiment_dir(self) -> Path:
        return self.exp_dir


class ExperimentComparison:
    """Compare multiple logged experiments."""

    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)

    def list_experiments(self) -> list[str]:
        if not self.log_dir.exists():
            return []
        return sorted(path.name for path in self.log_dir.iterdir() if path.is_dir())

    def load_experiment(self, experiment_name: str) -> dict[str, Any]:
        experiment_dir = self.log_dir / experiment_name
        payload: dict[str, Any] = {"name": experiment_name}

        config_file = experiment_dir / "config.json"
        metrics_file = experiment_dir / "metrics.json"
        if config_file.exists():
            payload["config"] = json.loads(config_file.read_text(encoding="utf-8"))
        if metrics_file.exists():
            payload["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
        return payload

    def compare_experiments(self, experiment_names: list[str] | None = None) -> str:
        selected = experiment_names or self.list_experiments()
        if not selected:
            return "No experiments found."

        experiments = [self.load_experiment(name) for name in selected]
        column_width = 30
        value_width = 22
        header = f"{'PARAMETER':{column_width}s} | " + " | ".join(
            f"{experiment['name'][:value_width]:{value_width}s}" for experiment in experiments
        )

        lines = [
            "=" * 70,
            "EXPERIMENT COMPARISON",
            "=" * 70,
            "",
            header,
            "-" * len(header),
            "",
            "HYPERPARAMETER COMPARISON:",
            "-" * len(header),
        ]

        config_keys = set()
        for experiment in experiments:
            config_keys.update(experiment.get("config", {}).get("config", {}).keys())

        for key in sorted(config_keys):
            values = [
                str(experiment.get("config", {}).get("config", {}).get(key, "N/A"))
                for experiment in experiments
            ]
            marker = " *" if len(set(values)) > 1 else "  "
            lines.append(
                f"{key:{column_width}s} |{marker}" + " | ".join(f"{value[:value_width]:{value_width}s}" for value in values)
            )

        lines.extend(["", "RESULTS COMPARISON:", "-" * len(header)])
        metric_keys = set()
        for experiment in experiments:
            metric_keys.update(experiment.get("metrics", {}).get("results", {}).keys())

        for key in sorted(metric_keys):
            values = []
            for experiment in experiments:
                value = experiment.get("metrics", {}).get("results", {}).get(key, "N/A")
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            marker = " *" if len(set(values)) > 1 else "  "
            lines.append(
                f"{key:{column_width}s} |{marker}" + " | ".join(f"{value[:value_width]:{value_width}s}" for value in values)
            )

        lines.append("")
        report = "\n".join(lines)
        comparison_file = self.log_dir / "comparison_report.txt"
        comparison_file.write_text(report, encoding="utf-8")
        return report
