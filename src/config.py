"""Typed configuration objects for experiments and benchmarking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from src.feature_sets import supported_feature_sets


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _resolve_data_path(path_value: Path, project_root: Path, dataset_dir: Path) -> Path:
    """Resolve data paths robustly for both dataset-prefixed and plain filenames."""

    if path_value.is_absolute():
        return path_value.resolve()

    candidate_project = (project_root / path_value).resolve()
    if candidate_project.exists():
        return candidate_project

    candidate_dataset = (dataset_dir / path_value).resolve()
    if candidate_dataset.exists():
        return candidate_dataset

    # Fall back to dataset_dir for historical configs using plain filenames.
    return candidate_dataset


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid_fields = {field_info.name for field_info in fields(instance)}
    for key, value in values.items():
        if key not in valid_fields:
            continue
        current = getattr(instance, key)
        if isinstance(current, Path) or key.endswith("_dir") or key.endswith("_file") or key == "project_root":
            setattr(instance, key, _as_path(value))
        else:
            setattr(instance, key, value)
    return instance


def _serialise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialise(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialise(item) for item in value]
    return value


@dataclass(slots=True)
class PathConfig:
    """Canonical project locations."""

    project_root: Path = field(default_factory=_project_root)
    dataset_dir: Path | None = None
    outputs_dir: Path | None = None
    logs_dir: Path | None = None
    configs_dir: Path | None = None

    def resolve(self) -> "PathConfig":
        self.project_root = self.project_root.resolve()
        self.dataset_dir = (self.dataset_dir or self.project_root / "dataset").resolve()
        self.outputs_dir = (self.outputs_dir or self.project_root / "outputs").resolve()
        self.logs_dir = (self.logs_dir or self.project_root / "logs").resolve()
        self.configs_dir = (self.configs_dir or self.project_root / "configs").resolve()
        return self


@dataclass(slots=True)
class DataConfig:
    """Dataset and aggregation settings."""

    daily_file: Path | None = None
    target_file: Path | None = None
    aggregation_mode: str = "weekly"
    feature_set: str = "baseline4"
    window_size: int = 8
    split_idx: int = -4

    def __post_init__(self) -> None:
        if self.aggregation_mode not in {"weekly", "quarterly"}:
            raise ValueError("aggregation_mode must be 'weekly' or 'quarterly'.")
        if self.feature_set not in supported_feature_sets():
            supported = ", ".join(supported_feature_sets())
            raise ValueError(f"feature_set must be one of: {supported}.")
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if self.split_idx == 0:
            raise ValueError("split_idx cannot be 0 because it would produce an empty split.")


@dataclass(slots=True)
class GraphConfig:
    """Dynamic and static graph construction settings."""

    graph_mode: str = "hybrid"
    sector_map_file: Path | None = None
    top_k: int = 5
    use_arm: bool = False
    use_static_graph: bool = True
    similarity_metric: str = "cosine"
    corr_threshold: float = 0.7

    def __post_init__(self) -> None:
        self.normalise()

    def normalise(self) -> "GraphConfig":
        if self.graph_mode not in {"static", "dynamic", "hybrid", "dual_graph"}:
            raise ValueError("graph_mode must be 'static', 'dynamic', 'hybrid', or 'dual_graph'.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.similarity_metric not in {"cosine", "pearson"}:
            raise ValueError("similarity_metric must be 'cosine' or 'pearson'.")
        if not 0.0 <= self.corr_threshold <= 1.0:
            raise ValueError("corr_threshold must be between 0 and 1.")
        if self.graph_mode == "dynamic":
            self.use_static_graph = False
        else:
            self.use_static_graph = True
        return self

    def uses_static_component(self) -> bool:
        return self.graph_mode in {"static", "hybrid", "dual_graph"}

    def uses_dynamic_component(self) -> bool:
        return self.graph_mode in {"dynamic", "hybrid", "dual_graph"}


@dataclass(slots=True)
class ModelConfig:
    """Neural architecture settings."""

    model_type: str = "deep"
    in_features: int = 4
    gnn_hidden: int = 128
    lstm_hidden: int = 256
    num_layers: int = 1
    heads: int = 2
    dropout: float = 0.6
    share_graph_lstm_backbone: bool = False
    shared_graph_lstm_hidden: int | None = None
    shared_graph_lstm_layers: int | None = None
    gat_add_self_loops: bool = True
    gcn_add_self_loops: bool = True
    gcn_normalize: bool = True

    def __post_init__(self) -> None:
        if self.model_type not in {"deep", "single"}:
            raise ValueError("model_type must be 'deep' or 'single'.")
        if self.in_features < 1:
            raise ValueError("in_features must be at least 1.")
        if self.gnn_hidden < 1 or self.lstm_hidden < 1:
            raise ValueError("Hidden sizes must be at least 1.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.heads < 1:
            raise ValueError("heads must be at least 1.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")
        if self.shared_graph_lstm_hidden is not None and self.shared_graph_lstm_hidden < 1:
            raise ValueError("shared_graph_lstm_hidden must be at least 1 when provided.")
        if self.shared_graph_lstm_layers is not None and self.shared_graph_lstm_layers < 1:
            raise ValueError("shared_graph_lstm_layers must be at least 1 when provided.")
        if self.share_graph_lstm_backbone and (
            self.shared_graph_lstm_hidden is None or self.shared_graph_lstm_layers is None
        ):
            raise ValueError(
                "shared_graph_lstm_hidden and shared_graph_lstm_layers must be set when "
                "share_graph_lstm_backbone is enabled."
            )


@dataclass(slots=True)
class LossConfig:
    """Loss function settings."""

    name: str = "mse"
    alpha: float = 0.5
    weight_mse: float = 0.45
    weight_corr: float = 0.45
    weight_penalty: float = 0.10

    def __post_init__(self) -> None:
        if self.name not in {"mse", "huber", "custom", "correlation"}:
            raise ValueError("Unsupported loss name.")
        for field_name in ("weight_mse", "weight_corr", "weight_penalty"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1.")

    def as_kwargs(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "weight_mse": self.weight_mse,
            "weight_corr": self.weight_corr,
            "weight_penalty": self.weight_penalty,
        }


@dataclass(slots=True)
class TrainingConfig:
    """Optimisation settings."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    print_every: int = 1
    early_stopping_patience: int = 10
    warmup_epochs: int = 5
    min_delta: float = 1e-4
    use_scheduler: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative.")


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime and logging behaviour."""

    random_seed: int = 42
    device: str = "auto"
    save_checkpoint: bool = True
    save_plots: bool = True
    show_plots: bool = False

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")


@dataclass(slots=True)
class BenchmarkConfig:
    """Benchmark protocol settings."""

    output_dir: Path | None = None
    protocol: str = "fixed_split"
    seeds: list[int] = field(default_factory=lambda: [42])
    gnn_epochs: int = 2
    baseline_epochs: int = 2
    device: str = "auto"
    n_folds: int = 5
    n_runs: int = 3
    include_statistical_models: bool = True
    validation_split: float = 0.1
    early_stopping_patience: int = 8
    min_train_size: int = 150
    test_step: int = 15
    tuned_params_file: Path | None = None

    def __post_init__(self) -> None:
        if self.protocol not in {"fixed_split", "cross_validation", "walk_forward"}:
            raise ValueError("protocol must be 'fixed_split', 'cross_validation', or 'walk_forward'.")
        if not self.seeds:
            raise ValueError("seeds must not be empty.")
        if self.gnn_epochs < 1 or self.baseline_epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2.")
        if self.n_runs < 1:
            raise ValueError("n_runs must be at least 1.")
        if not 0.0 < self.validation_split < 0.5:
            raise ValueError("validation_split must be between 0 and 0.5.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.min_train_size < 1:
            raise ValueError("min_train_size must be at least 1.")
        if self.test_step < 1:
            raise ValueError("test_step must be at least 1.")


@dataclass(slots=True)
class OptunaConfig:
    """Hyperparameter tuning settings."""

    n_trials: int = 2
    direction: str = "minimize"
    metric: str = "mae_interval"
    study_name: str = "gat_lstm_optuna"
    storage_url: str | None = None
    persist_study: bool = True
    resume_study: bool = True
    sampler_seed: int = 42
    max_epochs_per_trial: int = 2
    prune_after_epochs: int = 3

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'.")
        if not self.metric:
            raise ValueError("metric must be provided.")
        if self.storage_url is not None and not self.storage_url.strip():
            self.storage_url = None
        if self.max_epochs_per_trial < 1:
            raise ValueError("max_epochs_per_trial must be at least 1.")
        if self.prune_after_epochs < 1:
            raise ValueError("prune_after_epochs must be at least 1.")


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str | None = None
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)

    def resolve_paths(self) -> "ExperimentConfig":
        self.paths.resolve()
        if self.data.daily_file is not None and not isinstance(self.data.daily_file, Path):
            self.data.daily_file = Path(self.data.daily_file)
        if self.data.daily_file is not None:
            self.data.daily_file = _resolve_data_path(
                self.data.daily_file,
                project_root=self.paths.project_root,
                dataset_dir=self.paths.dataset_dir,
            )
        if self.data.daily_file is None:
            self.data.daily_file = self.paths.dataset_dir / "stock_market_19_24.csv"

        if self.data.target_file is not None and not isinstance(self.data.target_file, Path):
            self.data.target_file = Path(self.data.target_file)
        if self.data.target_file is not None:
            self.data.target_file = _resolve_data_path(
                self.data.target_file,
                project_root=self.paths.project_root,
                dataset_dir=self.paths.dataset_dir,
            )

        if self.benchmark.output_dir is None:
            self.benchmark.output_dir = self.paths.logs_dir
        elif not isinstance(self.benchmark.output_dir, Path):
            self.benchmark.output_dir = Path(self.benchmark.output_dir)
        if self.benchmark.output_dir is not None and not self.benchmark.output_dir.is_absolute():
            self.benchmark.output_dir = (self.paths.project_root / self.benchmark.output_dir).resolve()

        if self.benchmark.tuned_params_file is not None and not isinstance(self.benchmark.tuned_params_file, Path):
            self.benchmark.tuned_params_file = Path(self.benchmark.tuned_params_file)
        if self.benchmark.tuned_params_file is not None and not self.benchmark.tuned_params_file.is_absolute():
            self.benchmark.tuned_params_file = (self.paths.project_root / self.benchmark.tuned_params_file).resolve()

        if self.graph.sector_map_file is not None and not isinstance(self.graph.sector_map_file, Path):
            self.graph.sector_map_file = Path(self.graph.sector_map_file)
        if self.graph.sector_map_file is not None:
            self.graph.sector_map_file = _resolve_data_path(
                self.graph.sector_map_file,
                project_root=self.paths.project_root,
                dataset_dir=self.paths.dataset_dir,
            )

        return self

    def build_experiment_name(self) -> str:
        agg_prefix = "w" if self.data.aggregation_mode == "weekly" else "q"
        model_str = self.model.model_type
        feature_str = self.data.feature_set.replace("_", "-")
        loss_str = (
            f"mse{int(self.loss.weight_mse * 100)}_corr{int(self.loss.weight_corr * 100)}"
            if self.loss.name == "correlation"
            else self.loss.name
        )
        sim_str = f"{self.graph.similarity_metric[:3]}t{int(self.graph.corr_threshold * 10)}"
        return (
            f"{agg_prefix}_gat_{model_str}"
            f"_w{self.data.window_size}_k{self.graph.top_k}"
            f"_{self.graph.graph_mode}"
            f"_{feature_str}"
            f"_{loss_str}"
            f"_h{self.model.gnn_hidden}_{self.model.lstm_hidden}"
            f"_{sim_str}_e{self.training.epochs}"
        )

    def ensure_output_directories(self) -> None:
        self.resolve_paths()
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.configs_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return output_path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        config = cls()
        if "experiment_name" in payload:
            config.experiment_name = payload["experiment_name"]
        nested_sections = (
            ("paths", config.paths),
            ("data", config.data),
            ("graph", config.graph),
            ("model", config.model),
            ("loss", config.loss),
            ("training", config.training),
            ("runtime", config.runtime),
            ("benchmark", config.benchmark),
            ("optuna", config.optuna),
        )
        for section_name, section_instance in nested_sections:
            if section_name in payload:
                _update_dataclass(section_instance, payload[section_name])
        config.data.__post_init__()
        config.graph.normalise()
        return config.resolve_paths()

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def default_experiment_config() -> ExperimentConfig:
    """Convenience factory for tooling and scripts."""

    return ExperimentConfig().resolve_paths()
