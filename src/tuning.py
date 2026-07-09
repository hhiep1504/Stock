"""Optuna-based hyperparameter tuning pipeline."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.nn import GATConv
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.data import DataLoader as StockDataLoader, FeatureEngineer
from src.evaluation import Evaluator
from src.graph import GraphConstructor
from src.benchmarking import CNNLSTMRegressor, GCNLSTMRegressor, TemporalGCNOnlyModel
from src.models import build_model
from src.pipeline import (
    create_sequences,
    materialise_sequences_from_templates,
    prepare_dataset,
    scale_sequences_for_train_count,
    split_sequences,
)
from src.structures import PreparedDataset, SequenceSample
from src.training import Trainer
from src.utils import resolve_device, set_seed


SUPPORTED_FAMILIES = {"gat_deep", "gat_lstm", "gat_dual", "lstm", "gru", "cnn_lstm", "temporal_gcn", "gcn_lstm"}
# Only explicitly graph-tuned families rebuild sequences per trial.
# Main-benchmark families such as gcn_lstm and gat_lstm use the fixed graph from config.
GRAPH_TUNING_FAMILIES = {"gat_deep"}


def _uses_shared_graph_lstm_backbone(cfg: ExperimentConfig) -> bool:
    return bool(cfg.model.share_graph_lstm_backbone)


def _apply_shared_graph_lstm_backbone(cfg: ExperimentConfig) -> None:
    if not _uses_shared_graph_lstm_backbone(cfg):
        return
    cfg.model.lstm_hidden = int(cfg.model.shared_graph_lstm_hidden)
    cfg.model.num_layers = int(cfg.model.shared_graph_lstm_layers)


@dataclass(slots=True)
class TuningResult:
    """Returned summary after Optuna completes."""

    study_name: str
    best_value: float
    best_params: dict[str, Any]
    best_trial: int
    artifact_dir: Path


@dataclass(slots=True)
class MultiTuningResult:
    """Summary returned when tuning multiple model families."""

    runs: dict[str, TuningResult]


class NodeWiseSequenceRegressor(nn.Module):
    """Node-wise sequence baseline that predicts per stock independently."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=self.num_layers,
                dropout=dropout if self.num_layers > 1 else 0.0,
                batch_first=True,
            )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, input_dim = x_seq.shape
            node_sequences = x_seq.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, input_dim)
            output, _ = self.rnn(node_sequences)
            prediction = self.head(output[:, -1, :])
            return prediction.view(batch_size, num_nodes, 2)
        node_sequences = x_seq.permute(1, 0, 2)
        output, _ = self.rnn(node_sequences)
        return self.head(output[:, -1, :])


class NodeWiseCNNLSTMRegressor(nn.Module):
    """Node-wise CNN-LSTM baseline for tuning."""

    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.2,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, 2)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, input_dim = x_seq.shape
            node_sequences = x_seq.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, input_dim, seq_len)
            encoded = self.relu(self.conv1(node_sequences))
            encoded = self.pool(encoded)
            encoded = self.dropout1(encoded)
            encoded = encoded.transpose(1, 2)
            output, _ = self.lstm(encoded)
            output = self.dropout2(output)
            prediction = self.head(output[:, -1, :])
            return prediction.view(batch_size, num_nodes, 2)
        node_sequences = x_seq.permute(1, 2, 0)
        encoded = self.relu(self.conv1(node_sequences))
        encoded = self.pool(encoded)
        encoded = self.dropout1(encoded)
        encoded = encoded.transpose(1, 2)
        output, _ = self.lstm(encoded)
        output = self.dropout2(output)
        return self.head(output[:, -1, :])


class TunableGATLSTMRegressor(nn.Module):
    """Tuning-only GAT-LSTM variant with configurable GAT/LSTM depth."""

    def __init__(
        self,
        in_features: int,
        gnn_hidden: int = 32,
        lstm_hidden: int = 128,
        gat_layers: int = 1,
        heads: int = 4,
        gat_dropout: float = 0.2,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.2,
        add_self_loops: bool = True,
    ):
        super().__init__()
        if gat_layers not in {1, 2}:
            raise ValueError("gat_layers must be 1 or 2.")

        self.gat_layers = gat_layers
        self.gat_dropout = gat_dropout
        self.conv1 = GATConv(
            in_features,
            gnn_hidden,
            heads=heads,
            dropout=gat_dropout,
            add_self_loops=add_self_loops,
        )
        self.conv2 = None
        lstm_input_dim = gnn_hidden * heads

        if gat_layers == 2:
            self.conv2 = GATConv(
                gnn_hidden * heads,
                gnn_hidden,
                heads=heads,
                concat=False,
                dropout=gat_dropout,
                add_self_loops=add_self_loops,
            )
            lstm_input_dim = gnn_hidden

        self.temporal_dropout = nn.Dropout(lstm_dropout)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(lstm_hidden, 2)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            gnn_outputs = []
            for timestep in range(seq_len):
                encoded = F.elu(self.conv1(x_seq[:, timestep].reshape(batch_size * num_nodes, -1), edge_index))
                encoded = F.dropout(encoded, p=self.gat_dropout, training=self.training)
                if self.conv2 is not None:
                    encoded = F.elu(self.conv2(encoded, edge_index))
                    encoded = F.dropout(encoded, p=self.gat_dropout, training=self.training)
                gnn_outputs.append(encoded.view(batch_size, num_nodes, -1))

            spatial_sequence = torch.stack(gnn_outputs, dim=1).reshape(batch_size * num_nodes, seq_len, -1)
            spatial_sequence = self.temporal_dropout(spatial_sequence)
            output, _ = self.lstm(spatial_sequence)
            output = self.temporal_dropout(output)
            prediction = self.head(output[:, -1, :])
            return prediction.view(batch_size, num_nodes, 2)

        gnn_outputs = []
        for timestep in range(x_seq.size(0)):
            encoded = F.elu(self.conv1(x_seq[timestep], edge_index))
            encoded = F.dropout(encoded, p=self.gat_dropout, training=self.training)
            if self.conv2 is not None:
                encoded = F.elu(self.conv2(encoded, edge_index))
                encoded = F.dropout(encoded, p=self.gat_dropout, training=self.training)
            gnn_outputs.append(encoded)

        spatial_sequence = torch.stack(gnn_outputs, dim=0).permute(1, 0, 2)
        spatial_sequence = self.temporal_dropout(spatial_sequence)
        output, _ = self.lstm(spatial_sequence)
        output = self.temporal_dropout(output)
        return self.head(output[:, -1, :])


class TunableDualPathGATLSTMRegressor(nn.Module):
    """Dual-path GAT-LSTM with separate static and dynamic attention branches."""

    def __init__(
        self,
        in_features: int,
        gnn_hidden: int = 32,
        lstm_hidden: int = 128,
        gat_layers: int = 1,
        heads: int = 4,
        gat_dropout: float = 0.2,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.2,
        fusion_type: str = "concat",
        add_self_loops: bool = True,
    ):
        super().__init__()
        if gat_layers not in {1, 2}:
            raise ValueError("gat_layers must be 1 or 2.")
        if fusion_type not in {"concat", "sum", "gated"}:
            raise ValueError("fusion_type must be 'concat', 'sum', or 'gated'.")

        self.gat_dropout = gat_dropout
        self.fusion_type = fusion_type
        self.static_conv1 = GATConv(
            in_features,
            gnn_hidden,
            heads=heads,
            dropout=gat_dropout,
            add_self_loops=add_self_loops,
        )
        self.dynamic_conv1 = GATConv(
            in_features,
            gnn_hidden,
            heads=heads,
            dropout=gat_dropout,
            add_self_loops=add_self_loops,
        )

        self.static_conv2 = None
        self.dynamic_conv2 = None
        branch_dim = gnn_hidden * heads
        if gat_layers == 2:
            self.static_conv2 = GATConv(
                gnn_hidden * heads,
                gnn_hidden,
                heads=heads,
                concat=False,
                dropout=gat_dropout,
                add_self_loops=add_self_loops,
            )
            self.dynamic_conv2 = GATConv(
                gnn_hidden * heads,
                gnn_hidden,
                heads=heads,
                concat=False,
                dropout=gat_dropout,
                add_self_loops=add_self_loops,
            )
            branch_dim = gnn_hidden

        self.gate_linear = None
        fused_dim = branch_dim
        if fusion_type == "concat":
            fused_dim = branch_dim * 2
        elif fusion_type == "gated":
            self.gate_linear = nn.Linear(branch_dim * 2, branch_dim)

        self.temporal_dropout = nn.Dropout(lstm_dropout)
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(lstm_hidden, 2)

    def _encode_branch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        conv1: GATConv,
        conv2: GATConv | None,
    ) -> torch.Tensor:
        encoded = F.elu(conv1(x, edge_index))
        encoded = F.dropout(encoded, p=self.gat_dropout, training=self.training)
        if conv2 is not None:
            encoded = F.elu(conv2(encoded, edge_index))
            encoded = F.dropout(encoded, p=self.gat_dropout, training=self.training)
        return encoded

    def _fuse(self, static_encoded: torch.Tensor, dynamic_encoded: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([static_encoded, dynamic_encoded], dim=-1)
        if self.fusion_type == "sum":
            return 0.5 * (static_encoded + dynamic_encoded)
        gate = torch.sigmoid(self.gate_linear(torch.cat([static_encoded, dynamic_encoded], dim=-1)))
        return gate * static_encoded + (1.0 - gate) * dynamic_encoded

    def forward(self, x_seq: torch.Tensor, edge_index: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        static_edge_index, dynamic_edge_index = edge_index
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            gnn_outputs = []
            for timestep in range(seq_len):
                timestep_features = x_seq[:, timestep].reshape(batch_size * num_nodes, -1)
                static_encoded = self._encode_branch(
                    timestep_features,
                    static_edge_index,
                    self.static_conv1,
                    self.static_conv2,
                )
                dynamic_encoded = self._encode_branch(
                    timestep_features,
                    dynamic_edge_index,
                    self.dynamic_conv1,
                    self.dynamic_conv2,
                )
                fused = self._fuse(static_encoded, dynamic_encoded)
                gnn_outputs.append(fused.view(batch_size, num_nodes, -1))

            spatial_sequence = torch.stack(gnn_outputs, dim=1).reshape(batch_size * num_nodes, seq_len, -1)
            spatial_sequence = self.temporal_dropout(spatial_sequence)
            output, _ = self.lstm(spatial_sequence)
            output = self.temporal_dropout(output)
            prediction = self.head(output[:, -1, :])
            return prediction.view(batch_size, num_nodes, 2)

        gnn_outputs = []
        for timestep in range(x_seq.size(0)):
            static_encoded = self._encode_branch(
                x_seq[timestep],
                static_edge_index,
                self.static_conv1,
                self.static_conv2,
            )
            dynamic_encoded = self._encode_branch(
                x_seq[timestep],
                dynamic_edge_index,
                self.dynamic_conv1,
                self.dynamic_conv2,
            )
            gnn_outputs.append(self._fuse(static_encoded, dynamic_encoded))

        spatial_sequence = torch.stack(gnn_outputs, dim=0).permute(1, 0, 2)
        spatial_sequence = self.temporal_dropout(spatial_sequence)
        output, _ = self.lstm(spatial_sequence)
        output = self.temporal_dropout(output)
        return self.head(output[:, -1, :])


def _make_sequence_loader(sequences: list[SequenceSample], batch_size: int) -> TorchDataLoader:
    return TorchDataLoader(sequences, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch)


def _build_time_series_fold_indices(
    num_sequences: int,
    n_splits: int,
) -> list[tuple[list[int], list[int]]]:
    if num_sequences < 3:
        raise ValueError("Need at least 3 sequences to build expanding TimeSeriesSplit folds.")

    effective_splits = min(max(2, n_splits), num_sequences - 1)
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    folds: list[tuple[list[int], list[int]]] = []
    indices = list(range(num_sequences))

    for train_indices, val_indices in splitter.split(indices):
        folds.append((train_indices.tolist(), val_indices.tolist()))

    return folds


def _materialise_time_series_folds(
    sequences: list[SequenceSample],
    fold_indices: list[tuple[list[int], list[int]]],
) -> list[tuple[list[SequenceSample], list[SequenceSample]]]:
    folds: list[tuple[list[SequenceSample], list[SequenceSample]]] = []
    for train_indices, val_indices in fold_indices:
        fold_train = [sequences[index] for index in train_indices]
        fold_val = [sequences[index] for index in val_indices]
        folds.append((fold_train, fold_val))
    return folds


def _tuning_print_every(max_epochs: int) -> int:
    """Reduce console overhead during Optuna while keeping fold-level visibility."""

    return max(1, max_epochs)


def _apply_graph_trial_params(cfg: ExperimentConfig, trial: optuna.Trial) -> None:
    cfg.graph.graph_mode = trial.suggest_categorical("graph_mode", ["static", "dynamic", "hybrid"])
    cfg.graph.normalise()

    if cfg.graph.uses_dynamic_component():
        cfg.graph.similarity_metric = trial.suggest_categorical("similarity_metric", ["pearson", "cosine"])
        if cfg.graph.uses_static_component():
            cfg.graph.top_k = trial.suggest_categorical("top_k_hybrid", [1, 2, 3, 4, 5])
        else:
            cfg.graph.top_k = trial.suggest_categorical("top_k_dynamic", [2, 3, 4, 5, 6, 8, 10])

        if cfg.graph.similarity_metric == "pearson":
            cfg.graph.corr_threshold = trial.suggest_categorical(
                "corr_threshold_pearson",
                [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
            )
        else:
            cfg.graph.corr_threshold = trial.suggest_categorical(
                "corr_threshold_cosine",
                [0.85, 0.90, 0.95, 0.98, 0.995],
            )


def _graph_config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "graph_mode": cfg.graph.graph_mode,
        "use_static_graph": cfg.graph.use_static_graph,
        "similarity_metric": cfg.graph.similarity_metric,
        "top_k": cfg.graph.top_k,
        "corr_threshold": cfg.graph.corr_threshold,
        "sector_map_file": str(cfg.graph.sector_map_file) if cfg.graph.sector_map_file else None,
    }


def _graph_config_summary(cfg: ExperimentConfig) -> str:
    return (
        f"graph_mode={cfg.graph.graph_mode} | "
        f"use_static_graph={cfg.graph.use_static_graph} | "
        f"similarity_metric={cfg.graph.similarity_metric} | "
        f"top_k={cfg.graph.top_k} | "
        f"corr_threshold={cfg.graph.corr_threshold:.3f} | "
        f"sector_map_file={cfg.graph.sector_map_file}"
    )


def _build_graph_tuning_sequences(base_prepared: PreparedDataset, cfg: ExperimentConfig) -> list[SequenceSample]:
    data_loader = StockDataLoader(cfg.data.daily_file, cfg.data.target_file)
    data_loader.stock_codes = list(base_prepared.stock_codes)
    data_loader.df_daily_merged = base_prepared.daily_frame

    graph_constructor = GraphConstructor(base_prepared.stock_codes)
    static_edge_index = None
    if cfg.graph.uses_static_component():
        if cfg.graph.sector_map_file is not None:
            graph_constructor.sector_map = GraphConstructor.load_sector_map(cfg.graph.sector_map_file)
            graph_constructor.use_bridge_edges = False
        graph_constructor.sector_map = graph_constructor.get_sector_mapping()
        static_edge_index = graph_constructor.create_static_graph()

    return create_sequences(
        x_full=base_prepared.x_full_raw,
        y_full=base_prepared.y_full,
        data_loader=data_loader,
        graph_constructor=graph_constructor,
        valid_indices_map=base_prepared.valid_indices_map,
        graph_mode=cfg.graph.graph_mode,
        window_size=cfg.data.window_size,
        top_k=cfg.graph.top_k,
        aggregation_mode=cfg.data.aggregation_mode,
        static_edge_index=static_edge_index,
        corr_threshold=cfg.graph.corr_threshold,
        similarity_metric=cfg.graph.similarity_metric,
        use_arm=cfg.graph.use_arm,
    )


def _normalise_families(families: list[str] | None) -> list[str]:
    if not families:
        return ["gat_deep"]
    cleaned = []
    for family in families:
        family_name = family.strip().lower()
        if family_name == "all":
            return ["gat_deep", "gat_lstm", "lstm", "gru", "cnn_lstm", "gcn_lstm"]
        if family_name == "gat_single":
            family_name = "gat_deep"
        if family_name == "gcn_only":
            family_name = "temporal_gcn"
        if family_name not in SUPPORTED_FAMILIES:
            raise ValueError(f"Unsupported model family: {family_name}")
        if family_name not in cleaned:
            cleaned.append(family_name)
    return cleaned


def _apply_trial_params(base_config: ExperimentConfig, trial: optuna.Trial, family: str) -> ExperimentConfig:
    cfg = ExperimentConfig.from_dict(base_config.to_dict())

    if family == "lstm":
        cfg.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        cfg.training.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.5)
        cfg.model.num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    elif family == "gru":
        cfg.training.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.5)
        cfg.model.num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    elif family == "cnn_lstm":
        cfg.training.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        cfg.model.dropout = trial.suggest_float("dropout", 0.2, 0.5)
    elif family == "gat_lstm":
        cfg.training.learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        cfg.training.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        cfg.model.gnn_hidden = trial.suggest_categorical("gnn_hidden", [32, 64, 128, 192])
        cfg.model.heads = trial.suggest_categorical("heads", [1, 2, 4])
        if _uses_shared_graph_lstm_backbone(cfg):
            _apply_shared_graph_lstm_backbone(cfg)
        else:
            cfg.model.lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 96, 128, 192, 256])
            cfg.model.num_layers = trial.suggest_int("lstm_layers", 1, 2)
        cfg.model.dropout = trial.suggest_float("gat_dropout", 0.1, 0.5, step=0.1)
        trial.suggest_int("gat_layers", 1, 2)
        trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1)
    elif family == "gat_dual":
        cfg.graph.graph_mode = "dual_graph"
        cfg.graph.normalise()
        cfg.training.learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        cfg.training.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        cfg.model.gnn_hidden = trial.suggest_categorical("gnn_hidden", [32, 64, 128, 192])
        cfg.model.heads = trial.suggest_categorical("heads", [1, 2, 4])
        if _uses_shared_graph_lstm_backbone(cfg):
            _apply_shared_graph_lstm_backbone(cfg)
        else:
            cfg.model.lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 96, 128, 192, 256])
            cfg.model.num_layers = trial.suggest_int("lstm_layers", 1, 2)
        cfg.model.dropout = trial.suggest_float("gat_dropout", 0.1, 0.5, step=0.1)
        trial.suggest_int("gat_layers", 1, 2)
        trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1)
        trial.suggest_categorical("fusion_type", ["concat", "sum", "gated"])
    elif family == "gcn_lstm":
        cfg.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        cfg.training.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        cfg.model.model_type = "single"
        cfg.model.gnn_hidden = trial.suggest_categorical("gcn_hidden_dim", [16, 32, 64, 128])
        if _uses_shared_graph_lstm_backbone(cfg):
            _apply_shared_graph_lstm_backbone(cfg)
        else:
            cfg.model.lstm_hidden = trial.suggest_categorical("lstm_hidden_dim", [16, 32, 64, 128])
            cfg.model.num_layers = trial.suggest_int("lstm_layers", 1, 3)
        cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    else:
        cfg.training.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

    if family not in {"gat_lstm", "gat_dual", "gcn_lstm"}:
        cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

    if family == "gat_deep":
        cfg.model.model_type = "deep"
        cfg.model.gnn_hidden = trial.suggest_categorical("gnn_hidden", [64, 96, 128, 192, 256])
        cfg.model.lstm_hidden = trial.suggest_categorical("lstm_hidden", [96, 128, 192, 256, 384])
        cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.7)
        cfg.model.heads = trial.suggest_categorical("heads", [1, 2, 4])
        _apply_graph_trial_params(cfg, trial)

    return cfg


def _build_family_model(family: str, trial_cfg: ExperimentConfig, num_nodes: int, seq_len: int, trial: optuna.Trial):
    if family == "gat_deep":
        return build_model(trial_cfg.model, num_nodes=num_nodes)

    input_dim = trial_cfg.model.in_features
    if family == "gat_lstm":
        gat_layers = int(trial.params["gat_layers"]) if "gat_layers" in trial.params else trial.suggest_int("gat_layers", 1, 2)
        lstm_dropout = float(trial.params["lstm_dropout"]) if "lstm_dropout" in trial.params else trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1)
        return TunableGATLSTMRegressor(
            in_features=input_dim,
            gnn_hidden=trial_cfg.model.gnn_hidden,
            lstm_hidden=trial_cfg.model.lstm_hidden,
            gat_layers=gat_layers,
            heads=trial_cfg.model.heads,
            gat_dropout=trial_cfg.model.dropout,
            lstm_layers=trial_cfg.model.num_layers,
            lstm_dropout=lstm_dropout,
            add_self_loops=trial_cfg.model.gat_add_self_loops,
        )
    if family == "gat_dual":
        gat_layers = int(trial.params["gat_layers"]) if "gat_layers" in trial.params else trial.suggest_int("gat_layers", 1, 2)
        lstm_dropout = float(trial.params["lstm_dropout"]) if "lstm_dropout" in trial.params else trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1)
        fusion_type = str(trial.params["fusion_type"]) if "fusion_type" in trial.params else trial.suggest_categorical("fusion_type", ["concat", "sum", "gated"])
        return TunableDualPathGATLSTMRegressor(
            in_features=input_dim,
            gnn_hidden=trial_cfg.model.gnn_hidden,
            lstm_hidden=trial_cfg.model.lstm_hidden,
            gat_layers=gat_layers,
            heads=trial_cfg.model.heads,
            gat_dropout=trial_cfg.model.dropout,
            lstm_layers=trial_cfg.model.num_layers,
            lstm_dropout=lstm_dropout,
            fusion_type=fusion_type,
            add_self_loops=trial_cfg.model.gat_add_self_loops,
        )
    if family == "lstm":
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 192])
        return NodeWiseSequenceRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            rnn_type="lstm",
            num_layers=trial_cfg.model.num_layers,
            dropout=trial_cfg.model.dropout,
        )
    if family == "gru":
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 192])
        return NodeWiseSequenceRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            rnn_type="gru",
            num_layers=trial_cfg.model.num_layers,
            dropout=trial_cfg.model.dropout,
        )
    if family == "cnn_lstm":
        conv_channels = trial.suggest_categorical("conv_channels", [16, 32, 48, 64])
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 96, 128])
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        pool_size = trial.suggest_int("pool_size", 2, 3)
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)
        return NodeWiseCNNLSTMRegressor(
            input_dim=input_dim,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout=trial_cfg.model.dropout,
            num_lstm_layers=num_lstm_layers,
        )
    if family == "temporal_gcn":
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
        return TemporalGCNOnlyModel(
            in_features=input_dim,
            hidden=hidden_dim,
            seq_len=seq_len,
            add_self_loops=trial_cfg.model.gcn_add_self_loops,
            normalize=trial_cfg.model.gcn_normalize,
        )
    if family == "gcn_lstm":
        gcn_layers = int(trial.params["gcn_layers"]) if "gcn_layers" in trial.params else trial.suggest_int("gcn_layers", 1, 2)
        return GCNLSTMRegressor(
            in_features=input_dim,
            gcn_hidden_dim=trial_cfg.model.gnn_hidden,
            lstm_hidden_dim=trial_cfg.model.lstm_hidden,
            gcn_layers=gcn_layers,
            lstm_layers=trial_cfg.model.num_layers,
            dropout=trial_cfg.model.dropout,
            add_self_loops=trial_cfg.model.gcn_add_self_loops,
            normalize=trial_cfg.model.gcn_normalize,
        )

    raise ValueError(f"Unsupported family: {family}")


def _objective_for_lstm_family(
    base_config: ExperimentConfig,
    prepared_stock_count: int,
    cv_folds: list[tuple[list[SequenceSample], list[SequenceSample]]],
    device: torch.device,
    trial: optuna.Trial,
) -> float:
    trial_cfg = _apply_trial_params(base_config, trial, "lstm")
    set_seed(base_config.runtime.random_seed + trial.number)

    print(
        f"[Optuna][lstm] trial={trial.number + 1} | "
        f"lr={trial_cfg.training.learning_rate:.6g} | "
        f"weight_decay={trial_cfg.training.weight_decay:.6g} | "
        f"batch_size={trial_cfg.training.batch_size} | "
        f"hidden_dim={trial.params.get('hidden_dim')} | "
        f"dropout={trial_cfg.model.dropout:.3f} | "
        f"num_layers={trial_cfg.model.num_layers}"
    )

    fold_metrics: list[float] = []
    max_epochs = base_config.optuna.max_epochs_per_trial
    print_every = _tuning_print_every(max_epochs)

    for fold_index, (fold_train, fold_val) in enumerate(cv_folds, start=1):
        print(
            f"[Optuna][lstm] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(cv_folds)} | "
            f"train_samples={len(fold_train)} | val_samples={len(fold_val)}"
        )

        model = _build_family_model("lstm", trial_cfg, prepared_stock_count, 0, trial).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial_cfg.training.learning_rate,
            weight_decay=trial_cfg.training.weight_decay,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn="mse",
            scheduler=None,
            logger=None,
            loss_weights=trial_cfg.loss.as_kwargs(),
            device=device,
        )

        train_loader = _make_sequence_loader(fold_train, trial_cfg.training.batch_size)
        trainer.train(
            train_loader,
            num_epochs=max_epochs,
            print_every=print_every,
            early_stopping_patience=trial_cfg.training.early_stopping_patience,
            warmup_epochs=trial_cfg.training.warmup_epochs,
            min_delta=trial_cfg.training.min_delta,
        )

        evaluator = Evaluator(model, device=device)
        val_metrics = evaluator.evaluate(fold_val)
        fold_metric = float(val_metrics[base_config.optuna.metric])
        fold_metrics.append(fold_metric)

        current_average = sum(fold_metrics) / len(fold_metrics)
        print(
            f"[Optuna][lstm] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(cv_folds)} | "
            f"{base_config.optuna.metric}={fold_metric:.6f} | "
            f"running_mean={current_average:.6f}"
        )
        trial.report(current_average, step=fold_index)
        if trial.should_prune():
            print(
                f"[Optuna][lstm] trial={trial.number + 1} | "
                f"pruned_at_fold={fold_index} | running_mean={current_average:.6f}"
            )
            raise optuna.TrialPruned()

    mean_metric = sum(fold_metrics) / len(fold_metrics)
    print(
        f"[Optuna][lstm] trial={trial.number + 1} | "
        f"completed | mean_{base_config.optuna.metric}={mean_metric:.6f}"
    )
    trial.set_user_attr("metrics", {"fold_metrics": fold_metrics, "mean_metric": mean_metric})
    trial.set_user_attr("family", "lstm")
    trial.set_user_attr("batch_size", trial_cfg.training.batch_size)
    trial.set_user_attr("cv_strategy", f"TimeSeriesSplit(n_splits={len(cv_folds)})")
    return mean_metric


def _objective_for_gat_lstm_family(
    base_config: ExperimentConfig,
    prepared_stock_count: int,
    seq_len: int,
    cv_folds: list[tuple[list[SequenceSample], list[SequenceSample]]],
    device: torch.device,
    trial: optuna.Trial,
) -> float:
    trial_cfg = _apply_trial_params(base_config, trial, "gat_lstm")
    set_seed(base_config.runtime.random_seed + trial.number)

    gat_layers = trial.suggest_int("gat_layers", 1, 2)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1)
    max_epochs = base_config.optuna.max_epochs_per_trial
    print_every = _tuning_print_every(max_epochs)

    print(
        f"[Optuna][gat_lstm] trial={trial.number + 1} | "
        f"lr={trial_cfg.training.learning_rate:.6g} | "
        f"weight_decay={trial_cfg.training.weight_decay:.6g} | "
        f"batch_size={trial_cfg.training.batch_size} | "
        f"gnn_hidden={trial_cfg.model.gnn_hidden} | "
        f"gat_layers={gat_layers} | "
        f"heads={trial_cfg.model.heads} | "
        f"gat_dropout={trial_cfg.model.dropout:.3f} | "
        f"lstm_hidden={trial_cfg.model.lstm_hidden} | "
        f"lstm_layers={trial_cfg.model.num_layers} | "
        f"lstm_dropout={lstm_dropout:.3f}"
    )

    fold_metrics: list[float] = []
    for fold_index, (fold_train, fold_val) in enumerate(cv_folds, start=1):
        print(
            f"[Optuna][gat_lstm] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(cv_folds)} | "
            f"train_samples={len(fold_train)} | val_samples={len(fold_val)}"
        )

        model = _build_family_model(
            "gat_lstm",
            trial_cfg,
            prepared_stock_count,
            seq_len,
            trial,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial_cfg.training.learning_rate,
            weight_decay=trial_cfg.training.weight_decay,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn="mse",
            scheduler=None,
            logger=None,
            loss_weights=trial_cfg.loss.as_kwargs(),
            device=device,
        )

        train_loader = _make_sequence_loader(fold_train, trial_cfg.training.batch_size)
        trainer.train(
            train_loader,
            num_epochs=max_epochs,
            print_every=print_every,
            early_stopping_patience=trial_cfg.training.early_stopping_patience,
            warmup_epochs=trial_cfg.training.warmup_epochs,
            min_delta=trial_cfg.training.min_delta,
        )

        evaluator = Evaluator(model, device=device)
        val_metrics = evaluator.evaluate(fold_val)
        fold_metric = float(val_metrics[base_config.optuna.metric])
        fold_metrics.append(fold_metric)

        current_average = sum(fold_metrics) / len(fold_metrics)
        print(
            f"[Optuna][gat_lstm] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(cv_folds)} | "
            f"{base_config.optuna.metric}={fold_metric:.6f} | "
            f"running_mean={current_average:.6f}"
        )
        trial.report(current_average, step=fold_index)
        if trial.should_prune():
            print(
                f"[Optuna][gat_lstm] trial={trial.number + 1} | "
                f"pruned_at_fold={fold_index} | running_mean={current_average:.6f}"
            )
            raise optuna.TrialPruned()

    mean_metric = sum(fold_metrics) / len(fold_metrics)
    trial.set_user_attr("metrics", {"fold_metrics": fold_metrics, "mean_metric": mean_metric})
    trial.set_user_attr("family", "gat_lstm")
    trial.set_user_attr("batch_size", trial_cfg.training.batch_size)
    trial.set_user_attr("cv_strategy", f"TimeSeriesSplit(n_splits={len(cv_folds)})")
    return mean_metric


def _objective_for_gcn_lstm_family(
    base_config: ExperimentConfig,
    prepared_stock_count: int,
    seq_len: int,
    cv_folds: list[tuple[list[SequenceSample], list[SequenceSample]]],
    device: torch.device,
    trial: optuna.Trial,
) -> float:
    trial_cfg = _apply_trial_params(base_config, trial, "gcn_lstm")
    set_seed(base_config.runtime.random_seed + trial.number)
    gcn_layers = trial.suggest_int("gcn_layers", 1, 2)

    max_epochs = base_config.optuna.max_epochs_per_trial
    print_every = _tuning_print_every(max_epochs)

    print(
        f"[Optuna][gcn_lstm] trial={trial.number + 1} | "
        f"lr={trial_cfg.training.learning_rate:.6g} | "
        f"weight_decay={trial_cfg.training.weight_decay:.6g} | "
        f"batch_size={trial_cfg.training.batch_size} | "
        f"gcn_layers={gcn_layers} | "
        f"gcn_hidden_dim={trial.params.get('gcn_hidden_dim')} | "
        f"lstm_layers={trial.params.get('lstm_layers')} | "
        f"lstm_hidden_dim={trial.params.get('lstm_hidden_dim')} | "
        f"dropout={trial_cfg.model.dropout:.3f}"
    )

    fold_metrics: list[float] = []
    for fold_index, (fold_train, fold_val) in enumerate(cv_folds, start=1):
        print(
            f"[Optuna][gcn_lstm] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(cv_folds)} | "
            f"train_samples={len(fold_train)} | val_samples={len(fold_val)}"
        )

        model = _build_family_model(
            "gcn_lstm",
            trial_cfg,
            prepared_stock_count,
            seq_len,
            trial,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial_cfg.training.learning_rate,
            weight_decay=trial_cfg.training.weight_decay,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn="mse",
            scheduler=None,
            logger=None,
            loss_weights=trial_cfg.loss.as_kwargs(),
            device=device,
        )

        train_loader = _make_sequence_loader(fold_train, trial_cfg.training.batch_size)
        trainer.train(
            train_loader,
            num_epochs=max_epochs,
            print_every=print_every,
            early_stopping_patience=trial_cfg.training.early_stopping_patience,
            warmup_epochs=trial_cfg.training.warmup_epochs,
            min_delta=trial_cfg.training.min_delta,
        )

        evaluator = Evaluator(model, device=device)
        val_metrics = evaluator.evaluate(fold_val)
        fold_metric = float(val_metrics[base_config.optuna.metric])
        fold_metrics.append(fold_metric)

        current_average = sum(fold_metrics) / len(fold_metrics)
        print(
            f"[Optuna][gcn_lstm] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(cv_folds)} | "
            f"{base_config.optuna.metric}={fold_metric:.6f} | "
            f"running_mean={current_average:.6f}"
        )
        trial.report(current_average, step=fold_index)
        if trial.should_prune():
            print(
                f"[Optuna][gcn_lstm] trial={trial.number + 1} | "
                f"pruned_at_fold={fold_index} | running_mean={current_average:.6f}"
            )
            raise optuna.TrialPruned()

    mean_metric = sum(fold_metrics) / len(fold_metrics)
    trial.set_user_attr("metrics", {"fold_metrics": fold_metrics, "mean_metric": mean_metric})
    trial.set_user_attr("family", "gcn_lstm")
    trial.set_user_attr("batch_size", trial_cfg.training.batch_size)
    trial.set_user_attr("cv_strategy", f"TimeSeriesSplit(n_splits={len(cv_folds)})")
    return mean_metric


def _objective_for_family(
    family: str,
    base_config: ExperimentConfig,
    base_prepared: PreparedDataset,
    base_train_sequences: list[SequenceSample],
    fold_indices: list[tuple[list[int], list[int]]],
    device: torch.device,
    trial: optuna.Trial,
) -> float:
    trial_cfg = _apply_trial_params(base_config, trial, family)
    set_seed(base_config.runtime.random_seed + trial.number)
    sequence_templates = base_prepared.sequence_templates
    if family in GRAPH_TUNING_FAMILIES or trial_cfg.graph.graph_mode != base_config.graph.graph_mode:
        sequence_templates = _build_graph_tuning_sequences(base_prepared, trial_cfg)

    trial_fold_indices = fold_indices
    prepared_stock_count = len(base_prepared.stock_codes)
    seq_len = trial_cfg.data.window_size
    max_epochs = base_config.optuna.max_epochs_per_trial
    print_every = _tuning_print_every(max_epochs)
    fold_metrics: list[float] = []
    summary_printed = False

    for fold_index, (train_indices, val_indices) in enumerate(trial_fold_indices, start=1):
        train_sequence_count = max(train_indices) + 1
        feature_engineer = FeatureEngineer(
            base_prepared.daily_frame,
            base_prepared.stock_codes,
            aggregation_mode=trial_cfg.data.aggregation_mode,
        )
        x_full_scaled, fold_train = scale_sequences_for_train_count(
            feature_engineer=feature_engineer,
            x_full_raw=base_prepared.x_full_raw,
            y_full=base_prepared.y_full,
            sequence_templates=sequence_templates,
            window_size=trial_cfg.data.window_size,
            train_sequence_count=train_sequence_count,
            sequence_indices=train_indices,
        )
        fold_val = materialise_sequences_from_templates(
            x_full=x_full_scaled,
            y_full=base_prepared.y_full,
            sequence_templates=sequence_templates,
            window_size=trial_cfg.data.window_size,
            sequence_indices=val_indices,
        )
        model = _build_family_model(family, trial_cfg, prepared_stock_count, seq_len, trial).to(device)

        if not summary_printed:
            summary_parts = [
                f"lr={trial_cfg.training.learning_rate:.6g}",
                f"weight_decay={trial_cfg.training.weight_decay:.6g}",
                f"batch_size={trial_cfg.training.batch_size}",
            ]
            if family in {"lstm", "gru"}:
                summary_parts.extend(
                    [
                        f"hidden_dim={trial.params.get('hidden_dim')}",
                        f"dropout={trial_cfg.model.dropout:.3f}",
                        f"num_layers={trial_cfg.model.num_layers}",
                    ]
                )
            elif family == "cnn_lstm":
                summary_parts.extend(
                    [
                        f"conv_channels={trial.params.get('conv_channels')}",
                        f"kernel_size={trial.params.get('kernel_size')}",
                        f"pool_size={trial.params.get('pool_size')}",
                        f"lstm_hidden={trial.params.get('lstm_hidden')}",
                        f"num_lstm_layers={trial.params.get('num_lstm_layers')}",
                        f"dropout={trial_cfg.model.dropout:.3f}",
                    ]
                )
            elif family == "gat_lstm":
                if _uses_shared_graph_lstm_backbone(trial_cfg):
                    lstm_hidden_text = f"shared({trial_cfg.model.lstm_hidden})"
                    lstm_layers_text = f"shared({trial_cfg.model.num_layers})"
                else:
                    lstm_hidden_text = str(trial_cfg.model.lstm_hidden)
                    lstm_layers_text = str(trial_cfg.model.num_layers)
                summary_parts.extend(
                    [
                        f"gnn_hidden={trial_cfg.model.gnn_hidden}",
                        f"gat_layers={trial.params.get('gat_layers')}",
                        f"heads={trial_cfg.model.heads}",
                        f"gat_dropout={trial_cfg.model.dropout:.3f}",
                        f"lstm_hidden={lstm_hidden_text}",
                        f"lstm_layers={lstm_layers_text}",
                        f"lstm_dropout={trial.params.get('lstm_dropout')}",
                        _graph_config_summary(trial_cfg),
                    ]
                )
            elif family == "gat_dual":
                if _uses_shared_graph_lstm_backbone(trial_cfg):
                    lstm_hidden_text = f"shared({trial_cfg.model.lstm_hidden})"
                    lstm_layers_text = f"shared({trial_cfg.model.num_layers})"
                else:
                    lstm_hidden_text = str(trial_cfg.model.lstm_hidden)
                    lstm_layers_text = str(trial_cfg.model.num_layers)
                summary_parts.extend(
                    [
                        f"gnn_hidden={trial_cfg.model.gnn_hidden}",
                        f"gat_layers={trial.params.get('gat_layers')}",
                        f"heads={trial_cfg.model.heads}",
                        f"gat_dropout={trial_cfg.model.dropout:.3f}",
                        f"fusion_type={trial.params.get('fusion_type')}",
                        f"lstm_hidden={lstm_hidden_text}",
                        f"lstm_layers={lstm_layers_text}",
                        f"lstm_dropout={trial.params.get('lstm_dropout')}",
                        _graph_config_summary(trial_cfg),
                    ]
                )
            elif family == "gcn_lstm":
                if _uses_shared_graph_lstm_backbone(trial_cfg):
                    lstm_hidden_text = f"shared({trial_cfg.model.lstm_hidden})"
                    lstm_layers_text = f"shared({trial_cfg.model.num_layers})"
                else:
                    lstm_hidden_text = str(trial.params.get('lstm_hidden_dim'))
                    lstm_layers_text = str(trial.params.get('lstm_layers'))
                summary_parts.extend(
                    [
                        f"gcn_layers={trial.params.get('gcn_layers')}",
                        f"gcn_hidden_dim={trial.params.get('gcn_hidden_dim')}",
                        f"lstm_hidden_dim={lstm_hidden_text}",
                        f"lstm_layers={lstm_layers_text}",
                        f"dropout={trial_cfg.model.dropout:.3f}",
                        _graph_config_summary(trial_cfg),
                    ]
                )
            elif family == "gat_deep":
                summary_parts.extend(
                    [
                        f"model_type={trial_cfg.model.model_type}",
                        f"gnn_hidden={trial_cfg.model.gnn_hidden}",
                        f"lstm_hidden={trial_cfg.model.lstm_hidden}",
                        f"heads={trial_cfg.model.heads}",
                        f"dropout={trial_cfg.model.dropout:.3f}",
                        _graph_config_summary(trial_cfg),
                    ]
                )
            print(f"[Optuna][{family}] trial={trial.number + 1} | " + " | ".join(summary_parts))
            summary_printed = True

        print(
            f"[Optuna][{family}] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(trial_fold_indices)} | "
            f"train_samples={len(fold_train)} | val_samples={len(fold_val)}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial_cfg.training.learning_rate,
            weight_decay=trial_cfg.training.weight_decay,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn="mse",
            scheduler=None,
            logger=None,
            loss_weights=trial_cfg.loss.as_kwargs(),
            device=device,
        )

        train_loader = _make_sequence_loader(fold_train, trial_cfg.training.batch_size)
        trainer.train(
            train_loader,
            num_epochs=max_epochs,
            print_every=print_every,
            early_stopping_patience=trial_cfg.training.early_stopping_patience,
            warmup_epochs=trial_cfg.training.warmup_epochs,
            min_delta=trial_cfg.training.min_delta,
        )

        evaluator = Evaluator(model, device=device)
        val_metrics = evaluator.evaluate(fold_val)
        fold_metric = float(val_metrics[base_config.optuna.metric])
        fold_metrics.append(fold_metric)

        current_average = sum(fold_metrics) / len(fold_metrics)
        print(
            f"[Optuna][{family}] trial={trial.number + 1} | "
            f"fold={fold_index}/{len(trial_fold_indices)} | "
            f"{base_config.optuna.metric}={fold_metric:.6f} | "
            f"running_mean={current_average:.6f}"
        )
        trial.report(current_average, step=fold_index)
        if trial.should_prune():
            print(
                f"[Optuna][{family}] trial={trial.number + 1} | "
                f"pruned_at_fold={fold_index} | running_mean={current_average:.6f}"
            )
            raise optuna.TrialPruned()

    mean_metric = sum(fold_metrics) / len(fold_metrics)
    trial.set_user_attr("metrics", {"fold_metrics": fold_metrics, "mean_metric": mean_metric})
    trial.set_user_attr("family", family)
    trial.set_user_attr("batch_size", trial_cfg.training.batch_size)
    trial.set_user_attr("cv_strategy", f"TimeSeriesSplit(n_splits={len(trial_fold_indices)})")
    if family in GRAPH_TUNING_FAMILIES or trial_cfg.graph.graph_mode != base_config.graph.graph_mode:
        trial.set_user_attr("graph_config", _graph_config_to_dict(trial_cfg))
    return mean_metric


def run_optuna_tuning(
    config: ExperimentConfig,
    n_trials: int | None = None,
    max_epochs_per_trial: int | None = None,
    prune_after_epochs: int | None = None,
    families: list[str] | None = None,
) -> TuningResult | MultiTuningResult:
    """Run leakage-safe Optuna search using train/validation only.

    If multiple families are requested, returns a MultiTuningResult.
    """

    config.ensure_output_directories()
    tuning_config = ExperimentConfig.from_dict(config.to_dict())
    if max_epochs_per_trial is not None:
        tuning_config.optuna.max_epochs_per_trial = max_epochs_per_trial
    if prune_after_epochs is not None:
        tuning_config.optuna.prune_after_epochs = prune_after_epochs
    set_seed(config.runtime.random_seed)
    device = resolve_device(config.runtime.device)

    prepared = prepare_dataset(tuning_config)
    train_sequences, _ = split_sequences(prepared.sequences, tuning_config.data.split_idx)
    model_families = _normalise_families(families)
    fold_indices = _build_time_series_fold_indices(len(train_sequences), tuning_config.benchmark.n_folds)

    if len(model_families) > 1:
        family_runs: dict[str, TuningResult] = {}
        for family in model_families:
            family_runs[family] = _run_single_family_tuning(
                family=family,
                config=tuning_config,
                base_prepared=prepared,
                base_train_sequences=train_sequences,
                fold_indices=fold_indices,
                device=device,
                n_trials=n_trials,
            )
        return MultiTuningResult(runs=family_runs)

    return _run_single_family_tuning(
        family=model_families[0],
        config=tuning_config,
        base_prepared=prepared,
        base_train_sequences=train_sequences,
        fold_indices=fold_indices,
        device=device,
        n_trials=n_trials,
    )


def _run_single_family_tuning(
    family: str,
    config: ExperimentConfig,
    base_prepared: PreparedDataset,
    base_train_sequences: list[SequenceSample],
    fold_indices: list[tuple[list[int], list[int]]],
    device: torch.device,
    n_trials: int | None,
) -> TuningResult:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = config.paths.logs_dir / f"optuna_{family}_{run_stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    target_trials = n_trials or config.optuna.n_trials
    storage_url = config.optuna.storage_url
    if storage_url is None and config.optuna.persist_study:
        storage_path = (config.paths.logs_dir / "optuna_studies.db").resolve()
        storage_url = f"sqlite:///{storage_path.as_posix()}"

    study_base_name = f"{config.optuna.study_name}_{family}"
    study_name = study_base_name if config.optuna.resume_study else f"{study_base_name}_{run_stamp}"

    resume_enabled = config.optuna.resume_study and storage_url is not None

    print(
        f"[Optuna] Family={family} | target_trials={target_trials} | "
        f"max_epochs_per_trial={config.optuna.max_epochs_per_trial} | "
        f"prune_after_epochs={config.optuna.prune_after_epochs} | "
        f"resume={resume_enabled}"
    )

    sampler = optuna.samplers.TPESampler(seed=config.optuna.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(2, max(1, target_trials // 4)),
        n_warmup_steps=config.optuna.prune_after_epochs,
    )
    study = optuna.create_study(
        direction=config.optuna.direction,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=resume_enabled,
    )

    existing_trials = len(study.trials) if resume_enabled else 0
    effective_trials = max(0, target_trials - existing_trials) if resume_enabled else target_trials
    if resume_enabled:
        print(f"[Optuna] {family} | existing_trials={existing_trials} | remaining_trials={effective_trials}")

    progress_bar = tqdm(
        total=target_trials,
        initial=min(existing_trials, target_trials),
        desc=f"Tuning {family}",
        unit="trial",
        leave=True,
    )

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        progress_bar.update(1)
        status = str(trial.state).split(".")[-1]
        value_text = "n/a" if trial.value is None else f"{trial.value:.6f}"
        progress_bar.set_postfix(trial=f"{min(progress_bar.n, target_trials)}/{target_trials}", status=status, value=value_text)
        print(
            f"[Optuna] {family} | trial {trial.number + 1}/{target_trials} | "
            f"status={status} | value={value_text} | params={trial.params}"
        )

    def objective(trial: optuna.Trial) -> float:
        print(f"[Optuna] {family} | starting trial {trial.number + 1}/{target_trials}")
        return _objective_for_family(
            family=family,
            base_config=config,
            base_prepared=base_prepared,
            base_train_sequences=base_train_sequences,
            fold_indices=fold_indices,
            device=device,
            trial=trial,
        )

    if effective_trials > 0:
        study.optimize(
            objective,
            n_trials=effective_trials,
            gc_after_trial=True,
            callbacks=[progress_callback],
        )
    else:
        print(f"[Optuna] {family} | target already reached, no new trials launched.")

    progress_bar.close()

    best_payload = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "metric": config.optuna.metric,
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "best_params": study.best_trial.params,
        "best_metrics": study.best_trial.user_attrs.get("metrics", {}),
    }

    (artifact_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    trials_rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value": trial.value,
            "state": str(trial.state),
            **trial.params,
        }
        trials_rows.append(row)
    (artifact_dir / "trials.json").write_text(json.dumps(trials_rows, indent=2), encoding="utf-8")

    return TuningResult(
        study_name=study.study_name,
        best_value=float(study.best_value),
        best_params=copy.deepcopy(study.best_trial.params),
        best_trial=int(study.best_trial.number),
        artifact_dir=artifact_dir,
    )
