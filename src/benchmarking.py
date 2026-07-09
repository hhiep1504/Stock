"""Benchmark runners for baseline comparison studies."""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from torch_geometric.nn import GATConv, GCNConv

from src.config import ExperimentConfig
from src.data import FeatureEngineer
from src.evaluation import Evaluator
from src.models import build_model
from src.pipeline import materialise_sequences_from_templates, prepare_dataset, split_sequences
from src.structures import PreparedDataset, SequenceSample
from src.training import Trainer
from src.utils import batch_edge_indices, iter_sequence_minibatches, resolve_device, set_seed

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None


class _NullProgress:
    """Fallback progress wrapper when tqdm is not installed."""

    def __init__(self, iterable=None, **kwargs):
        del kwargs
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable if self.iterable is not None else [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return False

    def update(self, n: int = 1) -> None:
        del n

    def set_postfix_str(self, value: str) -> None:
        del value


def _progress(iterable=None, **kwargs):
    if tqdm is None:
        return _NullProgress(iterable, **kwargs)
    return tqdm(iterable, **kwargs)


def _advance_progress(progress_bar, label: str) -> None:
    progress_bar.set_postfix_str(label)
    progress_bar.update(1)


VAR = None
HAS_STATSMODELS = False
if importlib.util.find_spec("statsmodels") is not None:
    try:
        statsmodels_tsa_api = importlib.import_module("statsmodels.tsa.api")
        VAR = getattr(statsmodels_tsa_api, "VAR", None)
        HAS_STATSMODELS = VAR is not None
    except Exception:
        VAR = None
        HAS_STATSMODELS = False


class SequenceRegressor(nn.Module):
    """LSTM/GRU baseline for node-wise interval prediction."""

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
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, input_dim = x.shape
            node_sequences = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, input_dim)
            output, _ = self.rnn(node_sequences)
            prediction = self.head(output[:, -1, :])
            return prediction.view(batch_size, num_nodes, 2)
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :])


class CNNLSTMRegressor(nn.Module):
    """CNN-LSTM baseline for sequence modelling."""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.2,
        num_lstm_layers: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, input_dim = x.shape
            node_sequences = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, input_dim, seq_len)
            encoded = self.relu(self.conv1(node_sequences))
            encoded = self.pool(encoded)
            encoded = self.dropout1(encoded)
            encoded = encoded.transpose(1, 2)
            output, _ = self.lstm(encoded)
            output = self.dropout2(output)
            prediction = self.head(output[:, -1, :])
            return prediction.view(batch_size, num_nodes, 2)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        output = self.dropout2(output)
        return self.head(output[:, -1, :])


class GCNOnlyModel(nn.Module):
    """Spatial-only graph baseline using the last temporal frame."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.gcn1 = GCNConv(in_features, hidden, add_self_loops=add_self_loops, normalize=normalize)
        self.gcn2 = GCNConv(hidden, hidden, add_self_loops=add_self_loops, normalize=normalize)
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden, 2)

    def forward(self, x_nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x_nodes.dim() == 3:
            batch_size, num_nodes, feature_dim = x_nodes.shape
            x = x_nodes.reshape(batch_size * num_nodes, feature_dim)
            x = self.relu(self.gcn1(x, edge_index))
            x = self.relu(self.gcn2(x, edge_index))
            return self.head(x).view(batch_size, num_nodes, 2)
        x = self.relu(self.gcn1(x_nodes, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        return self.head(x)


class TemporalGCNOnlyModel(nn.Module):
    """Improved GCN baseline that aggregates over the full temporal window."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        seq_len: int = 8,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.gcn1 = GCNConv(in_features, hidden, add_self_loops=add_self_loops, normalize=normalize)
        self.gcn2 = GCNConv(hidden, hidden, add_self_loops=add_self_loops, normalize=normalize)
        self.temporal_agg = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden, 2)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            gcn_outputs = []
            for timestep in range(seq_len):
                features = self.relu(self.gcn1(x_seq[:, timestep].reshape(batch_size * num_nodes, -1), edge_index))
                features = self.relu(self.gcn2(features, edge_index))
                gcn_outputs.append(features.view(batch_size, num_nodes, -1))
            spatial_sequence = torch.stack(gcn_outputs, dim=1).reshape(batch_size * num_nodes, seq_len, -1)
            _, (hidden_state, _) = self.temporal_agg(spatial_sequence)
            return self.head(hidden_state.squeeze(0)).view(batch_size, num_nodes, 2)

        gcn_outputs = []
        for timestep in range(self.seq_len):
            features = self.relu(self.gcn1(x_seq[timestep], edge_index))
            features = self.relu(self.gcn2(features, edge_index))
            gcn_outputs.append(features)
        spatial_sequence = torch.stack(gcn_outputs, dim=0).permute(1, 0, 2)
        _, (hidden_state, _) = self.temporal_agg(spatial_sequence)
        return self.head(hidden_state.squeeze(0))


class GCNLSTMRegressor(nn.Module):
    """Configurable GCN-LSTM baseline with shallow spatial depth."""

    def __init__(
        self,
        in_features: int,
        gcn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        gcn_layers: int = 2,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        if gcn_layers not in {1, 2}:
            raise ValueError("gcn_layers must be 1 or 2 to avoid over-smoothing.")

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(
            GCNConv(in_features, gcn_hidden_dim, add_self_loops=add_self_loops, normalize=normalize)
        )
        for _ in range(1, gcn_layers):
            self.gcn_layers.append(
                GCNConv(gcn_hidden_dim, gcn_hidden_dim, add_self_loops=add_self_loops, normalize=normalize)
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.temporal_agg = nn.LSTM(
            input_size=gcn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(lstm_hidden_dim, 2)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            gcn_outputs = []
            for timestep in range(seq_len):
                encoded = x_seq[:, timestep].reshape(batch_size * num_nodes, -1)
                for layer in self.gcn_layers:
                    encoded = self.relu(layer(encoded, edge_index))
                    encoded = self.dropout(encoded)
                gcn_outputs.append(encoded.view(batch_size, num_nodes, -1))

            spatial_sequence = torch.stack(gcn_outputs, dim=1).reshape(batch_size * num_nodes, seq_len, -1)
            _, (hidden_state, _) = self.temporal_agg(spatial_sequence)
            prediction = self.head(hidden_state[-1])
            return prediction.view(batch_size, num_nodes, 2)

        gcn_outputs = []
        seq_len = x_seq.size(0)

        for timestep in range(seq_len):
            encoded = x_seq[timestep]
            for layer in self.gcn_layers:
                encoded = self.relu(layer(encoded, edge_index))
                encoded = self.dropout(encoded)
            gcn_outputs.append(encoded)

        spatial_sequence = torch.stack(gcn_outputs, dim=0).permute(1, 0, 2)
        _, (hidden_state, _) = self.temporal_agg(spatial_sequence)
        return self.head(hidden_state[-1])


class TunedGATLSTMRegressor(nn.Module):
    """GAT-LSTM variant matching the Optuna tuning search space."""

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE and MSE for the min/max interval outputs."""

    mae_min = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_max = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mse_min = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_max = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    return {
        "mae_interval": float(0.5 * (mae_min + mae_max)),
        "mae_min": float(mae_min),
        "mae_max": float(mae_max),
        "mse_interval": float(0.5 * (mse_min + mse_max)),
        "mse_min": float(mse_min),
        "mse_max": float(mse_max),
    }


def flatten_for_tabular(sequences: list[SequenceSample]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten temporal node features for sklearn-style baselines."""

    x_rows = []
    y_rows = []
    for sample in sequences:
        x_np = sample.x_seq.numpy()
        y_np = sample.y_target.numpy()
        for node_index in range(x_np.shape[1]):
            x_rows.append(x_np[:, node_index, :].reshape(-1))
            y_rows.append(y_np[node_index, :])
    return np.asarray(x_rows), np.asarray(y_rows)


def flatten_for_sequence_models(sequences: list[SequenceSample]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten temporal node features for node-wise sequence models."""

    x_rows = []
    y_rows = []
    for sample in sequences:
        x_np = sample.x_seq.numpy()
        y_np = sample.y_target.numpy()
        for node_index in range(x_np.shape[1]):
            x_rows.append(x_np[:, node_index, :])
            y_rows.append(y_np[node_index, :])
    return np.asarray(x_rows), np.asarray(y_rows)


def train_torch_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    """Train a simple torch baseline end to end on flattened node sequences."""

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train_t)
        loss = criterion(prediction, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(x_test_t).detach().cpu().numpy()


def train_sklearn_model(model_class, model_kwargs: dict, x_train, y_train, x_test, seed: int) -> np.ndarray:
    """Train a deterministic sklearn baseline."""

    if model_class in {SVR, LinearRegression}:
        model = MultiOutputRegressor(model_class(**model_kwargs))
    else:
        model = MultiOutputRegressor(model_class(random_state=seed, **model_kwargs))
    model.fit(x_train, y_train)
    return model.predict(x_test)


def evaluate_gat_lstm(
    train_sequences: list[SequenceSample],
    test_sequences: list[SequenceSample],
    config: ExperimentConfig,
    device: torch.device,
    loss_name: str,
    epochs: int,
) -> dict[str, float]:
    """Train and evaluate the proposed GAT-LSTM model."""

    model = build_model(config.model, num_nodes=train_sequences[0].y_target.shape[0]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_name,
        scheduler=None,
        logger=None,
        loss_weights=config.loss.as_kwargs(),
        device=device,
    )
    trainer.train(
        train_sequences,
        num_epochs=epochs,
        print_every=max(1, epochs // 5),
        early_stopping_patience=min(20, max(5, epochs // 4)),
        warmup_epochs=min(config.training.warmup_epochs, epochs),
        min_delta=config.training.min_delta,
        batch_size=config.training.batch_size,
    )
    evaluator = Evaluator(model, device=device)
    evaluation_results = evaluator.evaluate(test_sequences, batch_size=config.training.batch_size)
    return {
        "mae_interval": float(evaluation_results["mae_interval"]),
        "mae_min": float(evaluation_results["mae_min"]),
        "mae_max": float(evaluation_results["mae_max"]),
        "mse_interval": float(evaluation_results["mse_interval"]),
        "mse_min": float(evaluation_results["mse_min"]),
        "mse_max": float(evaluation_results["mse_max"]),
    }


def evaluate_gcn_only(
    train_sequences: list[SequenceSample],
    test_sequences: list[SequenceSample],
    epochs: int,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Evaluate the spatial-only GCN baseline using the last frame."""

    in_features = train_sequences[0].x_seq.shape[2]
    model = GCNOnlyModel(
        in_features=in_features,
        hidden=64,
        add_self_loops=config.model.gcn_add_self_loops,
        normalize=config.model.gcn_normalize,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for sample in train_sequences:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sample.x_seq[-1].to(device), sample.edge_index.long().to(device))
            loss = criterion(prediction, sample.y_target.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for sample in test_sequences:
            prediction = model(sample.x_seq[-1].to(device), sample.edge_index.long().to(device))
            y_true.append(sample.y_target.numpy())
            y_pred.append(prediction.cpu().numpy())
    return compute_metrics(np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0))


def evaluate_temporal_gcn_only(
    train_sequences: list[SequenceSample],
    test_sequences: list[SequenceSample],
    epochs: int,
    device: torch.device,
    seq_len: int,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Evaluate the improved GCN baseline using the full temporal window."""

    in_features = train_sequences[0].x_seq.shape[2]
    model = TemporalGCNOnlyModel(
        in_features=in_features,
        hidden=64,
        seq_len=seq_len,
        add_self_loops=config.model.gcn_add_self_loops,
        normalize=config.model.gcn_normalize,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = config.training.batch_size
    num_nodes = int(train_sequences[0].y_target.shape[0])
    edge_index_cache: dict[tuple[tuple[int, int, int], str], torch.Tensor] = {}
    batched_edge_index_cache: dict[tuple[tuple[int, int, int], int, int, str], torch.Tensor] = {}

    model.train()
    for _ in range(epochs):
        for batch_samples in iter_sequence_minibatches(train_sequences, batch_size):
            x_batch = torch.stack([sample.x_seq for sample in batch_samples], dim=0).to(
                device,
                non_blocking=True,
            )
            y_batch = torch.stack([sample.y_target for sample in batch_samples], dim=0).to(
                device,
                non_blocking=True,
            )
            edge_index = batch_edge_indices(
                [sample.edge_index for sample in batch_samples],
                num_nodes=num_nodes,
                device=device,
                edge_index_cache=edge_index_cache,
                batched_edge_index_cache=batched_edge_index_cache,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch, edge_index)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_samples in iter_sequence_minibatches(test_sequences, batch_size):
            x_batch = torch.stack([sample.x_seq for sample in batch_samples], dim=0).to(
                device,
                non_blocking=True,
            )
            y_batch = torch.stack([sample.y_target for sample in batch_samples], dim=0)
            edge_index = batch_edge_indices(
                [sample.edge_index for sample in batch_samples],
                num_nodes=num_nodes,
                device=device,
                edge_index_cache=edge_index_cache,
                batched_edge_index_cache=batched_edge_index_cache,
            )
            prediction = model(x_batch, edge_index)
            y_true.append(y_batch.numpy())
            y_pred.append(prediction.cpu().numpy())
    return compute_metrics(np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0))


def evaluate_var_or_fallback(y_full: torch.Tensor, window_size: int, split_point: int, model_name: str = "VAR") -> dict[str, float]:
    """Evaluate a VAR baseline, or fall back to naive persistence when needed."""

    y_np = y_full.numpy()
    t_test_start = split_point + window_size
    y_true = y_np[t_test_start:]

    predictions = []
    for timestep in range(t_test_start, y_np.shape[0]):
        if model_name == "VAR" and HAS_STATSMODELS and timestep >= 6:
            pred_t = np.zeros((y_np.shape[1], 2), dtype=np.float32)
            for target_idx in [0, 1]:
                train_matrix = y_np[:timestep, :, target_idx]
                try:
                    var_model = VAR(train_matrix)
                    fitted = var_model.fit(maxlags=min(4, timestep - 1), trend="c")
                    forecast = fitted.forecast(train_matrix[-fitted.k_ar :], steps=1)[0]
                    pred_t[:, target_idx] = forecast
                except Exception:
                    pred_t[:, target_idx] = y_np[timestep - 1, :, target_idx]
            predictions.append(pred_t)
        else:
            predictions.append(y_np[timestep - 1])

    y_pred = np.asarray(predictions)
    return compute_metrics(y_true.reshape(-1, 2), y_pred.reshape(-1, 2))


def aggregate_metric_dicts(metric_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate repeated runs as mean and standard deviation."""

    keys = metric_list[0].keys()
    aggregated = {}
    for key in keys:
        values = np.array([metrics[key] for metrics in metric_list], dtype=np.float64)
        aggregated[f"{key}_mean"] = float(values.mean())
        aggregated[f"{key}_std"] = float(values.std(ddof=0))
    return aggregated


def aggregate_fold_results(fold_results: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate fold-level metrics with mean, std, and 95 percent confidence intervals."""

    keys = fold_results[0].keys()
    summary = {}
    for key in keys:
        values = np.array([metrics[key] for metrics in fold_results], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        sem = std / np.sqrt(len(values)) if len(values) > 0 else 0.0
        ci = 1.96 * sem
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_ci_lower"] = float(mean - ci)
        summary[f"{key}_ci_upper"] = float(mean + ci)
    return summary


def _load_tuned_best_params(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["best_params"]


def _canonical_tuned_model_name(model_name: str) -> str:
    cleaned = model_name.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "lstm": "LSTM",
        "gru": "GRU",
        "cnn-lstm": "CNN-LSTM",
        "cnnlstm": "CNN-LSTM",
        "temporal-gcn": "Temporal GCN",
        "gcn-only": "Temporal GCN",
        "gcn": "Temporal GCN",
        "gcn-lstm": "GCN-LSTM",
        "gcnlstm": "GCN-LSTM",
        "gat-lstm": "GAT-LSTM",
        "gatlstm": "GAT-LSTM",
    }
    if cleaned not in aliases:
        raise ValueError(f"Unsupported tuned model name: {model_name}")
    return aliases[cleaned]


def _tuned_learning_rate(params: dict[str, Any], config: ExperimentConfig) -> float:
    return float(params.get("learning_rate", params.get("lr", config.training.learning_rate)))


def _build_tuned_model(
    model_name: str,
    params: dict[str, Any],
    input_dim: int,
    seq_len: int,
    config: ExperimentConfig,
) -> nn.Module:
    if model_name == "LSTM":
        return SequenceRegressor(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            rnn_type="lstm",
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
        )
    if model_name == "GRU":
        return SequenceRegressor(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            rnn_type="gru",
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
        )
    if model_name == "CNN-LSTM":
        return CNNLSTMRegressor(
            in_channels=input_dim,
            conv_channels=int(params["conv_channels"]),
            lstm_hidden=int(params["lstm_hidden"]),
            kernel_size=int(params["kernel_size"]),
            pool_size=int(params["pool_size"]),
            dropout=float(params["dropout"]),
            num_lstm_layers=int(params["num_lstm_layers"]),
        )
    if model_name == "Temporal GCN":
        return TemporalGCNOnlyModel(
            in_features=input_dim,
            hidden=int(params["hidden_dim"]),
            seq_len=seq_len,
            add_self_loops=config.model.gcn_add_self_loops,
            normalize=config.model.gcn_normalize,
        )
    if model_name == "GCN-LSTM":
        return GCNLSTMRegressor(
            in_features=input_dim,
            gcn_hidden_dim=int(params["gcn_hidden_dim"]),
            lstm_hidden_dim=int(params["lstm_hidden_dim"]),
            gcn_layers=int(params["gcn_layers"]),
            lstm_layers=int(params["lstm_layers"]),
            dropout=float(params["dropout"]),
            add_self_loops=config.model.gcn_add_self_loops,
            normalize=config.model.gcn_normalize,
        )
    if model_name == "GAT-LSTM":
        return TunedGATLSTMRegressor(
            in_features=input_dim,
            gnn_hidden=int(params["gnn_hidden"]),
            lstm_hidden=int(params["lstm_hidden"]),
            gat_layers=int(params["gat_layers"]),
            heads=int(params["heads"]),
            gat_dropout=float(params["gat_dropout"]),
            lstm_layers=int(params["lstm_layers"]),
            lstm_dropout=float(params["lstm_dropout"]),
            add_self_loops=config.model.gat_add_self_loops,
        )
    raise ValueError(f"Unsupported tuned model name: {model_name}")


def evaluate_tuned_model(
    model_name: str,
    params: dict[str, Any],
    train_sequences: list[SequenceSample],
    test_sequences: list[SequenceSample],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Train one tuned model on a fold and evaluate it."""

    input_dim = int(train_sequences[0].x_seq.shape[2])
    model = _build_tuned_model(
        model_name=model_name,
        params=params,
        input_dim=input_dim,
        seq_len=config.data.window_size,
        config=config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_tuned_learning_rate(params, config),
        weight_decay=float(params.get("weight_decay", config.training.weight_decay)),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn="mse",
        scheduler=None,
        logger=None,
        loss_weights=config.loss.as_kwargs(),
        device=device,
    )
    batch_size = int(params.get("batch_size", config.training.batch_size))
    epochs = config.benchmark.gnn_epochs if model_name in {"GCN-LSTM", "GAT-LSTM"} else config.benchmark.baseline_epochs
    trainer.train(
        train_sequences,
        num_epochs=epochs,
        print_every=max(1, epochs // 5),
        early_stopping_patience=min(20, max(5, epochs // 4)),
        warmup_epochs=min(config.training.warmup_epochs, epochs),
        min_delta=config.training.min_delta,
        batch_size=batch_size,
    )

    evaluator = Evaluator(model, device=device)
    evaluation_results = evaluator.evaluate(test_sequences, batch_size=batch_size)
    return {
        "mae_interval": float(evaluation_results["mae_interval"]),
        "mae_min": float(evaluation_results["mae_min"]),
        "mae_max": float(evaluation_results["mae_max"]),
        "mse_interval": float(evaluation_results["mse_interval"]),
        "mse_min": float(evaluation_results["mse_min"]),
        "mse_max": float(evaluation_results["mse_max"]),
    }


def run_final_neural_baseline(
    config: ExperimentConfig,
    family: str,
    tuned_params_file: str | Path,
) -> dict[str, Any]:
    """Train one tuned neural baseline on the train split and evaluate once on the test split."""

    normalized_family = family.strip().lower()
    if normalized_family not in {"lstm", "gru", "cnn_lstm"}:
        raise ValueError("family must be one of: lstm, gru, cnn_lstm")

    config.ensure_output_directories()
    set_seed(config.runtime.random_seed)
    prepared = prepare_dataset(config)
    train_sequences, test_sequences = split_sequences(prepared.sequences, config.data.split_idx)
    device = resolve_device(config.benchmark.device)
    best_params = _load_tuned_best_params(tuned_params_file)
    input_dim = int(train_sequences[0].x_seq.shape[2])

    if normalized_family == "lstm":
        model = SequenceRegressor(
            input_dim=input_dim,
            hidden_dim=int(best_params["hidden_dim"]),
            rnn_type="lstm",
            num_layers=int(best_params["num_layers"]),
            dropout=float(best_params["dropout"]),
        )
        model_label = "LSTM"
    elif normalized_family == "gru":
        model = SequenceRegressor(
            input_dim=input_dim,
            hidden_dim=int(best_params["hidden_dim"]),
            rnn_type="gru",
            num_layers=int(best_params["num_layers"]),
            dropout=float(best_params["dropout"]),
        )
        model_label = "GRU"
    else:
        model = CNNLSTMRegressor(
            in_channels=input_dim,
            conv_channels=int(best_params["conv_channels"]),
            lstm_hidden=int(best_params["lstm_hidden"]),
            kernel_size=int(best_params["kernel_size"]),
            pool_size=int(best_params["pool_size"]),
            dropout=float(best_params["dropout"]),
            num_lstm_layers=int(best_params["num_lstm_layers"]),
        )
        model_label = "CNN-LSTM"

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(best_params["learning_rate"]),
        weight_decay=float(best_params.get("weight_decay", 0.0)),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn="mse",
        scheduler=None,
        logger=None,
        loss_weights=config.loss.as_kwargs(),
        device=device,
    )
    final_batch_size = int(best_params.get("batch_size", config.training.batch_size))
    trainer.train(
        train_sequences,
        num_epochs=config.benchmark.baseline_epochs,
        print_every=max(1, config.benchmark.baseline_epochs // 5),
        early_stopping_patience=min(20, max(5, config.benchmark.baseline_epochs // 4)),
        warmup_epochs=min(config.training.warmup_epochs, config.benchmark.baseline_epochs),
        min_delta=config.training.min_delta,
        batch_size=final_batch_size,
    )

    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(test_sequences, batch_size=final_batch_size)

    run_dir = _benchmark_run_dir(config.benchmark.output_dir or config.paths.logs_dir)
    
    # Plot training loss history
    loss_plot_path = run_dir / f"training_loss_{normalized_family}.png"
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.loss_history, linewidth=2, label="Training Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title(f"{model_label} Training Loss History (Window {config.data.window_size})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    
    # Get predictions on test set for visualization
    from src.utils import iter_sequence_minibatches
    model.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for batch_samples in iter_sequence_minibatches(test_sequences, final_batch_size):
            x_batch = torch.stack([torch.tensor(s.x_seq, dtype=torch.float32) if not isinstance(s.x_seq, torch.Tensor) else s.x_seq for s in batch_samples]).to(device)
            y_batch = torch.stack([s.y_target if isinstance(s.y_target, torch.Tensor) else torch.tensor(s.y_target, dtype=torch.float32) for s in batch_samples])
            
            y_pred_batch = model(x_batch).detach().cpu().numpy()
            y_true_batch = y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch
            all_y_pred.append(y_pred_batch)
            all_y_true.append(y_true_batch)
    
    y_true_all = np.concatenate(all_y_true, axis=0).reshape(-1, 2)
    y_pred_all = np.concatenate(all_y_pred, axis=0).reshape(-1, 2)
    
    # Plot predictions vs actual
    pred_plot_path = run_dir / f"predictions_vs_actual_{normalized_family}.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Min interval plot
    axes[0].scatter(y_true_all[:, 0], y_pred_all[:, 0], alpha=0.5, s=30)
    min_lim = [min(y_true_all[:, 0].min(), y_pred_all[:, 0].min()),
               max(y_true_all[:, 0].max(), y_pred_all[:, 0].max())]
    axes[0].plot(min_lim, min_lim, 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Min Return', fontsize=11)
    axes[0].set_ylabel('Predicted Min Return', fontsize=11)
    axes[0].set_title(f'{model_label} - Min Return Predictions', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Max interval plot
    axes[1].scatter(y_true_all[:, 1], y_pred_all[:, 1], alpha=0.5, s=30, color='orange')
    max_lim = [min(y_true_all[:, 1].min(), y_pred_all[:, 1].min()),
               max(y_true_all[:, 1].max(), y_pred_all[:, 1].max())]
    axes[1].plot(max_lim, max_lim, 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('Actual Max Return', fontsize=11)
    axes[1].set_ylabel('Predicted Max Return', fontsize=11)
    axes[1].set_title(f'{model_label} - Max Return Predictions', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(pred_plot_path, dpi=150)
    plt.close()
    
    payload = {
        "family": normalized_family,
        "model": model_label,
        "window_size": config.data.window_size,
        "graph_mode": config.graph.graph_mode,
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "tuned_params_file": str(Path(tuned_params_file).resolve()),
        "best_params": best_params,
        "final_metrics": {
            "mae_interval": float(metrics["mae_interval"]),
            "mae_min": float(metrics["mae_min"]),
            "mae_max": float(metrics["mae_max"]),
            "mse_interval": float(metrics["mse_interval"]),
            "mse_min": float(metrics["mse_min"]),
            "mse_max": float(metrics["mse_max"]),
        },
        "plots": {
            "training_loss": str(loss_plot_path),
            "predictions_vs_actual": str(pred_plot_path),
        },
    }
    (run_dir / f"final_{normalized_family}_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def _benchmark_run_dir(output_root: Path) -> Path:
    run_dir = output_root / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _graph_assumption_rows(config: ExperimentConfig, sequences: list[SequenceSample]) -> list[tuple[str, str]]:
    directed_edge_counts = [int(sample.edge_index.size(1)) for sample in sequences]
    feature_dim = int(sequences[0].x_seq.shape[2]) if sequences else 0
    if directed_edge_counts:
        min_directed = min(directed_edge_counts)
        max_directed = max(directed_edge_counts)
        mean_directed = float(np.mean(directed_edge_counts))
        if min_directed == max_directed:
            num_edges_value = f"{min_directed} directed / {min_directed // 2} undirected per sequence"
        else:
            num_edges_value = (
                f"mean {mean_directed:.1f} directed "
                f"(min {min_directed}, max {max_directed})"
            )
    else:
        num_edges_value = "N/A"

    return [
        ("feature_set", config.data.feature_set),
        ("input_feature_dim", str(feature_dim)),
        ("graph_mode", config.graph.graph_mode),
        ("gat_self_loops", str(config.model.gat_add_self_loops).lower()),
        ("gcn_self_loops", str(config.model.gcn_add_self_loops).lower()),
        ("gcn_normalize", str(config.model.gcn_normalize).lower()),
        ("num_edges", num_edges_value),
    ]


def _graph_assumption_payload(config: ExperimentConfig, sequences: list[SequenceSample]) -> dict[str, Any]:
    directed_edge_counts = [int(sample.edge_index.size(1)) for sample in sequences]
    payload: dict[str, Any] = {
        "feature_set": config.data.feature_set,
        "input_feature_dim": int(sequences[0].x_seq.shape[2]) if sequences else 0,
        "graph_mode": config.graph.graph_mode,
        "gat_self_loops": config.model.gat_add_self_loops,
        "gcn_self_loops": config.model.gcn_add_self_loops,
        "gcn_normalize": config.model.gcn_normalize,
        "num_sequences": len(sequences),
        "num_edges_directed_unique": sorted(set(directed_edge_counts)),
    }
    if directed_edge_counts:
        payload["num_edges_directed_mean"] = float(np.mean(directed_edge_counts))
        payload["num_edges_directed_min"] = int(min(directed_edge_counts))
        payload["num_edges_directed_max"] = int(max(directed_edge_counts))
        payload["num_edges_undirected_unique"] = sorted({count // 2 for count in directed_edge_counts})
    else:
        payload["num_edges_directed_mean"] = None
        payload["num_edges_directed_min"] = None
        payload["num_edges_directed_max"] = None
        payload["num_edges_undirected_unique"] = []
    return payload


def _write_standard_benchmark_report(
    result_df: pd.DataFrame,
    config: ExperimentConfig,
    run_dir: Path,
    gat_metrics: dict[str, float],
    train_count: int,
    test_count: int,
    graph_assumptions: list[tuple[str, str]],
) -> None:
    report_path = run_dir / "benchmark_report.md"
    best_model = result_df.iloc[0]["model"]
    best_mae = float(result_df.iloc[0]["mae_interval"])
    gat_row = result_df[result_df["model"] == "GAT-LSTM"]
    gat_mae = float(gat_row.iloc[0]["mae_interval"]) if not gat_row.empty else None

    lines = [
        "# Benchmark Report: GAT-LSTM vs Baselines",
        "",
        f"- Run time: {datetime.now().isoformat()}",
        f"- Dataset: {config.data.daily_file}",
        f"- Aggregation mode: {config.data.aggregation_mode}",
        f"- Window size: {config.data.window_size}",
        f"- Split index: {config.data.split_idx}",
        f"- Train sequences: {train_count}",
        f"- Test sequences: {test_count}",
        "",
        "## Graph Assumptions",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key, value in graph_assumptions:
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
        "## Proposed Model Setup",
        "",
        f"- GAT model type: {config.model.model_type}",
        "- GAT loss: mse",
        f"- GAT seeds: {config.benchmark.seeds}",
        (
            f"- GAT MAE(interval) mean±std: "
            f"{gat_metrics['mae_interval_mean']:.6f}±{gat_metrics['mae_interval_std']:.6f}"
        ),
        "",
        "## Ranking by MAE (interval)",
        "",
        "```text",
        result_df.to_string(index=False),
        "```",
        "",
        f"- Best model: **{best_model}** (MAE interval = {best_mae:.6f})",
        ]
    )
    if gat_mae is not None:
        lines.append(f"- GAT-LSTM MAE interval: **{gat_mae:.6f}**")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_crossval_benchmark_report(
    result_df: pd.DataFrame,
    config: ExperimentConfig,
    run_dir: Path,
    graph_assumptions: list[tuple[str, str]],
) -> None:
    report_path = run_dir / "benchmark_report.md"
    best_model = result_df.iloc[0]["model"]
    best_mae = float(result_df.iloc[0]["mae_interval_mean"])
    protocol_label = "Walk-Forward" if config.benchmark.protocol == "walk_forward" else "Cross-Validation"
    actual_folds = int(result_df["n_folds"].max()) if "n_folds" in result_df.columns else config.benchmark.n_folds
    total_runs = int(result_df["n_runs"].max()) if "n_runs" in result_df.columns else actual_folds * config.benchmark.n_runs
    runs_per_fold = total_runs // actual_folds if actual_folds else config.benchmark.n_runs

    lines = [
        f"# Benchmark Report: GAT-LSTM vs Baselines ({protocol_label})",
        "",
        "## Experimental Protocol",
        "",
        "- All models were trained with MSE loss for a fairer comparison.",
        "- GCN-only baseline used the full temporal sequence rather than the last frame only.",
        f"- Feature set: {config.data.feature_set}",
        f"- Protocol: {config.benchmark.protocol}",
        f"- Folds evaluated: {actual_folds}",
        f"- Runs per fold: {runs_per_fold}",
        f"- Min train size: {config.benchmark.min_train_size}",
        f"- Test step: {config.benchmark.test_step}",
        f"- Seeds: {config.benchmark.seeds}",
        "",
        "## Graph Assumptions",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key, value in graph_assumptions:
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
        "## Final Rankings",
        "",
        "| Rank | Model | MAE Mean | MAE Std | 95% CI Lower | 95% CI Upper | Total Runs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for rank, (_, row) in enumerate(result_df.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['mae_interval_mean']:.6f} | "
            f"{row['mae_interval_std']:.6f} | {row['mae_interval_ci_lower']:.6f} | "
            f"{row['mae_interval_ci_upper']:.6f} | {int(row['n_runs'])} |"
        )

    lines.extend(
        [
            "",
            "## Winner",
            "",
            f"**{best_model}** achieved the best mean interval MAE of **{best_mae:.6f}**.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_standard_benchmark(config: ExperimentConfig) -> pd.DataFrame:
    """Run the standard fixed-split benchmark suite."""

    config.ensure_output_directories()
    set_seed(config.runtime.random_seed)
    prepared = prepare_dataset(config)
    train_sequences, test_sequences = split_sequences(prepared.sequences, config.data.split_idx)
    device = resolve_device(config.benchmark.device)

    x_train_tab, y_train_tab = flatten_for_tabular(train_sequences)
    x_test_tab, y_test_tab = flatten_for_tabular(test_sequences)
    x_train_seq, y_train_seq = flatten_for_sequence_models(train_sequences)
    x_test_seq, _ = flatten_for_sequence_models(test_sequences)

    results: list[dict[str, object]] = []

    def add_result(group: str, model_name: str, metrics: dict[str, float], note: str = "") -> None:
        results.append(
            {
                "feature_set": config.data.feature_set,
                "group": group,
                "model": model_name,
                **metrics,
                "note": note,
            }
        )

    gbt_pred = train_sklearn_model(
        GradientBoostingRegressor,
        {"n_estimators": 300, "learning_rate": 0.05},
        x_train_tab,
        y_train_tab,
        x_test_tab,
        config.runtime.random_seed,
    )
    add_result("Group 1", "GBT", compute_metrics(y_test_tab, gbt_pred), "Gradient boosting baseline")

    rf_pred = train_sklearn_model(
        RandomForestRegressor,
        {"n_estimators": 500, "n_jobs": -1},
        x_train_tab,
        y_train_tab,
        x_test_tab,
        config.runtime.random_seed,
    )
    add_result("Group 1", "RF", compute_metrics(y_test_tab, rf_pred), "Random forest baseline")

    svr_pred = train_sklearn_model(
        SVR,
        {"kernel": "rbf", "C": 1.0, "epsilon": 0.01},
        x_train_tab,
        y_train_tab,
        x_test_tab,
        config.runtime.random_seed,
    )
    add_result("Group 1", "SVR", compute_metrics(y_test_tab, svr_pred), "Support vector regression baseline")

    dnn = MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=config.runtime.random_seed,
        )
    )
    dnn.fit(x_train_tab, y_train_tab)
    add_result("Group 1", "DNN", compute_metrics(y_test_tab, dnn.predict(x_test_tab)), "Static MLP baseline")

    glm = MultiOutputRegressor(LinearRegression())
    glm.fit(x_train_tab, y_train_tab)
    add_result("Group 1", "GLM", compute_metrics(y_test_tab, glm.predict(x_test_tab)), "Linear baseline")

    gat_seed_metrics = []
    for seed in config.benchmark.seeds:
        set_seed(seed)
        gat_seed_metrics.append(
            evaluate_gat_lstm(
                train_sequences=train_sequences,
                test_sequences=test_sequences,
                config=config,
                device=device,
                loss_name="mse",
                epochs=config.benchmark.gnn_epochs,
            )
        )
    gat_metrics = aggregate_metric_dicts(gat_seed_metrics)
    add_result(
        "Proposed",
        "GAT-LSTM",
        {
            "mae_interval": gat_metrics["mae_interval_mean"],
            "mae_min": gat_metrics["mae_min_mean"],
            "mae_max": gat_metrics["mae_max_mean"],
            "mse_interval": gat_metrics["mse_interval_mean"],
            "mse_min": gat_metrics["mse_min_mean"],
            "mse_max": gat_metrics["mse_max_mean"],
        },
        f"Mean±std across seeds: {gat_metrics['mae_interval_mean']:.6f}±{gat_metrics['mae_interval_std']:.6f}",
    )

    lstm_pred = train_torch_regressor(
        SequenceRegressor(input_dim=x_train_seq.shape[2], hidden_dim=96, rnn_type="lstm"),
        x_train_seq,
        y_train_seq,
        x_test_seq,
        epochs=config.benchmark.baseline_epochs,
        lr=1e-3,
        device=device,
    )
    add_result("Group 2", "Standalone LSTM", compute_metrics(y_test_tab, lstm_pred), "Temporal-only baseline")

    gru_pred = train_torch_regressor(
        SequenceRegressor(input_dim=x_train_seq.shape[2], hidden_dim=96, rnn_type="gru"),
        x_train_seq,
        y_train_seq,
        x_test_seq,
        epochs=config.benchmark.baseline_epochs,
        lr=1e-3,
        device=device,
    )
    add_result("Group 2", "GRU", compute_metrics(y_test_tab, gru_pred), "Temporal-only baseline")

    cnn_lstm_pred = train_torch_regressor(
        CNNLSTMRegressor(in_channels=x_train_seq.shape[2], conv_channels=32, lstm_hidden=96),
        x_train_seq,
        y_train_seq,
        x_test_seq,
        epochs=config.benchmark.baseline_epochs,
        lr=1e-3,
        device=device,
    )
    add_result("Group 3", "CNN-LSTM", compute_metrics(y_test_tab, cnn_lstm_pred), "Spatial-temporal baseline")

    gcn_metrics = evaluate_temporal_gcn_only(
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        epochs=config.benchmark.baseline_epochs,
        device=device,
        seq_len=config.data.window_size,
        config=config,
    )
    add_result("Group 3", "Standalone GCN", gcn_metrics, "Spatial-only baseline")

    if config.benchmark.include_statistical_models:
        split_point = len(train_sequences)
        var_metrics = evaluate_var_or_fallback(prepared.y_full, config.data.window_size, split_point, model_name="VAR")
        add_result("Group 4", "VAR", var_metrics, "Classical multivariate time-series baseline")
        naive_metrics = evaluate_var_or_fallback(prepared.y_full, config.data.window_size, split_point, model_name="Naive")
        add_result("Group 4", "Naive Persistence", naive_metrics, "Persistence sanity check")

    result_df = pd.DataFrame(results).sort_values(by="mae_interval", ascending=True).reset_index(drop=True)
    run_dir = _benchmark_run_dir(config.benchmark.output_dir)
    graph_assumptions = _graph_assumption_rows(config, prepared.sequences)
    graph_assumption_payload = _graph_assumption_payload(config, prepared.sequences)
    result_df.to_csv(run_dir / "benchmark_results.csv", index=False)
    (run_dir / "benchmark_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "graph_assumptions.json").write_text(
        json.dumps(graph_assumption_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_standard_benchmark_report(
        result_df,
        config,
        run_dir,
        gat_metrics,
        len(train_sequences),
        len(test_sequences),
        graph_assumptions,
    )
    return result_df


def _materialise_expanding_window_fold(
    prepared,
    window_size: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> tuple[list[SequenceSample], list[SequenceSample]]:
    scaler = FeatureEngineer.fit_scaler_on_train_sequences(
        prepared.x_full_raw,
        train_sequence_count=train_end,
        window_size=window_size,
    )
    x_full_scaled = FeatureEngineer.transform_with_scaler(prepared.x_full_raw, scaler)
    train_sequences = materialise_sequences_from_templates(
        x_full=x_full_scaled,
        y_full=prepared.y_full,
        sequence_templates=prepared.sequence_templates,
        window_size=window_size,
        sequence_indices=list(range(train_end)),
    )
    test_sequences = materialise_sequences_from_templates(
        x_full=x_full_scaled,
        y_full=prepared.y_full,
        sequence_templates=prepared.sequence_templates,
        window_size=window_size,
        sequence_indices=list(range(test_start, test_end)),
    )
    return train_sequences, test_sequences


def _restrict_prepared_to_periods(
    prepared: PreparedDataset,
    common_periods: list[Any],
    window_size: int,
) -> PreparedDataset:
    common_set = set(common_periods)
    positions = [index for index, label in enumerate(prepared.valid_indices_map) if label in common_set]
    if len(positions) < window_size + 2:
        raise ValueError("Common period set is too small for the configured window size.")

    expected_positions = list(range(positions[0], positions[-1] + 1))
    if positions != expected_positions:
        raise ValueError("Common periods must form one contiguous chronological block.")

    start = positions[0]
    stop = positions[-1] + 1
    filtered_periods = prepared.valid_indices_map[start:stop]
    if filtered_periods != common_periods:
        raise ValueError("Prepared dataset period order does not match the common-period order.")

    sequence_count = len(filtered_periods) - window_size
    if sequence_count < 2:
        raise ValueError("Common period block does not create enough sequences.")

    return PreparedDataset(
        daily_frame=prepared.daily_frame,
        stock_codes=prepared.stock_codes,
        x_full=prepared.x_full_raw[start:stop],
        x_full_raw=prepared.x_full_raw[start:stop],
        y_full=prepared.y_full[start:stop],
        feature_names=prepared.feature_names,
        valid_indices_map=filtered_periods,
        static_edge_index=prepared.static_edge_index,
        sequence_templates=prepared.sequence_templates[start : start + sequence_count],
        sequences=[],
    )


def _common_contiguous_periods(prepared_datasets: list[PreparedDataset]) -> list[Any]:
    if len(prepared_datasets) < 2:
        raise ValueError("At least two prepared datasets are required.")

    common_set = set(prepared_datasets[0].valid_indices_map)
    for prepared in prepared_datasets[1:]:
        common_set &= set(prepared.valid_indices_map)

    ordered_common = [label for label in prepared_datasets[0].valid_indices_map if label in common_set]
    if not ordered_common:
        return []

    position_maps = [
        {label: index for index, label in enumerate(prepared.valid_indices_map)}
        for prepared in prepared_datasets
    ]

    blocks: list[list[Any]] = []
    current_block: list[Any] = []
    previous_positions: list[int] | None = None

    for label in ordered_common:
        positions = [position_map[label] for position_map in position_maps]
        is_contiguous = (
            previous_positions is None
            or all(position == previous_position + 1 for position, previous_position in zip(positions, previous_positions))
        )
        if not is_contiguous and current_block:
            blocks.append(current_block)
            current_block = []
        current_block.append(label)
        previous_positions = positions

    if current_block:
        blocks.append(current_block)

    return max(blocks, key=len)


def run_walk_forward_benchmark(
    config: ExperimentConfig,
    prepared: PreparedDataset | None = None,
    run_dir: Path | None = None,
    protocol_label: str = "walk_forward",
    common_periods: list[Any] | None = None,
) -> pd.DataFrame:
    """Run an expanding-window benchmark that only trains on past periods."""

    config.ensure_output_directories()
    set_seed(config.runtime.random_seed)
    if prepared is None:
        prepared = prepare_dataset(config)
    device = resolve_device(config.benchmark.device)

    num_sequences = len(prepared.sequence_templates)
    first_test_start = min(config.benchmark.min_train_size, num_sequences - 1)
    if first_test_start < 1:
        raise ValueError("Not enough sequences for walk-forward benchmarking.")

    fold_indices = []
    for test_start in range(first_test_start, num_sequences, config.benchmark.test_step):
        test_end = min(num_sequences, test_start + config.benchmark.test_step)
        if test_start < test_end:
            fold_indices.append((test_start, test_end))

    all_results: dict[str, list[dict[str, float]]] = {
        "LSTM": [],
        "GRU": [],
        "CNN-LSTM": [],
        "GCN-only": [],
        "GAT-LSTM": [],
    }

    fold_iterator = _progress(
        enumerate(fold_indices, start=1),
        total=len(fold_indices),
        desc=f"{config.data.feature_set} folds",
        unit="fold",
    )
    for fold_number, (fold_start, fold_end) in fold_iterator:
        train_sequences, test_sequences = _materialise_expanding_window_fold(
            prepared=prepared,
            window_size=config.data.window_size,
            train_end=fold_start,
            test_start=fold_start,
            test_end=fold_end,
        )
        if not train_sequences or not test_sequences:
            continue

        x_train_tab, y_train_tab = flatten_for_tabular(train_sequences)
        x_test_tab, y_test_tab = flatten_for_tabular(test_sequences)
        x_train_seq, y_train_seq = flatten_for_sequence_models(train_sequences)
        x_test_seq, _ = flatten_for_sequence_models(test_sequences)

        fold_results: dict[str, list[dict[str, float]]] = {key: [] for key in all_results}

        run_iterator = _progress(
            range(config.benchmark.n_runs),
            desc=f"fold {fold_number}/{len(fold_indices)} runs",
            unit="run",
            leave=False,
        )
        for run_index in run_iterator:
            seed = config.benchmark.seeds[run_index % len(config.benchmark.seeds)]
            set_seed(seed)

            with _progress(
                total=len(fold_results),
                desc=f"fold {fold_number} run {run_index + 1} models",
                unit="model",
                leave=False,
            ) as model_progress:
                lstm_pred = train_torch_regressor(
                    SequenceRegressor(input_dim=x_train_seq.shape[2], hidden_dim=96, rnn_type="lstm"),
                    x_train_seq,
                    y_train_seq,
                    x_test_seq,
                    epochs=config.benchmark.baseline_epochs,
                    lr=1e-3,
                    device=device,
                )
                fold_results["LSTM"].append(compute_metrics(y_test_tab, lstm_pred))
                _advance_progress(model_progress, "LSTM")

                gru_pred = train_torch_regressor(
                    SequenceRegressor(input_dim=x_train_seq.shape[2], hidden_dim=96, rnn_type="gru"),
                    x_train_seq,
                    y_train_seq,
                    x_test_seq,
                    epochs=config.benchmark.baseline_epochs,
                    lr=1e-3,
                    device=device,
                )
                fold_results["GRU"].append(compute_metrics(y_test_tab, gru_pred))
                _advance_progress(model_progress, "GRU")

                cnn_lstm_pred = train_torch_regressor(
                    CNNLSTMRegressor(in_channels=x_train_seq.shape[2], conv_channels=32, lstm_hidden=96),
                    x_train_seq,
                    y_train_seq,
                    x_test_seq,
                    epochs=config.benchmark.baseline_epochs,
                    lr=1e-3,
                    device=device,
                )
                fold_results["CNN-LSTM"].append(compute_metrics(y_test_tab, cnn_lstm_pred))
                _advance_progress(model_progress, "CNN-LSTM")

                fold_results["GCN-only"].append(
                    evaluate_temporal_gcn_only(
                        train_sequences,
                        test_sequences,
                        epochs=config.benchmark.baseline_epochs,
                        device=device,
                        seq_len=config.data.window_size,
                        config=config,
                    )
                )
                _advance_progress(model_progress, "GCN-only")

                fold_results["GAT-LSTM"].append(
                    evaluate_gat_lstm(
                        train_sequences=train_sequences,
                        test_sequences=test_sequences,
                        config=config,
                        device=device,
                        loss_name="mse",
                        epochs=config.benchmark.gnn_epochs,
                    )
                )
                _advance_progress(model_progress, "GAT-LSTM")

        for model_name, metrics_list in fold_results.items():
            all_results[model_name].append(aggregate_fold_results(metrics_list))

    final_rows = []
    for model_name, folds in all_results.items():
        if not folds:
            continue
        fold_means = np.array([fold["mae_interval_mean"] for fold in folds], dtype=np.float64)
        mean = float(fold_means.mean())
        std = float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0
        sem = std / np.sqrt(len(fold_means)) if len(fold_means) > 0 else 0.0
        ci = 1.96 * sem

        final_rows.append(
            {
                "feature_set": config.data.feature_set,
                "model": model_name,
                "mae_interval_mean": mean,
                "mae_interval_std": std,
                "mae_interval_ci_lower": float(mean - ci),
                "mae_interval_ci_upper": float(mean + ci),
                "n_folds": len(fold_means),
                "n_runs": len(fold_means) * config.benchmark.n_runs,
            }
        )

    result_df = pd.DataFrame(final_rows).sort_values(by="mae_interval_mean", ascending=True).reset_index(drop=True)
    run_dir = run_dir or _benchmark_run_dir(config.benchmark.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_assumptions = _graph_assumption_rows(config, prepared.sequence_templates)
    graph_assumption_payload = _graph_assumption_payload(config, prepared.sequence_templates)
    graph_assumption_payload.update(
        {
            "protocol": protocol_label,
            "min_train_size": config.benchmark.min_train_size,
            "test_step": config.benchmark.test_step,
            "folds": fold_indices,
            "common_period_count": len(common_periods) if common_periods is not None else None,
            "common_period_start": common_periods[0] if common_periods else None,
            "common_period_end": common_periods[-1] if common_periods else None,
        }
    )
    result_df.to_csv(run_dir / "benchmark_results.csv", index=False)
    (run_dir / "benchmark_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "graph_assumptions.json").write_text(
        json.dumps(graph_assumption_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_crossval_benchmark_report(result_df, config, run_dir, graph_assumptions)
    return result_df


def run_tuned_walk_forward_benchmark(
    config: ExperimentConfig,
    tuned_param_files: dict[str, str | Path],
    prepared: PreparedDataset | None = None,
    run_dir: Path | None = None,
    protocol_label: str = "tuned_walk_forward",
) -> pd.DataFrame:
    """Run walk-forward evaluation using Optuna best_params files."""

    config.ensure_output_directories()
    set_seed(config.runtime.random_seed)
    if prepared is None:
        prepared = prepare_dataset(config)
    device = resolve_device(config.benchmark.device)

    tuned_payloads: dict[str, dict[str, Any]] = {}
    tuned_sources: dict[str, str] = {}
    for raw_name, params_file in tuned_param_files.items():
        model_name = _canonical_tuned_model_name(raw_name)
        if model_name in tuned_payloads:
            raise ValueError(f"Duplicate tuned model mapping for {model_name}")
        params_path = Path(params_file)
        tuned_payloads[model_name] = _load_tuned_best_params(params_path)
        tuned_sources[model_name] = str(params_path)

    if not tuned_payloads:
        raise ValueError("At least one tuned model params file is required.")

    num_sequences = len(prepared.sequence_templates)
    first_test_start = min(config.benchmark.min_train_size, num_sequences - 1)
    if first_test_start < 1:
        raise ValueError("Not enough sequences for tuned walk-forward benchmarking.")

    fold_indices = []
    for test_start in range(first_test_start, num_sequences, config.benchmark.test_step):
        test_end = min(num_sequences, test_start + config.benchmark.test_step)
        if test_start < test_end:
            fold_indices.append((test_start, test_end))

    all_results: dict[str, list[dict[str, float]]] = {model_name: [] for model_name in tuned_payloads}
    fold_iterator = _progress(
        enumerate(fold_indices, start=1),
        total=len(fold_indices),
        desc=f"{config.data.feature_set} tuned folds",
        unit="fold",
    )
    for fold_number, (fold_start, fold_end) in fold_iterator:
        train_sequences, test_sequences = _materialise_expanding_window_fold(
            prepared=prepared,
            window_size=config.data.window_size,
            train_end=fold_start,
            test_start=fold_start,
            test_end=fold_end,
        )
        if not train_sequences or not test_sequences:
            continue

        fold_results: dict[str, list[dict[str, float]]] = {key: [] for key in all_results}
        run_iterator = _progress(
            range(config.benchmark.n_runs),
            desc=f"fold {fold_number}/{len(fold_indices)} tuned runs",
            unit="run",
            leave=False,
        )
        for run_index in run_iterator:
            seed = config.benchmark.seeds[run_index % len(config.benchmark.seeds)]
            set_seed(seed)
            with _progress(
                total=len(tuned_payloads),
                desc=f"fold {fold_number} run {run_index + 1} tuned models",
                unit="model",
                leave=False,
            ) as model_progress:
                for model_name, params in tuned_payloads.items():
                    fold_results[model_name].append(
                        evaluate_tuned_model(
                            model_name=model_name,
                            params=params,
                            train_sequences=train_sequences,
                            test_sequences=test_sequences,
                            config=config,
                            device=device,
                        )
                    )
                    _advance_progress(model_progress, model_name)

        for model_name, metrics_list in fold_results.items():
            all_results[model_name].append(aggregate_fold_results(metrics_list))

    final_rows = []
    for model_name, folds in all_results.items():
        if not folds:
            continue
        fold_means = np.array([fold["mae_interval_mean"] for fold in folds], dtype=np.float64)
        mean = float(fold_means.mean())
        std = float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0
        sem = std / np.sqrt(len(fold_means)) if len(fold_means) > 0 else 0.0
        ci = 1.96 * sem
        final_rows.append(
            {
                "feature_set": config.data.feature_set,
                "graph_mode": config.graph.graph_mode,
                "model": model_name,
                "mae_interval_mean": mean,
                "mae_interval_std": std,
                "mae_interval_ci_lower": float(mean - ci),
                "mae_interval_ci_upper": float(mean + ci),
                "n_folds": len(fold_means),
                "n_runs": len(fold_means) * config.benchmark.n_runs,
                "tuned_params_file": tuned_sources[model_name],
            }
        )

    result_df = pd.DataFrame(final_rows).sort_values(by="mae_interval_mean", ascending=True).reset_index(drop=True)
    run_dir = run_dir or _benchmark_run_dir(config.benchmark.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_assumptions = _graph_assumption_rows(config, prepared.sequence_templates)
    graph_assumption_payload = _graph_assumption_payload(config, prepared.sequence_templates)
    graph_assumption_payload.update(
        {
            "protocol": protocol_label,
            "min_train_size": config.benchmark.min_train_size,
            "test_step": config.benchmark.test_step,
            "folds": fold_indices,
        }
    )
    result_df.to_csv(run_dir / "tuned_benchmark_results.csv", index=False)
    (run_dir / "benchmark_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "tuned_params_manifest.json").write_text(
        json.dumps(tuned_sources, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "graph_assumptions.json").write_text(
        json.dumps(graph_assumption_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_crossval_benchmark_report(result_df, config, run_dir, graph_assumptions)
    return result_df


def run_common_period_feature_benchmark(
    configs: list[ExperimentConfig],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run walk-forward benchmarks after forcing all configs onto one common period block."""

    if len(configs) < 2:
        raise ValueError("At least two configs are required for a common-period benchmark.")

    for config in configs:
        config.resolve_paths()

    prepared_items = [(config, prepare_dataset(config)) for config in configs]
    common_periods = _common_contiguous_periods([prepared for _, prepared in prepared_items])
    if not common_periods:
        raise ValueError("No common contiguous valid periods across feature-set configs.")

    output_root = Path(output_dir) if output_dir is not None else configs[0].benchmark.output_dir / "common_period"
    if not output_root.is_absolute():
        output_root = (configs[0].paths.project_root / output_root).resolve()
    common_run_dir = _benchmark_run_dir(output_root)

    combined_rows = []
    common_summary = {
        "protocol": "common_period_walk_forward",
        "feature_sets": [config.data.feature_set for config, _ in prepared_items],
        "common_period_count": len(common_periods),
        "common_period_start": common_periods[0],
        "common_period_end": common_periods[-1],
        "window_size": configs[0].data.window_size,
        "min_train_size": configs[0].benchmark.min_train_size,
        "test_step": configs[0].benchmark.test_step,
        "items": [],
    }

    for config, prepared in prepared_items:
        restricted = _restrict_prepared_to_periods(
            prepared=prepared,
            common_periods=common_periods,
            window_size=config.data.window_size,
        )
        feature_run_dir = common_run_dir / config.data.feature_set
        result_df = run_walk_forward_benchmark(
            config=config,
            prepared=restricted,
            run_dir=feature_run_dir,
            protocol_label="common_period_walk_forward",
            common_periods=common_periods,
        )
        combined_rows.append(result_df)
        common_summary["items"].append(
            {
                "feature_set": config.data.feature_set,
                "feature_names": restricted.feature_names,
                "input_feature_dim": len(restricted.feature_names),
                "periods": len(restricted.valid_indices_map),
                "sequences": len(restricted.sequence_templates),
                "run_dir": str(feature_run_dir),
            }
        )

    combined = pd.concat(combined_rows, ignore_index=True)
    combined = combined.sort_values(["model", "mae_interval_mean", "feature_set"]).reset_index(drop=True)
    combined.to_csv(common_run_dir / "common_period_benchmark_results.csv", index=False)
    (common_run_dir / "common_period_summary.json").write_text(
        json.dumps(common_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return combined


def run_cross_validation_benchmark(config: ExperimentConfig) -> pd.DataFrame:
    """Run the improved multi-fold benchmark protocol."""

    config.ensure_output_directories()
    set_seed(config.runtime.random_seed)
    prepared = prepare_dataset(config)
    sequences = prepared.sequences
    device = resolve_device(config.benchmark.device)

    n_folds = min(config.benchmark.n_folds, len(sequences))
    if n_folds < 2:
        raise ValueError("At least two folds are required for cross-validation benchmarking.")

    fold_size = max(1, len(sequences) // n_folds)
    fold_indices = []
    for fold_index in range(n_folds):
        start = fold_index * fold_size
        end = len(sequences) if fold_index == n_folds - 1 else min(len(sequences), (fold_index + 1) * fold_size)
        fold_indices.append((start, end))

    all_results: dict[str, list[dict[str, float]]] = {
        "GBT": [],
        "RF": [],
        "SVR": [],
        "GLM": [],
        "DNN": [],
        "LSTM": [],
        "GRU": [],
        "CNN-LSTM": [],
        "GCN-only": [],
        "GAT-LSTM": [],
    }

    for fold_start, fold_end in fold_indices:
        test_sequences = sequences[fold_start:fold_end]
        train_sequences = sequences[:fold_start] + sequences[fold_end:]
        if not train_sequences or not test_sequences:
            continue

        x_train_tab, y_train_tab = flatten_for_tabular(train_sequences)
        x_test_tab, y_test_tab = flatten_for_tabular(test_sequences)
        x_train_seq, y_train_seq = flatten_for_sequence_models(train_sequences)
        x_test_seq, _ = flatten_for_sequence_models(test_sequences)

        fold_results: dict[str, list[dict[str, float]]] = {key: [] for key in all_results}

        for run_index in range(config.benchmark.n_runs):
            seed = config.benchmark.seeds[run_index % len(config.benchmark.seeds)]
            set_seed(seed)

            fold_results["GBT"].append(
                compute_metrics(
                    y_test_tab,
                    train_sklearn_model(
                        GradientBoostingRegressor,
                        {"n_estimators": 300, "learning_rate": 0.05},
                        x_train_tab,
                        y_train_tab,
                        x_test_tab,
                        seed,
                    ),
                )
            )
            fold_results["RF"].append(
                compute_metrics(
                    y_test_tab,
                    train_sklearn_model(
                        RandomForestRegressor,
                        {"n_estimators": 100, "n_jobs": -1},
                        x_train_tab,
                        y_train_tab,
                        x_test_tab,
                        seed,
                    ),
                )
            )
            fold_results["SVR"].append(
                compute_metrics(
                    y_test_tab,
                    train_sklearn_model(
                        SVR,
                        {"kernel": "rbf", "C": 1.0, "epsilon": 0.01},
                        x_train_tab,
                        y_train_tab,
                        x_test_tab,
                        seed,
                    ),
                )
            )

            glm = MultiOutputRegressor(LinearRegression()).fit(x_train_tab, y_train_tab)
            fold_results["GLM"].append(compute_metrics(y_test_tab, glm.predict(x_test_tab)))

            dnn = MultiOutputRegressor(
                MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    random_state=seed,
                )
            ).fit(x_train_tab, y_train_tab)
            fold_results["DNN"].append(compute_metrics(y_test_tab, dnn.predict(x_test_tab)))

            lstm_pred = train_torch_regressor(
                SequenceRegressor(input_dim=x_train_seq.shape[2], hidden_dim=96, rnn_type="lstm"),
                x_train_seq,
                y_train_seq,
                x_test_seq,
                epochs=config.benchmark.baseline_epochs,
                lr=1e-3,
                device=device,
            )
            fold_results["LSTM"].append(compute_metrics(y_test_tab, lstm_pred))

            gru_pred = train_torch_regressor(
                SequenceRegressor(input_dim=x_train_seq.shape[2], hidden_dim=96, rnn_type="gru"),
                x_train_seq,
                y_train_seq,
                x_test_seq,
                epochs=config.benchmark.baseline_epochs,
                lr=1e-3,
                device=device,
            )
            fold_results["GRU"].append(compute_metrics(y_test_tab, gru_pred))

            cnn_lstm_pred = train_torch_regressor(
                CNNLSTMRegressor(in_channels=x_train_seq.shape[2], conv_channels=32, lstm_hidden=96),
                x_train_seq,
                y_train_seq,
                x_test_seq,
                epochs=config.benchmark.baseline_epochs,
                lr=1e-3,
                device=device,
            )
            fold_results["CNN-LSTM"].append(compute_metrics(y_test_tab, cnn_lstm_pred))

            fold_results["GCN-only"].append(
                evaluate_temporal_gcn_only(
                    train_sequences,
                    test_sequences,
                    epochs=config.benchmark.baseline_epochs,
                    device=device,
                    seq_len=config.data.window_size,
                    config=config,
                )
            )
            fold_results["GAT-LSTM"].append(
                evaluate_gat_lstm(
                    train_sequences=train_sequences,
                    test_sequences=test_sequences,
                    config=config,
                    device=device,
                    loss_name="mse",
                    epochs=config.benchmark.gnn_epochs,
                )
            )

        for model_name, metrics_list in fold_results.items():
            all_results[model_name].append(aggregate_fold_results(metrics_list))

    final_rows = []
    for model_name, folds in all_results.items():
        fold_means = np.array([fold["mae_interval_mean"] for fold in folds], dtype=np.float64)
        mean = float(fold_means.mean())
        std = float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0
        sem = std / np.sqrt(len(fold_means)) if len(fold_means) > 0 else 0.0
        ci = 1.96 * sem

        final_rows.append(
            {
                "feature_set": config.data.feature_set,
                "model": model_name,
                "mae_interval_mean": mean,
                "mae_interval_std": std,
                "mae_interval_ci_lower": float(mean - ci),
                "mae_interval_ci_upper": float(mean + ci),
                "n_folds": len(fold_means),
                "n_runs": len(fold_means) * config.benchmark.n_runs,
            }
        )

    result_df = pd.DataFrame(final_rows).sort_values(by="mae_interval_mean", ascending=True).reset_index(drop=True)
    run_dir = _benchmark_run_dir(config.benchmark.output_dir)
    graph_assumptions = _graph_assumption_rows(config, prepared.sequences)
    graph_assumption_payload = _graph_assumption_payload(config, prepared.sequences)
    result_df.to_csv(run_dir / "benchmark_results.csv", index=False)
    (run_dir / "benchmark_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "graph_assumptions.json").write_text(
        json.dumps(graph_assumption_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_crossval_benchmark_report(result_df, config, run_dir, graph_assumptions)
    return result_df


def combine_benchmark_results(csv_files: list[str | Path], output_path: str | Path | None = None) -> pd.DataFrame:
    """Combine multiple benchmark CSV files into a single leaderboard."""

    frames = [pd.read_csv(path) for path in csv_files]
    combined = pd.concat(frames, ignore_index=True)

    if "mae_interval" in combined.columns:
        sort_column = "mae_interval"
    elif "mae_interval_mean" in combined.columns:
        sort_column = "mae_interval_mean"
    else:
        raise ValueError("Could not find an MAE ranking column in the provided files.")

    combined = combined.sort_values(sort_column, ascending=True).reset_index(drop=True)
    combined.insert(0, "rank", range(1, len(combined) + 1))

    if output_path:
        Path(output_path).write_text(combined.to_csv(index=False), encoding="utf-8")

    return combined
