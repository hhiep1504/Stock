"""GAT-LSTM model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.config import ModelConfig


class StockGATLSTM(nn.Module):
    """Single-layer GAT encoder followed by a temporal LSTM."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int = 4,
        gnn_hidden: int = 128,
        lstm_hidden: int = 256,
        heads: int = 1,
        dropout: float = 0.6,
        add_self_loops: bool = True,
    ):
        super().__init__()
        del num_nodes

        self.conv1 = GATConv(
            in_features,
            gnn_hidden,
            heads=heads,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )
        self.lstm = nn.LSTM(
            input_size=gnn_hidden * heads,
            hidden_size=lstm_hidden,
            batch_first=True,
            dropout=0.2,
        )
        self.linear = nn.Linear(lstm_hidden, 2)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            gnn_outputs = []
            for timestep in range(seq_len):
                encoded = self.conv1(x_seq[:, timestep].reshape(batch_size * num_nodes, -1), edge_index)
                gnn_outputs.append(F.elu(encoded).view(batch_size, num_nodes, -1))

            spatial_sequence = torch.stack(gnn_outputs, dim=1).reshape(batch_size * num_nodes, seq_len, -1)
            _, (hidden_state, _) = self.lstm(spatial_sequence)
            prediction = self.linear(hidden_state.squeeze(0))
            return prediction.view(batch_size, num_nodes, 2)

        gnn_outputs = []
        for timestep in range(x_seq.size(0)):
            encoded = self.conv1(x_seq[timestep], edge_index)
            gnn_outputs.append(F.elu(encoded))

        spatial_sequence = torch.stack(gnn_outputs, dim=0)
        lstm_input = spatial_sequence.permute(1, 0, 2)
        _, (hidden_state, _) = self.lstm(lstm_input)
        return self.linear(hidden_state.squeeze(0))


class DeepStockGATLSTM(nn.Module):
    """Two-layer GAT encoder followed by a temporal LSTM."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int = 4,
        gnn_hidden: int = 128,
        lstm_hidden: int = 256,
        heads: int = 2,
        dropout: float = 0.1,
        add_self_loops: bool = True,
    ):
        super().__init__()
        del num_nodes

        self.conv1 = GATConv(
            in_features,
            gnn_hidden,
            heads=heads,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )
        self.conv2 = GATConv(
            gnn_hidden * heads,
            gnn_hidden,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )
        self.lstm = nn.LSTM(input_size=gnn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.linear = nn.Linear(lstm_hidden, 2)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            gnn_outputs = []
            for timestep in range(seq_len):
                encoded = F.elu(self.conv1(x_seq[:, timestep].reshape(batch_size * num_nodes, -1), edge_index))
                encoded = F.elu(self.conv2(encoded, edge_index))
                gnn_outputs.append(encoded.view(batch_size, num_nodes, -1))

            spatial_sequence = torch.stack(gnn_outputs, dim=1).reshape(batch_size * num_nodes, seq_len, -1)
            _, (hidden_state, _) = self.lstm(spatial_sequence)
            prediction = self.linear(hidden_state.squeeze(0))
            return prediction.view(batch_size, num_nodes, 2)

        gnn_outputs = []
        for timestep in range(x_seq.size(0)):
            encoded = F.elu(self.conv1(x_seq[timestep], edge_index))
            encoded = F.elu(self.conv2(encoded, edge_index))
            gnn_outputs.append(encoded)

        spatial_sequence = torch.stack(gnn_outputs, dim=0)
        lstm_input = spatial_sequence.permute(1, 0, 2)
        _, (hidden_state, _) = self.lstm(lstm_input)
        return self.linear(hidden_state.squeeze(0))


def build_model(config: ModelConfig, num_nodes: int) -> nn.Module:
    """Instantiate the requested GAT-LSTM variant."""

    if config.model_type == "deep":
        return DeepStockGATLSTM(
            num_nodes=num_nodes,
            in_features=config.in_features,
            gnn_hidden=config.gnn_hidden,
            lstm_hidden=config.lstm_hidden,
            heads=config.heads,
            dropout=max(0.1, min(config.dropout, 0.6)),
            add_self_loops=config.gat_add_self_loops,
        )
    return StockGATLSTM(
        num_nodes=num_nodes,
        in_features=config.in_features,
        gnn_hidden=config.gnn_hidden,
        lstm_hidden=config.lstm_hidden,
        heads=config.heads,
        dropout=config.dropout,
        add_self_loops=config.gat_add_self_loops,
    )
