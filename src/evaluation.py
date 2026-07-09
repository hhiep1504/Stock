"""Evaluation, reporting, and visualisation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.structures import SequenceSample
from src.utils import get_cached_edge_index, iter_sequence_minibatches, stack_sequence_batch


class Evaluator:
    """Evaluate trained models on held-out sequences."""

    def __init__(self, model, device: torch.device | None = None):
        self.model = model
        self.device = device or torch.device("cpu")

    def evaluate(self, test_sequences: list[SequenceSample], batch_size: int = 32) -> dict[str, Any]:
        if not test_sequences:
            raise ValueError("Test sequences are empty.")

        self.model.eval()
        predictions = []
        targets = []
        mae_min_sum = 0.0
        mae_max_sum = 0.0
        mse_min_sum = 0.0
        mse_max_sum = 0.0
        edge_index_cache: dict[tuple[tuple[int, int, int], str], torch.Tensor] = {}
        batched_edge_index_cache: dict[tuple[tuple[int, int, int], int, int, str], torch.Tensor] = {}

        with torch.no_grad():
            for batch_samples in iter_sequence_minibatches(test_sequences, batch_size):
                batched_inputs = stack_sequence_batch(
                    batch_samples,
                    device=self.device,
                    edge_index_cache=edge_index_cache,
                    batched_edge_index_cache=batched_edge_index_cache,
                )
                if batched_inputs is not None:
                    x_batch, y_batch, edge_index = batched_inputs
                    output = self.model(x_batch, edge_index)
                    error_min = output[:, :, 0] - y_batch[:, :, 0]
                    error_max = output[:, :, 1] - y_batch[:, :, 1]

                    mae_min_sum += error_min.abs().mean(dim=1).sum().item()
                    mae_max_sum += error_max.abs().mean(dim=1).sum().item()
                    mse_min_sum += error_min.pow(2).mean(dim=1).sum().item()
                    mse_max_sum += error_max.pow(2).mean(dim=1).sum().item()

                    predictions.extend(list(output.cpu().numpy()))
                    targets.extend(list(y_batch.cpu().numpy()))
                    continue

                for sample in batch_samples:
                    x_seq = sample.x_seq.to(self.device)
                    y_target = sample.y_target.to(self.device)
                    if sample.static_edge_index is not None and sample.dynamic_edge_index is not None:
                        edge_index = (
                            get_cached_edge_index(sample.static_edge_index, self.device, edge_index_cache),
                            get_cached_edge_index(sample.dynamic_edge_index, self.device, edge_index_cache),
                        )
                    else:
                        edge_index = get_cached_edge_index(sample.edge_index, self.device, edge_index_cache)

                    output = self.model(x_seq, edge_index)
                    error_min = output[:, 0] - y_target[:, 0]
                    error_max = output[:, 1] - y_target[:, 1]

                    mae_min_sum += error_min.abs().mean().item()
                    mae_max_sum += error_max.abs().mean().item()
                    mse_min_sum += error_min.pow(2).mean().item()
                    mse_max_sum += error_max.pow(2).mean().item()

                    predictions.append(output.cpu().numpy())
                    targets.append(y_target.cpu().numpy())

        count = len(test_sequences)
        return {
            "mae_interval": 0.5 * ((mae_min_sum / count) + (mae_max_sum / count)),
            "mae_min": mae_min_sum / count,
            "mae_max": mae_max_sum / count,
            "mse_interval": 0.5 * ((mse_min_sum / count) + (mse_max_sum / count)),
            "mse_min": mse_min_sum / count,
            "mse_max": mse_max_sum / count,
            "predictions": predictions,
            "targets": targets,
        }

    @staticmethod
    def print_metrics(results: dict[str, Any]) -> None:
        print("=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"MAE (interval): {results['mae_interval']:.6f}")
        print(f"MAE_min:        {results['mae_min']:.6f}")
        print(f"MAE_max:        {results['mae_max']:.6f}")
        print(f"MSE (interval): {results['mse_interval']:.6f}")
        print(f"MSE_min:        {results['mse_min']:.6f}")
        print(f"MSE_max:        {results['mse_max']:.6f}")
        print("=" * 70)


class Visualizer:
    """Generate publication-ready plots for diagnostics."""

    def __init__(self, stock_codes: list[str], aggregation_mode: str = "quarterly"):
        self.stock_codes = stock_codes
        self.aggregation_mode = aggregation_mode
        sns.set_theme(style="whitegrid")

    def plot_predictions(
        self,
        preds: list[np.ndarray],
        targets: list[np.ndarray],
        valid_indices_map: list[Any],
        split_idx: int,
        save_dir: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """Plot predicted and observed intervals stock-by-stock."""

        save_root = Path(save_dir) if save_dir else None
        if save_root:
            save_root.mkdir(parents=True, exist_ok=True)

        test_time_labels = valid_indices_map[split_idx:]
        if self.aggregation_mode == "weekly":
            x_labels = [str(label).replace("_", "/") for label in test_time_labels]
        else:
            x_labels = [f"Q{quarter}/{year}" for year, quarter in test_time_labels]

        for offset in range(0, len(self.stock_codes), 3):
            figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
            axes = np.atleast_1d(axes)

            for axis_index, axis in enumerate(axes):
                stock_index = offset + axis_index
                if stock_index >= len(self.stock_codes):
                    axis.axis("off")
                    continue

                code = self.stock_codes[stock_index]
                pred_min = [pred[stock_index][0] for pred in preds]
                pred_max = [pred[stock_index][1] for pred in preds]
                true_min = [target[stock_index][0] for target in targets]
                true_max = [target[stock_index][1] for target in targets]

                axis.plot(true_min, "k--", label="True Min", linewidth=2, marker="s", markersize=5)
                axis.plot(true_max, "k--", label="True Max", linewidth=2, marker="s", markersize=5)
                axis.plot(pred_min, "r-o", label="Pred Min", linewidth=2, markersize=6)
                axis.plot(pred_max, "b-o", label="Pred Max", linewidth=2, markersize=6)
                axis.fill_between(range(len(true_min)), true_min, true_max, color="gray", alpha=0.15)
                axis.set_xticks(range(len(x_labels)))
                axis.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=10)
                axis.set_title(code, fontweight="bold")
                axis.axhline(y=0, color="gray", linewidth=0.8, alpha=0.6)
                if axis_index == 0:
                    axis.set_ylabel("Return")

            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=10)
            figure.suptitle("GAT-LSTM Predictions on the Test Set", fontsize=14, fontweight="bold")
            figure.tight_layout(rect=[0, 0, 1, 0.92])

            if save_root:
                figure.savefig(save_root / f"predictions_{offset}.png", dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(figure)

    def analyze_range_compression(
        self,
        preds: list[np.ndarray],
        targets: list[np.ndarray],
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> dict[str, float]:
        """Diagnose whether predicted intervals are too narrow."""

        pred_ranges = np.array([(pred[:, 1] - pred[:, 0]).mean() for pred in preds], dtype=np.float64)
        true_ranges = np.array([(target[:, 1] - target[:, 0]).mean() for target in targets], dtype=np.float64)
        range_ratio = float((pred_ranges / (true_ranges + 1e-8)).mean())
        range_bias = float((pred_ranges - true_ranges).mean())

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(true_ranges, "k-o", label="True Range", linewidth=2, markersize=4)
        axes[0].plot(pred_ranges, "r-s", label="Predicted Range", linewidth=2, markersize=4)
        axes[0].fill_between(range(len(true_ranges)), pred_ranges, true_ranges, color="red", alpha=0.2)
        axes[0].set_xlabel("Time Period")
        axes[0].set_ylabel("Average Range")
        axes[0].set_title("Range Compression Over Time")
        axes[0].legend()

        axes[1].hist(true_ranges, bins=15, alpha=0.6, label="True Range", color="black")
        axes[1].hist(pred_ranges, bins=15, alpha=0.6, label="Predicted Range", color="red")
        axes[1].set_xlabel("Range Value")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Range Distribution")
        axes[1].legend()

        figure.tight_layout()

        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figure)

        return {
            "true_range_mean": float(true_ranges.mean()),
            "true_range_std": float(true_ranges.std()),
            "pred_range_mean": float(pred_ranges.mean()),
            "pred_range_std": float(pred_ranges.std()),
            "range_ratio": range_ratio,
            "range_bias": range_bias,
        }

    def visualize_graph(
        self,
        x_tensor: torch.Tensor,
        edge_index: torch.Tensor,
        period_label: str,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """Visualise one graph snapshot."""

        node_features = x_tensor.detach().cpu().numpy()[:, :3]
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
        raw_labels = kmeans.fit_predict(StandardScaler().fit_transform(node_features))

        cluster_volatility = []
        for label in range(4):
            cluster_mask = raw_labels == label
            cluster_volatility.append((label, node_features[cluster_mask, 0].mean()))
        sorted_clusters = sorted(cluster_volatility, key=lambda item: item[1])
        new_label_map = {old_label: rank for rank, (old_label, _) in enumerate(sorted_clusters)}
        sorted_labels = [new_label_map[label] for label in raw_labels]
        node_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.stock_codes)))
        graph.add_edges_from(map(tuple, edge_index.t().tolist()))

        positions = nx.spring_layout(graph, k=0.8, seed=123, iterations=50)
        figure, axis = plt.subplots(figsize=(12, 8))
        nx.draw_networkx_edges(graph, positions, ax=axis, width=1.5, alpha=0.6)
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_color=[node_colors[label] for label in sorted_labels],
            node_size=600,
            alpha=0.9,
            edgecolors="white",
            linewidths=2,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={index: code for index, code in enumerate(self.stock_codes)},
            font_size=9,
            font_weight="bold",
            ax=axis,
        )
        axis.set_title(f"Stock Network Graph: {period_label}", fontsize=14, fontweight="bold")
        axis.axis("off")

        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figure)
