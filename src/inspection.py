"""Dataset inspection helpers for pre-training analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.config import ExperimentConfig
from src.evaluation import Visualizer
from src.graph import DEFAULT_BRIDGE_EDGES, DEFAULT_ISOLATED_NODES
from src.pipeline import prepare_dataset, split_sequences


@dataclass(slots=True)
class TSNEVisualizationResult:
    """Artifacts produced by a pre-training t-SNE inspection run."""

    output_dir: Path
    image_path: Path
    csv_path: Path
    num_sequences: int
    perplexity: float


@dataclass(slots=True)
class GraphVisualizationResult:
    """Artifacts produced by a graph snapshot inspection run."""

    output_dir: Path
    image_paths: list[Path]
    summary_csv_path: Path
    summary_json_path: Path
    node_summary_csv_path: Path
    snapshot_indices: list[int]


def _inspection_run_dir(output_root: Path) -> Path:
    run_dir = output_root / f"tsne_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _label_to_string(label: object) -> str:
    if isinstance(label, tuple):
        return "/".join(str(item) for item in label)
    return str(label)


def _safe_filename_label(label: object) -> str:
    return _label_to_string(label).replace("/", "_").replace(" ", "_")


def _flatten_sequences(sequences) -> np.ndarray:
    return np.stack([sample.x_seq.detach().cpu().numpy().reshape(-1) for sample in sequences], axis=0)


def _select_graph_snapshot_indices(num_sequences: int, split_point: int, graph_mode: str) -> list[int]:
    candidate_indices = [0]
    if split_point > 1:
        candidate_indices.append(split_point - 1)
    if split_point < num_sequences:
        candidate_indices.append(split_point)
    if num_sequences > 1:
        candidate_indices.append(num_sequences - 1)

    if graph_mode in {"dynamic", "hybrid"} and num_sequences > 2:
        candidate_indices.append(num_sequences // 2)

    ordered = []
    seen = set()
    for index in candidate_indices:
        bounded_index = min(max(0, index), num_sequences - 1)
        if bounded_index not in seen:
            ordered.append(bounded_index)
            seen.add(bounded_index)
    return ordered


def _graph_snapshot_summary(sample, sequence_index: int, num_nodes: int, split_label: str) -> dict[str, object]:
    edge_index = sample.edge_index
    directed_edges = int(edge_index.size(1))
    undirected_edges = directed_edges // 2
    degree = np.bincount(edge_index[0].detach().cpu().numpy(), minlength=num_nodes)
    density = undirected_edges / max(1, (num_nodes * (num_nodes - 1) / 2))
    return {
        "sequence_index": sequence_index,
        "label": _label_to_string(sample.label),
        "split": split_label,
        "edge_count_directed": directed_edges,
        "edge_count_undirected": undirected_edges,
        "density": float(density),
        "avg_degree": float(degree.mean()),
        "isolated_nodes": int((degree == 0).sum()),
    }


def _node_topology_summary(stock_codes: list[str], edge_index) -> pd.DataFrame:
    graph = nx.Graph()
    graph.add_nodes_from(stock_codes)
    graph.add_edges_from((stock_codes[src], stock_codes[dst]) for src, dst in edge_index.t().tolist())

    degree_by_node = dict(graph.degree())
    betweenness_by_node = nx.betweenness_centrality(graph, normalized=True)
    bridge_set = {frozenset(edge) for edge in DEFAULT_BRIDGE_EDGES}
    bridge_neighbor_counts = {ticker: 0 for ticker in stock_codes}
    for edge in bridge_set:
        source_ticker, target_ticker = tuple(edge)
        if source_ticker in bridge_neighbor_counts:
            bridge_neighbor_counts[source_ticker] += 1
        if target_ticker in bridge_neighbor_counts:
            bridge_neighbor_counts[target_ticker] += 1

    rows = []
    for ticker in stock_codes:
        rows.append(
            {
                "ticker": ticker,
                "degree": int(degree_by_node.get(ticker, 0)),
                "betweenness_centrality": float(betweenness_by_node.get(ticker, 0.0)),
                "is_isolated_prior": ticker in DEFAULT_ISOLATED_NODES,
                "bridge_neighbors": int(bridge_neighbor_counts.get(ticker, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["betweenness_centrality", "degree", "ticker"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _resolve_perplexity(num_samples: int) -> float:
    if num_samples < 3:
        raise ValueError("Need at least 3 sequences for t-SNE visualisation.")
    if num_samples < 10:
        return max(2.0, (num_samples - 1) / 3.0)
    return min(30.0, max(5.0, (num_samples - 1) / 3.0))


def run_pretraining_tsne(config: ExperimentConfig) -> TSNEVisualizationResult:
    """Generate a pre-training t-SNE view of sequence-level inputs."""

    config.ensure_output_directories()
    prepared = prepare_dataset(config)
    train_sequences, test_sequences = split_sequences(prepared.sequences, config.data.split_idx)

    x_matrix = _flatten_sequences(prepared.sequences)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_matrix)

    if x_scaled.shape[1] > 50 and x_scaled.shape[0] > 3:
        pca_components = min(50, x_scaled.shape[0] - 1, x_scaled.shape[1])
        x_projected = PCA(n_components=pca_components, random_state=config.runtime.random_seed).fit_transform(x_scaled)
    else:
        x_projected = x_scaled

    perplexity = _resolve_perplexity(x_projected.shape[0])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=config.runtime.random_seed,
    )
    coordinates = tsne.fit_transform(x_projected)

    split_labels = ["train"] * len(train_sequences) + ["test"] * len(test_sequences)
    sequence_labels = [_label_to_string(sample.label) for sample in prepared.sequences]
    edge_counts = [int(sample.edge_index.size(1)) for sample in prepared.sequences]
    dataframe = pd.DataFrame(
        {
            "sequence_index": np.arange(len(prepared.sequences)),
            "label": sequence_labels,
            "split": split_labels,
            "edge_count_directed": edge_counts,
            "tsne_1": coordinates[:, 0],
            "tsne_2": coordinates[:, 1],
        }
    )

    run_dir = _inspection_run_dir(config.paths.outputs_dir / "inspections")
    csv_path = run_dir / "pretrain_tsne.csv"
    image_path = run_dir / "pretrain_tsne.png"
    dataframe.to_csv(csv_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    train_mask = dataframe["split"] == "train"
    test_mask = ~train_mask
    axes[0].scatter(
        dataframe.loc[train_mask, "tsne_1"],
        dataframe.loc[train_mask, "tsne_2"],
        s=35,
        alpha=0.75,
        c="#2563eb",
        label="train",
    )
    axes[0].scatter(
        dataframe.loc[test_mask, "tsne_1"],
        dataframe.loc[test_mask, "tsne_2"],
        s=60,
        alpha=0.9,
        c="#dc2626",
        marker="x",
        label="test",
    )
    axes[0].set_title("t-SNE by Split")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend()

    chronology_scatter = axes[1].scatter(
        dataframe["tsne_1"],
        dataframe["tsne_2"],
        c=dataframe["sequence_index"],
        cmap="viridis",
        s=40,
        alpha=0.85,
    )
    axes[1].scatter(
        dataframe.loc[test_mask, "tsne_1"],
        dataframe.loc[test_mask, "tsne_2"],
        s=70,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        label="test highlight",
    )
    axes[1].set_title("t-SNE by Chronology")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(loc="best")
    colorbar = figure.colorbar(chronology_scatter, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Sequence index")

    figure.suptitle(
        (
            f"Pre-training t-SNE | aggregation={config.data.aggregation_mode} | "
            f"window={config.data.window_size} | graph_mode={config.graph.graph_mode}"
        ),
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return TSNEVisualizationResult(
        output_dir=run_dir,
        image_path=image_path,
        csv_path=csv_path,
        num_sequences=len(prepared.sequences),
        perplexity=perplexity,
    )


def run_graph_visualization(config: ExperimentConfig) -> GraphVisualizationResult:
    """Generate representative graph snapshot plots before training."""

    config.ensure_output_directories()
    prepared = prepare_dataset(config)
    train_sequences, test_sequences = split_sequences(prepared.sequences, config.data.split_idx)
    split_point = len(train_sequences)
    total_sequences = len(prepared.sequences)
    snapshot_indices = _select_graph_snapshot_indices(total_sequences, split_point, config.graph.graph_mode)

    run_dir = _inspection_run_dir(config.paths.outputs_dir / "inspections" / "graphs")
    visualizer = Visualizer(prepared.stock_codes, aggregation_mode=config.data.aggregation_mode)

    image_paths: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    node_summary_frame = _node_topology_summary(prepared.stock_codes, prepared.sequences[0].edge_index)

    for sequence_index in snapshot_indices:
        sample = prepared.sequences[sequence_index]
        split_label = "train" if sequence_index < split_point else "test"
        period_label = _label_to_string(sample.label)
        filename_label = _safe_filename_label(sample.label)
        image_path = run_dir / f"graph_{sequence_index:03d}_{filename_label}.png"
        visualizer.visualize_graph(
            x_tensor=sample.x_seq[-1],
            edge_index=sample.edge_index,
            period_label=period_label,
            save_path=image_path,
            show=False,
        )
        image_paths.append(image_path)
        summary_rows.append(
            _graph_snapshot_summary(
                sample=sample,
                sequence_index=sequence_index,
                num_nodes=len(prepared.stock_codes),
                split_label=split_label,
            )
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_csv_path = run_dir / "graph_snapshot_summary.csv"
    summary_json_path = run_dir / "graph_snapshot_summary.json"
    node_summary_csv_path = run_dir / "node_topology_summary.csv"
    summary_frame.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(summary_frame.to_json(orient="records", indent=2), encoding="utf-8")
    node_summary_frame.to_csv(node_summary_csv_path, index=False)

    metadata = {
        "graph_mode": config.graph.graph_mode,
        "num_nodes": len(prepared.stock_codes),
        "num_sequences": total_sequences,
        "snapshot_indices": snapshot_indices,
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "bridge_edges": DEFAULT_BRIDGE_EDGES,
        "isolated_nodes": sorted(DEFAULT_ISOLATED_NODES),
    }
    (run_dir / "graph_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return GraphVisualizationResult(
        output_dir=run_dir,
        image_paths=image_paths,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        node_summary_csv_path=node_summary_csv_path,
        snapshot_indices=snapshot_indices,
    )
