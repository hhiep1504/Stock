import json
import math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import torch

from src.config import ExperimentConfig
from src.data import DataLoader, FeatureEngineer
from src.graph import GraphConstructor

cfg = ExperimentConfig.from_json(Path('configs/default_experiment.json'))
cfg.resolve_paths()

data_loader = DataLoader(cfg.data.daily_file, cfg.data.target_file)
daily_frame = data_loader.load_daily_data()
stock_codes = data_loader.get_stock_codes()
feature_engineer = FeatureEngineer(daily_frame, stock_codes, aggregation_mode=cfg.data.aggregation_mode)
feat_std, feat_mean, feat_return, feat_skew = feature_engineer.compute_features()
target_min, target_max = feature_engineer.compute_targets()
x_full, y_full, valid_indices_map = feature_engineer.create_tensors(
    feat_std, feat_mean, feat_return, feat_skew, target_min, target_max
)

graph_constructor = GraphConstructor(stock_codes)
graph_constructor.sector_map = graph_constructor.get_sector_mapping()
static_edge_index = graph_constructor.create_static_graph()

N = len(stock_codes)
MAX_EDGES = N * (N - 1) / 2

# static undirected edge set
static_pairs = set()
if static_edge_index.numel() > 0:
    ei = static_edge_index.cpu().numpy()
    for s, t in zip(ei[0], ei[1]):
        if s == t:
            continue
        static_pairs.add((int(min(s, t)), int(max(s, t))))
STATIC_EDGES = len(static_pairs)
STATIC_DENSITY = STATIC_EDGES / MAX_EDGES


def subset_prices_for_window(start_index: int, window_size: int):
    if cfg.data.aggregation_mode == 'weekly':
        start_week = valid_indices_map[start_index]
        end_week = valid_indices_map[start_index + window_size - 1]
        return data_loader.get_week_range(start_week, end_week)
    start_year, start_quarter = valid_indices_map[start_index]
    end_year, end_quarter = valid_indices_map[start_index + window_size - 1]
    return data_loader.get_date_range(start_year, start_quarter, end_year, end_quarter)


def similarity_matrix(daily_subset, metric):
    if metric == 'pearson':
        return daily_subset.corr().abs().fillna(0.0).to_numpy(dtype=np.float64)
    matrix = daily_subset.T.to_numpy(dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normalised = matrix / norms
    sim = normalised @ normalised.T
    return np.abs(sim)


def undirected_pairs(edge_index: torch.Tensor):
    pairs = set()
    if edge_index.numel() == 0:
        return pairs
    ei = edge_index.cpu().numpy()
    for s, t in zip(ei[0], ei[1]):
        if s == t:
            continue
        pairs.add((int(min(s, t)), int(max(s, t))))
    return pairs


def metrics_for(window_size, top_k, threshold, metric):
    dyn_edge_counts = []
    avg_corrs = []
    jaccs = []
    cross_static = []
    hybrid_dens = []
    prev_pairs = None

    for start_index in range(len(x_full) - window_size):
        subset = subset_prices_for_window(start_index, window_size)
        sim = similarity_matrix(subset, metric)
        sim = np.array(sim, dtype=np.float64, copy=True)
        np.fill_diagonal(sim, 0.0)

        edge_index = graph_constructor.create_dynamic_graph(
            subset,
            top_k=top_k,
            use_arm=False,
            corr_threshold=threshold,
            similarity_metric=metric,
        )
        pairs = undirected_pairs(edge_index)
        dyn_edge_counts.append(len(pairs))

        if pairs:
            corr_vals = [float(sim[i, j]) for i, j in pairs]
            avg_corrs.append(float(np.mean(corr_vals)))
            cross_static.append(float(len([p for p in pairs if p not in static_pairs]) / len(pairs)))
        else:
            avg_corrs.append(0.0)
            cross_static.append(0.0)

        union_pairs = pairs | static_pairs
        hybrid_dens.append(len(union_pairs) / MAX_EDGES)

        if prev_pairs is not None:
            union = len(prev_pairs | pairs)
            inter = len(prev_pairs & pairs)
            jaccs.append(0.0 if union == 0 else inter / union)
        prev_pairs = pairs

    return {
        'AvgEdges': round(mean(dyn_edge_counts), 3),
        'AvgCorr': round(mean(avg_corrs), 3),
        'StabJacc': round(mean(jaccs), 3),
        'CrossStaticPct': round(100 * mean(cross_static), 1),
        'HybridDens': round(mean(hybrid_dens), 3),
        'StdEdge': round(pstdev(dyn_edge_counts), 3),
    }

comparisons = [
    {'window': 8, 'top_k': 3, 'threshold': 0.7, 'metric': 'pearson'},
    {'window': 8, 'top_k': 4, 'threshold': 0.7, 'metric': 'pearson'},
    {'window': 8, 'top_k': 5, 'threshold': 0.7, 'metric': 'pearson'},
    {'window': 8, 'top_k': 4, 'threshold': 0.6, 'metric': 'pearson'},
    {'window': 8, 'top_k': 4, 'threshold': 0.8, 'metric': 'pearson'},
    {'window': 8, 'top_k': 4, 'threshold': 0.95, 'metric': 'cosine'},
]

rows = []
for c in comparisons:
    metrics = metrics_for(c['window'], c['top_k'], c['threshold'], c['metric'])
    row = {
        'Config': f"w{c['window']}_{c['metric']}_k{c['top_k']}_t{c['threshold']}",
        **metrics,
    }
    rows.append(row)

chosen_by_window = []
for w in [4, 8, 12]:
    metrics = metrics_for(w, 4, 0.7, 'pearson')
    chosen_by_window.append({
        'Config': f"w{w}_pearson_k4_t0.7",
        **metrics,
    })

payload = {
    'static_baseline': {
        'StaticEdges': STATIC_EDGES,
        'StaticDensity': round(STATIC_DENSITY, 3),
    },
    'comparison_window_8': rows,
    'chosen_across_windows': chosen_by_window,
}
print(json.dumps(payload, indent=2))