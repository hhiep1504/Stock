# from src.gat_lstm.config import ExperimentConfig
# from src.gat_lstm.pipeline import prepare_dataset
# import numpy as np

# cfg = ExperimentConfig.from_json("configs/default_experiment.json")
# prepared = prepare_dataset(cfg)
# n = len(prepared.stock_codes)

# densities, avg_degrees, isolated_counts = [], [], []

# for sample in prepared.sequences:
#     edge_index = sample.edge_index
#     e = edge_index.size(1) // 2
#     density = e / (n * (n - 1) / 2)
#     degree = np.bincount(edge_index[0].cpu().numpy(), minlength=n)
#     densities.append(density)
#     avg_degrees.append(degree.mean())
#     isolated_counts.append(int((degree == 0).sum()))

# print("density mean:", np.mean(densities))
# print("density min/max:", np.min(densities), np.max(densities))
# print("avg_degree mean:", np.mean(avg_degrees))
# print("isolated mean:", np.mean(isolated_counts))




from src.config import ExperimentConfig
from src.pipeline import prepare_dataset
from src.graph import GraphConstructor

cfg = ExperimentConfig.from_json("configs/default_experiment.json")
prepared = prepare_dataset(cfg)
gc = GraphConstructor(prepared.stock_codes)
sector_map = gc.get_sector_mapping()

missing = [code for code in prepared.stock_codes if sector_map.get(code, "Unknown") == "Unknown"]
print("stock_codes:", prepared.stock_codes)
print("missing:", missing)