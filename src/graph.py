"""Graph construction utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.utils import to_undirected


DEFAULT_SECTOR_GROUPS = {
    # Knowledge-based graph prior:
    # any two stocks in the same group receive an undirected edge.
    "Residential": [
        "VHM",
        "NVL",
        "PDR",
        "NLG",
        "KDH",
        "DIG",
        "HDG",
        "NTL",
        "HDC",
        "HPX",
        "NBB",
        "ITC",
        "CCL",
        "TCH",
        "TDC",
    ],
    "Industrial": ["KBC", "SZL", "LHG", "TIP", "D2D", "IJC"],
    "Services": ["CRE", "DXG"],
}

DEFAULT_ISOLATED_NODES = {"VPI", "TIX", "CDC"}
NON_CONNECTING_GROUPS = {"Unknown", "Isolated"}
DEFAULT_BRIDGE_EDGES = [
    # Corporate ecosystem bridge in Binh Duong.
    ("IJC", "TDC"),
    # Distribution-to-developer bridge.
    ("CRE", "VHM"),
    # Northern industrial-to-residential bridge via Hai Phong.
    ("KBC", "TCH"),
    # Dong Nai industrial-to-residential bridge.
    ("D2D", "NLG"),
    # Hybrid / competitor bridge supported by observed correlation.
    ("DXG", "DIG"),
]


def _build_default_sector_map() -> dict[str, str]:
    sector_map: dict[str, str] = {}
    for sector_name, tickers in DEFAULT_SECTOR_GROUPS.items():
        for ticker in tickers:
            sector_map[ticker] = sector_name
    for ticker in DEFAULT_ISOLATED_NODES:
        sector_map[ticker] = "Isolated"
    return sector_map


DEFAULT_SECTOR_MAP = _build_default_sector_map()


class GraphConstructor:
    """Create static, dynamic, and hybrid stock graphs."""

    def __init__(
        self,
        stock_codes: list[str],
        sector_map: Optional[dict[str, str]] = None,
        use_bridge_edges: bool = True,
    ):
        self.stock_codes = stock_codes
        self.sector_map = sector_map or {}
        self.use_bridge_edges = use_bridge_edges
        self.static_edge_index: torch.Tensor | None = None

    def get_sector_mapping(self) -> dict[str, str]:
        """Return the active stock-to-sector mapping."""

        if self.sector_map:
            return {
                ticker: self.sector_map.get(ticker, DEFAULT_SECTOR_MAP.get(ticker, "Unknown"))
                for ticker in self.stock_codes
            }
        return {ticker: DEFAULT_SECTOR_MAP.get(ticker, "Unknown") for ticker in self.stock_codes}

    @staticmethod
    def load_sector_map(path: str | Path) -> dict[str, str]:
        """Load ticker-to-sector mappings from a CSV file."""

        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Sector map file not found: {input_path}")

        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Sector map file has no header: {input_path}")

            field_lookup = {name.lower(): name for name in reader.fieldnames}
            ticker_field = next(
                (field_lookup[name] for name in ("ticker", "symbol", "code") if name in field_lookup),
                None,
            )
            sector_field = next(
                (
                    field_lookup[name]
                    for name in (
                        "industry",
                        "industry_name",
                        "industryname",
                        "icb_name",
                        "icbname",
                        "graph_group",
                        "sector",
                    )
                    if name in field_lookup
                ),
                None,
            )
            if ticker_field is None or sector_field is None:
                raise ValueError(
                    f"Expected ticker/code and sector/industry columns in {input_path}; "
                    f"found {reader.fieldnames}"
                )

            mapping: dict[str, str] = {}
            for row in reader:
                ticker = str(row.get(ticker_field, "")).strip().upper()
                sector = str(row.get(sector_field, "")).strip()
                if ticker and sector:
                    mapping[ticker] = sector
            return mapping

    def create_static_graph(self) -> torch.Tensor:
        """Connect stocks that belong to the same knowledge-based sector group."""

        edges: set[tuple[int, int]] = set()
        ticker_to_index = {ticker: index for index, ticker in enumerate(self.stock_codes)}
        for source_idx, source_ticker in enumerate(self.stock_codes):
            for target_idx in range(source_idx + 1, len(self.stock_codes)):
                target_ticker = self.stock_codes[target_idx]
                source_group = self.sector_map.get(source_ticker, "Unknown")
                target_group = self.sector_map.get(target_ticker, "Unknown")
                if (
                    source_group not in NON_CONNECTING_GROUPS
                    and source_group == target_group
                ):
                    edges.add((source_idx, target_idx))

        # Add legacy hand-crafted bridges only for the built-in 26-stock fallback map.
        if self.use_bridge_edges:
            for source_ticker, target_ticker in DEFAULT_BRIDGE_EDGES:
                if source_ticker not in ticker_to_index or target_ticker not in ticker_to_index:
                    continue
                source_idx = ticker_to_index[source_ticker]
                target_idx = ticker_to_index[target_ticker]
                if source_idx == target_idx:
                    continue
                edge = (min(source_idx, target_idx), max(source_idx, target_idx))
                edges.add(edge)

        if not edges:
            self.static_edge_index = torch.empty((2, 0), dtype=torch.long)
            return self.static_edge_index

        sorted_edges = sorted(edges)
        edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()
        self.static_edge_index = to_undirected(edge_index)
        return self.static_edge_index

    def create_dynamic_graph(
        self,
        daily_subset,
        top_k: int = 5,
        use_arm: bool = False,
        corr_threshold: float = 0.7,
        similarity_metric: str = "cosine",
    ) -> torch.Tensor:
        """Create a period-specific graph from price co-movement."""

        del use_arm
        final_edges: list[list[int]] = []

        if len(daily_subset) > 5:
            if similarity_metric == "cosine":
                similarity_matrix = self._cosine_similarity_matrix(daily_subset)
            else:
                similarity_matrix = daily_subset.corr().abs().fillna(0.0).to_numpy()

            # Some NumPy/Pandas combinations can expose a read-only view here,
            # which breaks in-place diagonal zeroing on Linux environments.
            similarity_matrix = np.array(similarity_matrix, dtype=np.float64, copy=True)
            np.fill_diagonal(similarity_matrix, 0.0)

            for node_idx in range(len(self.stock_codes)):
                candidate_indices = np.argsort(similarity_matrix[node_idx])[-top_k:][::-1]
                for neighbour_idx in candidate_indices:
                    if similarity_matrix[node_idx, neighbour_idx] >= corr_threshold:
                        final_edges.append([node_idx, int(neighbour_idx)])

        if not final_edges:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor(final_edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        return torch.unique(edge_index, dim=1)

    def create_hybrid_graph(self, daily_subset, top_k: int = 5) -> torch.Tensor:
        """Combine the static sector graph and dynamic similarity graph."""

        dynamic_edges = self.create_dynamic_graph(daily_subset, top_k=top_k)

        if self.static_edge_index is None or self.static_edge_index.numel() == 0:
            return dynamic_edges
        if dynamic_edges.numel() == 0:
            return self.static_edge_index

        merged = torch.cat([self.static_edge_index, dynamic_edges], dim=1)
        return torch.unique(merged, dim=1)

    @staticmethod
    def _cosine_similarity_matrix(daily_subset) -> np.ndarray:
        """Compute absolute cosine similarity between stock price series."""

        matrix = daily_subset.T.to_numpy(dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normalised = matrix / norms
        similarity = normalised @ normalised.T
        return np.abs(similarity)
