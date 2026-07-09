"""Shared data structures for the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch


@dataclass(slots=True)
class SequenceSample:
    """One temporal graph sample."""

    x_seq: torch.Tensor
    y_target: torch.Tensor
    edge_index: torch.Tensor
    label: Any
    static_edge_index: torch.Tensor | None = None
    dynamic_edge_index: torch.Tensor | None = None


@dataclass(slots=True)
class PreparedDataset:
    """Container returned by the data-preparation pipeline."""

    daily_frame: pd.DataFrame
    stock_codes: list[str]
    x_full: torch.Tensor
    x_full_raw: torch.Tensor
    y_full: torch.Tensor
    feature_names: list[str]
    valid_indices_map: list[Any]
    static_edge_index: torch.Tensor | None
    sequence_templates: list[SequenceSample] = field(default_factory=list)
    sequences: list[SequenceSample] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentRunResult:
    """Structured result returned by the main training pipeline."""

    experiment_dir: Path | None
    metrics: dict[str, Any]
    train_sequences: int
    test_sequences: int
    stock_codes: list[str]
