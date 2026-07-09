"""Shared runtime utilities for the refactored GAT-LSTM stack."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.structures import SequenceSample


def set_seed(seed: int = 42) -> None:
    """Lock the main RNGs for repeatable runs on a single machine."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except AttributeError:
        pass

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def resolve_device(preferred: str = "auto") -> torch.device:
    """Resolve the execution device from a simple policy string."""

    preferred = preferred.lower().strip()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if preferred != "auto":
        raise ValueError(f"Unsupported device policy: {preferred}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str | None:
    """Hash a file for reproducibility manifests."""

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.new(algorithm)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_metadata(project_root: str | Path | None = None) -> dict[str, str | None]:
    """Return git metadata when the workspace is inside a repository."""

    root = Path(project_root) if project_root is not None else Path.cwd()

    def _git(command: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *command],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            return completed.stdout.strip() or None
        except Exception:
            return None

    return {
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["rev-parse", "HEAD"]),
        "is_dirty": _git(["status", "--porcelain"]) is not None,
    }


def get_environment_snapshot() -> dict[str, Any]:
    """Collect a compact environment fingerprint for experiment artifacts."""

    cuda_name = None
    if torch.cuda.is_available():
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = None

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": cuda_name,
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }


def iter_sequence_minibatches(
    sequences: list[SequenceSample],
    batch_size: int,
) -> Iterator[list[SequenceSample]]:
    """Yield simple contiguous minibatches from a list of sequence samples."""

    effective_batch_size = max(1, batch_size)
    for start_index in range(0, len(sequences), effective_batch_size):
        yield sequences[start_index : start_index + effective_batch_size]


def _edge_index_identity(edge_index: torch.Tensor) -> tuple[int, int, int]:
    return (id(edge_index), int(edge_index.data_ptr()), int(edge_index.numel()))


def get_cached_edge_index(
    edge_index: torch.Tensor,
    device: torch.device,
    cache: dict[tuple[tuple[int, int, int], str], torch.Tensor],
) -> torch.Tensor:
    """Move an edge index to the target device once and reuse it."""

    cache_key = (_edge_index_identity(edge_index), str(device))
    cached = cache.get(cache_key)
    if cached is None:
        cached = edge_index.long().to(device, non_blocking=True)
        cache[cache_key] = cached
    return cached


def samples_share_edge_index(batch_samples: list[SequenceSample]) -> bool:
    """Return True when every sample in the batch uses the same graph."""

    if len(batch_samples) < 2:
        return False

    first_edge_index = batch_samples[0].edge_index
    for sample in batch_samples[1:]:
        if sample.edge_index is first_edge_index:
            continue
        if sample.edge_index.shape != first_edge_index.shape:
            return False
        if not torch.equal(sample.edge_index, first_edge_index):
            return False
    return True


def samples_have_dual_graph(batch_samples: list[SequenceSample]) -> bool:
    """Return True when every sample carries separate static and dynamic graphs."""

    return bool(batch_samples) and all(
        sample.static_edge_index is not None and sample.dynamic_edge_index is not None
        for sample in batch_samples
    )


def get_batched_edge_index(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
    edge_index_cache: dict[tuple[tuple[int, int, int], str], torch.Tensor],
    batched_edge_index_cache: dict[tuple[tuple[int, int, int], int, int, str], torch.Tensor],
) -> torch.Tensor:
    """Create or reuse a block-diagonal batched edge index for identical graphs."""

    base_edge_index = get_cached_edge_index(edge_index, device, edge_index_cache)
    if batch_size == 1:
        return base_edge_index

    cache_key = (_edge_index_identity(edge_index), int(batch_size), int(num_nodes), str(device))
    cached = batched_edge_index_cache.get(cache_key)
    if cached is not None:
        return cached

    offsets = torch.arange(batch_size, device=device, dtype=base_edge_index.dtype) * num_nodes
    expanded = base_edge_index.unsqueeze(0).expand(batch_size, -1, -1) + offsets[:, None, None]
    batched_edge_index = expanded.permute(1, 0, 2).reshape(2, -1).contiguous()
    batched_edge_index_cache[cache_key] = batched_edge_index
    return batched_edge_index


def batch_edge_indices(
    edge_indices: list[torch.Tensor],
    num_nodes: int,
    device: torch.device,
    edge_index_cache: dict[tuple[tuple[int, int, int], str], torch.Tensor],
    batched_edge_index_cache: dict[tuple[tuple[int, int, int], int, int, str], torch.Tensor],
) -> torch.Tensor:
    """Batch either identical or distinct edge sets into a single PyG-compatible edge index."""

    if not edge_indices:
        raise ValueError("edge_indices must not be empty.")

    first_edge_index = edge_indices[0]
    all_equal = True
    for edge_index in edge_indices[1:]:
        if edge_index is first_edge_index:
            continue
        if edge_index.shape != first_edge_index.shape or not torch.equal(edge_index, first_edge_index):
            all_equal = False
            break

    if all_equal:
        return get_batched_edge_index(
            first_edge_index,
            batch_size=len(edge_indices),
            num_nodes=num_nodes,
            device=device,
            edge_index_cache=edge_index_cache,
            batched_edge_index_cache=batched_edge_index_cache,
        )

    pyg_batch = Batch.from_data_list(
        [Data(edge_index=edge_index.long(), num_nodes=num_nodes) for edge_index in edge_indices]
    )
    return pyg_batch.edge_index.to(device, non_blocking=True)


def stack_sequence_batch(
    batch_samples: list[SequenceSample],
    device: torch.device,
    edge_index_cache: dict[tuple[tuple[int, int, int], str], torch.Tensor],
    batched_edge_index_cache: dict[tuple[tuple[int, int, int], int, int, str], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None:
    """Stack samples into a true tensor batch, using PyG batching when graphs differ."""

    if not batch_samples:
        return None

    x_batch = torch.stack([sample.x_seq for sample in batch_samples], dim=0).to(device, non_blocking=True)
    y_batch = torch.stack([sample.y_target for sample in batch_samples], dim=0).to(device, non_blocking=True)
    num_nodes = int(batch_samples[0].y_target.shape[0])

    if samples_have_dual_graph(batch_samples):
        static_edge_index = batch_edge_indices(
            [sample.static_edge_index for sample in batch_samples if sample.static_edge_index is not None],
            num_nodes=num_nodes,
            device=device,
            edge_index_cache=edge_index_cache,
            batched_edge_index_cache=batched_edge_index_cache,
        )
        dynamic_edge_index = batch_edge_indices(
            [sample.dynamic_edge_index for sample in batch_samples if sample.dynamic_edge_index is not None],
            num_nodes=num_nodes,
            device=device,
            edge_index_cache=edge_index_cache,
            batched_edge_index_cache=batched_edge_index_cache,
        )
        return x_batch, y_batch, (static_edge_index, dynamic_edge_index)

    edge_index = batch_edge_indices(
        [sample.edge_index for sample in batch_samples],
        num_nodes=num_nodes,
        device=device,
        edge_index_cache=edge_index_cache,
        batched_edge_index_cache=batched_edge_index_cache,
    )
    return x_batch, y_batch, edge_index
