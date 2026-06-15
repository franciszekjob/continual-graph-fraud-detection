"""Configuration objects for the continual fraud-detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Where the raw Kaggle CSVs live and how much of them to use."""

    data_dir: Path = Path("data")
    transactions_file: str = "train_transaction.csv"
    # The 339 V-columns add ~800MB of float32 features; enable on the cluster.
    use_v_features: bool = False
    # Subsample for smoke tests (None = full dataset).
    nrows: int | None = None


@dataclass
class GraphConfig:
    """How transaction nodes are wired together.

    Each entry in ``edge_key_groups`` is a tuple of columns forming a composite
    entity key. Transactions sharing the exact same key value form a group; a
    new transaction is linked to its ``max_neighbors_per_key`` most recent
    predecessors in the group. This keeps the edge count linear in the number
    of transactions (a naive all-pairs clique within groups is O(n^2) and blows
    up on hub entities).
    """

    edge_key_groups: tuple[tuple[str, ...], ...] = (
        ("card1", "card2", "card3", "card4", "card5", "card6"),
        ("card1", "addr1"),
        ("card1", "P_emaildomain"),
    )
    max_neighbors_per_key: int = 5


@dataclass
class ConceptConfig:
    """Temporal stream definition for pyCLAD."""

    n_concepts: int = 6  # 1 concept ~ 1 month of TransactionDT
    train_ratio: float = 0.7  # temporal split inside each concept
    # One-class regime: train only on legitimate transactions, evaluate on all.
    normal_only_training: bool = True


@dataclass
class ModelConfig:
    """PyGOD detector selection and shared hyperparameters."""

    name: str = "dominant"  # see MODEL_REGISTRY in run_experiment.py
    hid_dim: int = 64
    num_layers: int = 2
    epoch: int = 50
    lr: float = 0.004
    contamination: float = 0.035  # ~ fraud base rate in IEEE-CIS
    # batch_size > 0 makes PyGOD train with torch_geometric NeighborLoader
    # (mini-batch neighborhood sampling) instead of full-batch — required to
    # keep GPU memory bounded on the global graph.
    batch_size: int = 4096
    num_neigh: int = 10
    gpu: int = -1  # -1 = CPU; set 0 on the cluster (auto-detected by the CLI)
    extra: dict[str, Any] = field(default_factory=dict)

    def detector_params(self) -> dict[str, Any]:
        return {
            "hid_dim": self.hid_dim,
            "num_layers": self.num_layers,
            "epoch": self.epoch,
            "lr": self.lr,
            "contamination": self.contamination,
            "batch_size": self.batch_size,
            "num_neigh": self.num_neigh,
            "gpu": self.gpu,
            **self.extra,
        }
