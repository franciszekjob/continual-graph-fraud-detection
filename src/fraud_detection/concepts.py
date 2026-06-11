"""Temporal concept splitting: TransactionDT -> 6 monthly pyCLAD concepts.

Concept payloads are *global node indices* into the single transaction graph
(shape ``(n, 1)`` int64 arrays), not feature matrices. pyCLAD strategies only
ever concatenate and forward these arrays, so replay buffers work unchanged,
and the PyGOD adapter resolves them back to subgraphs of the global graph.
"""

from __future__ import annotations

import logging

import numpy as np
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset

from fraud_detection.config import ConceptConfig

logger = logging.getLogger(__name__)


def _as_payload(node_indices: np.ndarray) -> np.ndarray:
    return node_indices.astype(np.int64).reshape(-1, 1)


def build_concepts_dataset(
    labels: np.ndarray, transaction_dt: np.ndarray, cfg: ConceptConfig
) -> ConceptsDataset:
    """Bucket transactions into ``n_concepts`` sequential equal-width time
    intervals and build train/test concepts ordered by time.

    Inside each concept the split is also temporal (first ``train_ratio`` of
    the interval trains, the rest tests) to avoid look-ahead leakage. With
    ``normal_only_training`` the train side keeps only legitimate transactions
    (one-class regime); test sides keep everything plus labels.
    """
    dt = np.asarray(transaction_dt)
    edges = np.linspace(dt.min(), dt.max(), cfg.n_concepts + 1)
    concept_of = np.clip(np.searchsorted(edges, dt, side="right") - 1, 0, cfg.n_concepts - 1)

    train_concepts, test_concepts = [], []
    for i in range(cfg.n_concepts):
        # Rows are pre-sorted by TransactionDT, so indices are in time order.
        idx = np.flatnonzero(concept_of == i)
        if len(idx) == 0:
            raise ValueError(f"Concept {i} is empty — reduce n_concepts or check data.")
        split = int(len(idx) * cfg.train_ratio)
        train_idx, test_idx = idx[:split], idx[split:]
        if cfg.normal_only_training:
            train_idx = train_idx[labels[train_idx] == 0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Concept {i} has an empty train or test split.")

        name = f"month_{i + 1}"
        train_concepts.append(Concept(name, data=_as_payload(train_idx)))
        test_concepts.append(
            Concept(name, data=_as_payload(test_idx), labels=labels[test_idx])
        )
        logger.info(
            "%s: %d train / %d test nodes (test fraud rate %.2f%%)",
            name,
            len(train_idx),
            len(test_idx),
            100 * labels[test_idx].mean(),
        )

    return ConceptsDataset(
        name="ieee-cis-fraud-temporal",
        train_concepts=train_concepts,
        test_concepts=test_concepts,
    )
