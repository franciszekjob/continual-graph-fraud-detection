"""Global homogeneous transaction-graph assembly.

Nodes are individual transactions; an undirected edge connects two
transactions that share an identical composite entity key (e.g. the same
card1..card6 combination). Within each entity group transactions are linked
to their ``max_neighbors_per_key`` most recent predecessors instead of all
pairs, which bounds the edge count to O(N * k) and keeps the single global
graph comfortably inside RAM.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected

from fraud_detection.config import GraphConfig
from fraud_detection.preprocessing import PreparedData

logger = logging.getLogger(__name__)


def _composite_key_codes(key_frame: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    """Factorize a tuple of columns into integer group codes (-1 = missing)."""
    sub = key_frame[list(cols)]
    codes, _ = pd.MultiIndex.from_frame(sub).factorize()
    codes = np.asarray(codes, dtype=np.int64)
    codes[~sub.notna().all(axis=1).to_numpy()] = -1
    return codes


def _temporal_edges_within_groups(
    codes: np.ndarray, dt: np.ndarray, max_neighbors: int
) -> np.ndarray:
    """Link each node to up to ``max_neighbors`` earlier nodes of its group.

    Sorting by (group, time) places each group contiguously in temporal order,
    so "the k-th previous transaction of the same entity" is just a shift by k
    in the sorted array — the whole construction is vectorized.
    """
    order = np.lexsort((dt, codes))
    sorted_codes = codes[order]
    src, dst = [], []
    for k in range(1, max_neighbors + 1):
        same_group = (sorted_codes[k:] == sorted_codes[:-k]) & (sorted_codes[k:] >= 0)
        src.append(order[k:][same_group])
        dst.append(order[:-k][same_group])
    return np.stack([np.concatenate(src), np.concatenate(dst)])


def build_transaction_graph(prepared: PreparedData, cfg: GraphConfig) -> Data:
    """Assemble the single global ``torch_geometric.data.Data`` object.

    ``x``: node features, ``edge_index``: undirected structural connections,
    ``y``: isFraud labels, ``transaction_id``/``transaction_dt``: per-node
    mapping back to the original DataFrame rows (used for concept indexing;
    PyG subsets them automatically together with ``x`` on ``Data.subgraph``).
    """
    num_nodes = prepared.num_nodes
    edge_blocks = []
    for cols in cfg.edge_key_groups:
        missing = [c for c in cols if c not in prepared.key_frame.columns]
        if missing:
            logger.warning("Skipping edge key %s (missing columns %s)", cols, missing)
            continue
        codes = _composite_key_codes(prepared.key_frame, cols)
        edges = _temporal_edges_within_groups(
            codes, prepared.transaction_dt, cfg.max_neighbors_per_key
        )
        logger.info("Edge key %s -> %d directed edges", "+".join(cols), edges.shape[1])
        edge_blocks.append(edges)

    if not edge_blocks:
        raise ValueError("No edge key group produced edges — check edge_key_groups.")

    edge_index = torch.from_numpy(np.concatenate(edge_blocks, axis=1)).long()
    edge_index = to_undirected(coalesce(edge_index, num_nodes=num_nodes), num_nodes=num_nodes)

    data = Data(
        x=torch.from_numpy(prepared.features),
        edge_index=edge_index,
        y=torch.from_numpy(prepared.labels),
        num_nodes=num_nodes,
    )
    data.transaction_id = torch.from_numpy(prepared.transaction_ids)
    data.transaction_dt = torch.from_numpy(prepared.transaction_dt)

    logger.info(
        "Global graph: %d nodes, %d undirected edges, %d isolated nodes",
        num_nodes,
        edge_index.shape[1] // 2,
        num_nodes - torch.unique(edge_index).numel(),
    )
    return data
