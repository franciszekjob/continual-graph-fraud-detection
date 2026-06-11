"""PyGOD -> pyCLAD model adapter.

pyCLAD strategies call ``fit(X)`` / ``predict(X)`` with plain numpy arrays,
while PyGOD detectors consume ``torch_geometric.data.Data`` objects. This
adapter bridges the two: ``X`` carries global node indices, which are resolved
to an induced subgraph of the single global transaction graph. Training and
scoring then run through the PyGOD detector, which — with ``batch_size > 0``
and ``num_neigh`` — samples mini-batch neighborhoods on the fly with PyG's
``NeighborLoader``, keeping GPU memory bounded regardless of subgraph size.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Type

import numpy as np
import torch
from pyclad.models.model import Model
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class PyGODAdapter(Model):
    """Wrap any PyGOD detector class as a pyCLAD :class:`Model`.

    Args:
        graph: the global transaction graph (``x``, ``edge_index``, mapping
            attributes). Kept on CPU; PyGOD moves sampled batches to the GPU.
        detector_cls: a PyGOD detector class, e.g. ``pygod.detector.DOMINANT``.
        detector_params: hyperparameters forwarded to ``detector_cls``.
        model_name: display name for pyCLAD reports (defaults to class name).

    Note on continual semantics: like the PyOD adapters shipped with pyCLAD,
    each ``fit`` call (re)trains the detector from scratch on the data the
    *strategy* hands over — Naive passes only the current concept, Replay adds
    its buffer, Cumulative passes everything seen so far. The strategies thus
    differ in data curation, not in weight transfer.
    """

    def __init__(
        self,
        graph: Data,
        detector_cls: Type,
        detector_params: dict[str, Any] | None = None,
        model_name: str | None = None,
    ):
        self._graph = graph
        self._detector_cls = detector_cls
        self._params = self._filter_params(detector_cls, dict(detector_params or {}))
        self._name = model_name or detector_cls.__name__
        self._detector = None

    @staticmethod
    def _filter_params(detector_cls: Type, params: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(detector_cls.__init__)
        if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            return params
        accepted = {k: v for k, v in params.items() if k in signature.parameters}
        dropped = sorted(set(params) - set(accepted))
        if dropped:
            logger.warning(
                "%s does not accept %s — dropping", detector_cls.__name__, dropped
            )
        return accepted

    def _subgraph(self, data: np.ndarray) -> Data:
        """Induce the subgraph spanned by the requested global node indices.

        ``Data.subgraph`` preserves the order of the index tensor, so row i of
        every output below corresponds to row i of the incoming ``data``.

        Isolated nodes get a self-loop: reconstruction-based PyGOD detectors
        build their adjacency target with ``to_dense_adj(edge_index)`` (no
        ``max_num_nodes``), so a trailing isolated node would otherwise shrink
        the matrix and crash batch indexing.
        """
        node_idx = torch.as_tensor(
            np.asarray(data).reshape(-1), dtype=torch.long
        )
        subgraph = self._graph.subgraph(node_idx)
        connected = torch.zeros(subgraph.num_nodes, dtype=torch.bool)
        connected[subgraph.edge_index.reshape(-1)] = True
        isolated = torch.nonzero(~connected).reshape(-1)
        if isolated.numel():
            subgraph.edge_index = torch.cat(
                [subgraph.edge_index, isolated.expand(2, -1)], dim=1
            )
        return subgraph

    def fit(self, data: np.ndarray) -> None:
        subgraph = self._subgraph(data)
        logger.info(
            "[%s] fit on %d nodes / %d edges",
            self._name,
            subgraph.num_nodes,
            subgraph.num_edges,
        )
        self._detector = self._detector_cls(**self._params)
        self._detector.fit(subgraph)

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y_pred, anomaly_scores)`` aligned row-by-row with ``data``,
        as required by the pyCLAD ``Model`` contract."""
        if self._detector is None:
            raise RuntimeError("predict() called before fit().")
        subgraph = self._subgraph(data)
        pred, score = self._detector.predict(subgraph, return_pred=True, return_score=True)
        y_pred = np.asarray(pred.cpu() if torch.is_tensor(pred) else pred, dtype=np.int64)
        scores = np.nan_to_num(
            np.asarray(score.cpu() if torch.is_tensor(score) else score, dtype=np.float64)
        )
        return y_pred, scores

    def name(self) -> str:
        return f"PyGOD-{self._name}"

    def additional_info(self) -> dict[str, Any]:
        return {"detector_params": {k: str(v) for k, v in self._params.items()}}
