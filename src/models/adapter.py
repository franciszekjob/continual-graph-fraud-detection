import numpy as np
import pandas as pd
from pygod.detector import OCGNN

from src.graph.builder import TransactionGraphBuilder


class GraphAnomalyDetector:
    """
    Adapter bridging PyGOD graph-based detector with a sklearn-like interface
    expected by pyCLAD. Accepts DataFrames, builds graphs internally.
    """

    def __init__(self, model_cls=OCGNN, model_kwargs=None):
        kw = {"epoch": 30, "verbose": False}
        if model_kwargs:
            kw.update(model_kwargs)
        self.model = model_cls(**kw)
        self.graph_builder = TransactionGraphBuilder()

    def fit(self, df: pd.DataFrame) -> "GraphAnomalyDetector":
        graph = self.graph_builder.build(df)
        self.model.fit(graph)
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        key = id(df)
        if not hasattr(self, "_graph_cache") or self._graph_cache_key != key:
            self._graph_cache = self.graph_builder.build(df)
            self._graph_cache_key = key
        scores = self.model.decision_function(self._graph_cache)
        return np.array(scores)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.score_samples(df)
        threshold = np.percentile(scores, 95)
        return (scores >= threshold).astype(int)
