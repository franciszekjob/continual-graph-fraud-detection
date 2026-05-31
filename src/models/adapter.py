import numpy as np
import pandas as pd
from pygod.detector import DOMINANT

from src.graph.builder import TransactionGraphBuilder


class GraphAnomalyDetector:
    """
    Adapter bridging PyGOD graph-based detector with a sklearn-like interface
    expected by pyCLAD. Accepts DataFrames, builds graphs internally.
    """

    def __init__(self, model_cls=DOMINANT, **model_kwargs):
        model_kwargs.setdefault("epoch", 20)
        model_kwargs.setdefault("verbose", False)
        self.model = model_cls(**model_kwargs)
        self.graph_builder = TransactionGraphBuilder()

    def fit(self, df: pd.DataFrame) -> "GraphAnomalyDetector":
        graph = self.graph_builder.build(df)
        self.model.fit(graph)
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        graph = self.graph_builder.build(df)
        scores = self.model.decision_function(graph)
        return np.array(scores)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.score_samples(df)
        threshold = np.percentile(scores, 95)
        return (scores >= threshold).astype(int)
