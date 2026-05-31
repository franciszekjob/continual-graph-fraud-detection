import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.preprocessor import get_node_features


DEFAULT_EDGE_FEATURES = ["card1", "card2", "addr1", "P_emaildomain"]


class TransactionGraphBuilder:
    def __init__(
        self,
        edge_features: list[str] = DEFAULT_EDGE_FEATURES,
        max_entity_transactions: int = 500,
    ):
        self.edge_features = edge_features
        self.max_entity_transactions = max_entity_transactions

    def build(self, df: pd.DataFrame) -> Data:
        df = df.reset_index(drop=True)
        node_features = get_node_features(df)

        edges = self._build_edges(df)

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df)), dtype=torch.long)

        if edges:
            edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def _build_edges(self, df: pd.DataFrame) -> list[tuple[int, int]]:
        edges = set()
        for feature in self.edge_features:
            if feature not in df.columns:
                continue
            groups = df.groupby(feature).groups
            for val, indices in groups.items():
                if val in ("unknown", "", float("nan")):
                    continue
                idx_list = list(indices)
                if len(idx_list) > self.max_entity_transactions:
                    continue
                for i in range(len(idx_list)):
                    for j in range(i + 1, len(idx_list)):
                        a, b = idx_list[i], idx_list[j]
                        edges.add((min(a, b), max(a, b)))
        return list(edges)
