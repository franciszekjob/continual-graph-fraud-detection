import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.preprocessor import get_node_features


DEFAULT_EDGE_FEATURES = ["card1", "card2", "P_emaildomain", "R_emaildomain"]


class TransactionGraphBuilder:
    def __init__(
        self,
        edge_features: list[str] = DEFAULT_EDGE_FEATURES,
        max_entity_transactions: int = 50,
    ):
        self.edge_features = edge_features
        self.max_entity_transactions = max_entity_transactions

    def build(self, df: pd.DataFrame) -> Data:
        df = df.reset_index(drop=True)
        node_features = get_node_features(df)

        edge_index = self._build_edges(df)

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df)), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def _build_edges(self, df: pd.DataFrame) -> torch.Tensor:
        parts = []
        for feature in self.edge_features:
            if feature not in df.columns:
                continue
            groups = df.groupby(feature).groups
            for val, indices in groups.items():
                if val in ("unknown", "", float("nan")):
                    continue
                idx = np.array(indices, dtype=np.int32)
                if len(idx) < 2 or len(idx) > self.max_entity_transactions:
                    continue
                ii, jj = np.triu_indices(len(idx), k=1)
                src, dst = idx[ii], idx[jj]
                parts.append(np.stack([src, dst]))
                parts.append(np.stack([dst, src]))

        if not parts:
            return torch.zeros((2, 0), dtype=torch.long)

        edges = np.concatenate(parts, axis=1)
        edges = np.unique(edges, axis=1)
        return torch.tensor(edges, dtype=torch.long)
