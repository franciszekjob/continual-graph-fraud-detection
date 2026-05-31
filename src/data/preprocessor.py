import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURE_COLS = (
    ["TransactionAmt"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in range(1, 40)]
)


def temporal_split(df: pd.DataFrame, n_windows: int = 6) -> list[pd.DataFrame]:
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    indices = np.array_split(np.arange(len(df)), n_windows)
    return [df.iloc[idx].reset_index(drop=True) for idx in indices]


def get_node_features(df: pd.DataFrame) -> np.ndarray:
    available = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
    X = df[available].values.astype(np.float32)
    scaler = StandardScaler()
    return scaler.fit_transform(X)
