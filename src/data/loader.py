import os
import numpy as np
import pandas as pd

from src.data.preprocessor import NUMERIC_FEATURE_COLS

_EDGE_COLS = ["card1", "card2", "P_emaildomain", "R_emaildomain"]
_CAT_COLS = ["P_emaildomain", "R_emaildomain"]
_NEEDED_COLS = ["TransactionID", "TransactionDT", "isFraud"] + NUMERIC_FEATURE_COLS + _EDGE_COLS

_FLOAT32_COLS = NUMERIC_FEATURE_COLS + ["TransactionDT"]
_DTYPE = {col: np.float32 for col in _FLOAT32_COLS}

_CHUNKSIZE = 10_000
_MEDIAN_SAMPLE_ROWS = 20_000


def _resolve_usecols_and_dtype(tx_path: str) -> tuple[list[str], dict]:
    available = pd.read_csv(tx_path, nrows=0).columns.tolist()
    usecols = [c for c in _NEEDED_COLS if c in available]
    dtype = {k: v for k, v in _DTYPE.items() if k in usecols}
    return usecols, dtype


def _estimate_medians(tx_path: str, usecols: list[str], dtype: dict) -> pd.Series:
    sample = pd.read_csv(tx_path, usecols=usecols, dtype=dtype, nrows=_MEDIAN_SAMPLE_ROWS)
    return sample.select_dtypes(include="number").median()


def _compute_window_boundaries(tx_path: str, n_windows: int) -> np.ndarray:
    dt = pd.read_csv(tx_path, usecols=["TransactionDT"], dtype={"TransactionDT": np.float32})["TransactionDT"]
    return np.percentile(dt.dropna(), np.linspace(0, 100, n_windows + 1))


def load_ieee_cis_windowed(
    data_dir: str,
    n_windows: int = 6,
    max_rows_per_window: int | None = None,
) -> list[pd.DataFrame]:
    tx_path = os.path.join(data_dir, "train_transaction.csv")

    usecols, dtype = _resolve_usecols_and_dtype(tx_path)
    cat_cols = [c for c in _CAT_COLS if c in usecols]

    print("  Pass 1/3: computing window boundaries...")
    boundaries = _compute_window_boundaries(tx_path, n_windows)

    print("  Pass 2/3: estimating fill values from sample...")
    numeric_medians = _estimate_medians(tx_path, usecols, dtype)

    print("  Pass 3/3: loading data in chunks...")
    buckets: list[list[pd.DataFrame]] = [[] for _ in range(n_windows)]
    counts = [0] * n_windows

    for chunk in pd.read_csv(tx_path, usecols=usecols, dtype=dtype, chunksize=_CHUNKSIZE):
        for col in numeric_medians.index:
            if col in chunk.columns:
                chunk[col] = chunk[col].fillna(numeric_medians[col])
        for col in cat_cols:
            chunk[col] = chunk[col].fillna("unknown")

        for i in range(n_windows):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (chunk["TransactionDT"] >= lo) & (
                (chunk["TransactionDT"] <= hi) if i == n_windows - 1 else (chunk["TransactionDT"] < hi)
            )
            sub = chunk[mask]
            if len(sub) == 0:
                continue
            if max_rows_per_window is not None:
                remaining = max_rows_per_window - counts[i]
                if remaining <= 0:
                    continue
                sub = sub.iloc[:remaining]
            buckets[i].append(sub)
            counts[i] += len(sub)

        if max_rows_per_window is not None and all(c >= max_rows_per_window for c in counts):
            break

    return [
        pd.concat(b).reset_index(drop=True) if b else pd.DataFrame()
        for b in buckets
    ]
