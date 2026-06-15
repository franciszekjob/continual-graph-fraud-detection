"""Loading and feature preprocessing for the IEEE-CIS transaction table.

Produces a dense float32 feature matrix (StandardScaler-normalized numericals
+ frequency-encoded categoricals) plus the raw key columns needed for graph
construction. Rows are globally sorted by ``TransactionDT`` so that node index
order is temporal — concept bucketing and within-concept splits then reduce to
simple slicing on positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import DataConfig

logger = logging.getLogger(__name__)

CARD_COLS = [f"card{i}" for i in range(1, 7)]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
V_COLS = [f"V{i}" for i in range(1, 340)]

NUMERICAL_COLS = ["TransactionAmt", "dist1", "dist2", *C_COLS, *D_COLS]
CATEGORICAL_COLS = [
    "ProductCD",
    *CARD_COLS,
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *M_COLS,
]
META_COLS = ["TransactionID", "TransactionDT", "isFraud"]


@dataclass
class PreparedData:
    """Preprocessed dataset, aligned row-by-row (= node-by-node)."""

    key_frame: pd.DataFrame  # raw columns used for edge construction
    features: np.ndarray  # (N, F) float32
    labels: np.ndarray  # (N,) int64, isFraud
    transaction_dt: np.ndarray  # (N,) int64, seconds offset
    transaction_ids: np.ndarray  # (N,) int64, original TransactionID

    @property
    def num_nodes(self) -> int:
        return len(self.labels)


def _frequency_encode(column: pd.Series) -> np.ndarray:
    """Map each category to its relative frequency (missing = own category)."""
    filled = column.fillna("__missing__").astype(str)
    return filled.map(filled.value_counts(normalize=True)).to_numpy(np.float32)


def load_and_preprocess(cfg: DataConfig, edge_key_cols: Iterable[str]) -> PreparedData:
    path = Path(cfg.data_dir) / cfg.transactions_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the IEEE-CIS data with "
            "`kaggle competitions download -c ieee-fraud-detection` "
            f"and extract it into {cfg.data_dir}/."
        )

    header = pd.read_csv(path, nrows=0).columns
    numerical = [c for c in NUMERICAL_COLS if c in header]
    if cfg.use_v_features:
        numerical += [c for c in V_COLS if c in header]
    categorical = [c for c in CATEGORICAL_COLS if c in header]
    edge_keys = [c for c in dict.fromkeys(edge_key_cols) if c in header]

    usecols = list(dict.fromkeys(META_COLS + numerical + categorical + edge_keys))
    logger.info("Reading %s (%d columns)", path, len(usecols))
    df = pd.read_csv(path, usecols=usecols, nrows=cfg.nrows)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    num_block = df[numerical].to_numpy(np.float32)
    medians = np.nanmedian(num_block, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    nan_rows, nan_cols = np.nonzero(np.isnan(num_block))
    num_block[nan_rows, nan_cols] = medians[nan_cols]

    cat_block = np.column_stack([_frequency_encode(df[c]) for c in categorical])

    features = StandardScaler().fit_transform(
        np.hstack([num_block, cat_block])
    ).astype(np.float32)

    logger.info(
        "Prepared %d transactions, %d features (%.1f%% fraud)",
        len(df),
        features.shape[1],
        100 * df["isFraud"].mean(),
    )
    return PreparedData(
        key_frame=df[edge_keys].copy(),
        features=features,
        labels=df["isFraud"].to_numpy(np.int64),
        transaction_dt=df["TransactionDT"].to_numpy(np.int64),
        transaction_ids=df["TransactionID"].to_numpy(np.int64),
    )
