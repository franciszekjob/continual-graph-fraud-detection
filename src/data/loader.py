import os
import pandas as pd


def load_ieee_cis(data_dir: str) -> pd.DataFrame:
    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")

    transactions = pd.read_csv(tx_path)
    identity = pd.read_csv(id_path)

    df = transactions.merge(identity, on="TransactionID", how="left")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("unknown")

    return df
