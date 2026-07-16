"""
Stage 2: Preprocessing + feature engineering.

engineer_features() adds derived signals on top of the raw columns.
build_preprocessor() returns a sklearn ColumnTransformer that scales
numeric columns and one-hot encodes categoricals, fit only on train data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import TARGET_COL

NUMERIC_FEATURES = [
    "amount",
    "hour_of_day",
    "account_age_days",
    "n_txn_last_hour",
    "distance_from_home_km",
    "amount_to_age_ratio",
    "is_night",
]
CATEGORICAL_FEATURES = [
    "is_foreign_merchant",
    "merchant_category_code",
    "customer_tenure_band",
]
ID_COLS = ["transaction_id"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Extend this as you get access to richer raw data
    (e.g. real velocity windows, counterparty graph features, device signals)."""
    df = df.copy()
    df["amount_to_age_ratio"] = df["amount"] / (df["account_age_days"] + 1)
    df["is_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("scale", StandardScaler())])
    categorical_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def split_X_y(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    return X, y
