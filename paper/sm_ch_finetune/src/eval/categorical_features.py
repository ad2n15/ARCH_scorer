"""Categorical covariates for tabular ensemble models."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def _bucket_top_k(series: pd.Series, top_k: int | None) -> pd.Series:
    s = series.fillna("unknown").astype(str)
    if top_k is None or top_k <= 0:
        return s
    keep = set(s.value_counts().head(top_k).index)
    return s.where(s.isin(keep), "__other__")


def fit_one_hot(
    train: pd.Series,
    test: pd.Series,
    *,
    col_name: str,
    top_k: int | None = None,
) -> tuple:
    """Fit one-hot encoder on train; return (X_train, X_test, feature_names)."""
    tr = _bucket_top_k(train, top_k).to_frame()
    te = _bucket_top_k(test, top_k).to_frame()
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(tr)
    X_te = enc.transform(te)
    names = list(enc.get_feature_names_out([col_name]))
    return X_tr, X_te, names
