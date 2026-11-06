"""Classifiers for CD34+ API variant track-score features."""
from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def make_track_classifier(
    name: str,
    *,
    C: float = 0.5,
    max_iter: int = 2000,
    random_state: int = 42,
    scale_pos_weight: float | None = None,
) -> ClassifierMixin:
    if name == "lr":
        return LogisticRegression(max_iter=max_iter, C=C)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            class_weight="balanced",
            random_state=random_state,
        )
    if name in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "xgboost is required for classifier 'xgb'. "
                "Install with: ./run_binemo_l40.sh -m pip install xgboost"
            ) from e
        # Mirror RF depth; handle imbalance via scale_pos_weight (default 1 if unset)
        spw = 1.0 if scale_pos_weight is None else float(scale_pos_weight)
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=random_state,
            n_jobs=4,
            tree_method="hist",
        )
    raise ValueError(f"Unknown track model: {name!r} (use lr, rf, or xgb)")


def fit_predict_proba(
    clf: ClassifierMixin,
    X_tr,
    y_tr,
    X_te,
    *,
    oof_train: bool = False,
    cv: int = 5,
) -> tuple:
    """Fit classifier; return (train_proba or None, test_proba)."""
    if oof_train:
        proba_tr = cross_val_predict(clf, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)[:, 1]
        return proba_tr, proba_te
    clf.fit(X_tr, y_tr)
    proba_te = clf.predict_proba(X_te)[:, 1]
    return None, proba_te
