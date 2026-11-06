"""Threshold-aware classification metrics for imbalanced SM/CH evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def contingency_table(y: np.ndarray, prob: np.ndarray, *, threshold: float) -> dict:
    yhat = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    n_ch = int(y.sum())
    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": int(len(y)),
        "n_pred_ch": int(yhat.sum()),
        "n_true_ch": n_ch,
        "sensitivity": float(tp / n_ch) if n_ch else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
    }


def prevalence_matched_table(y: np.ndarray, prob: np.ndarray) -> dict:
    """Predict CH for the top-k scored variants, k = true CH count."""
    k = int(y.sum())
    yhat = np.zeros(len(y), dtype=int)
    if k > 0:
        top_idx = np.argsort(-prob)[:k]
        yhat[top_idx] = 1
    thr = float(prob[np.argsort(-prob)[k - 1]]) if k > 0 else float("inf")
    out = contingency_table(y, prob, threshold=thr)
    out["method"] = "prevalence_matched_top_k"
    out["k"] = k
    return out


def top_fraction_table(y: np.ndarray, prob: np.ndarray, *, frac: float) -> dict:
    """Predict positive for the top `frac` of scores (e.g. 0.05 = top 5%)."""
    n = len(y)
    k = max(1, int(np.ceil(float(frac) * n))) if n else 0
    yhat = np.zeros(n, dtype=int)
    if k > 0:
        top_idx = np.argsort(-prob)[:k]
        yhat[top_idx] = 1
    thr = float(prob[np.argsort(-prob)[k - 1]]) if k > 0 else float("inf")
    out = contingency_table(y, prob, threshold=thr)
    out["method"] = "top_fraction"
    out["frac"] = float(frac)
    out["k"] = int(k)
    return out


def tune_threshold_youden(y: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, prob)
    if len(thresholds) <= 1:
        return 0.5
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def tune_threshold_max_f1(y: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.unique(prob)
    if len(thresholds) == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def score_summary(prob: np.ndarray, y: np.ndarray) -> dict:
    y = y.astype(int)
    sm = prob[y == 0]
    ch = prob[y == 1]
    return {
        "prob_ch_all": {
            "min": float(prob.min()),
            "max": float(prob.max()),
            "mean": float(prob.mean()),
            "median": float(np.median(prob)),
            "n_ge_0_5": int((prob >= 0.5).sum()),
        },
        "prob_ch_true_sm": {
            "mean": float(sm.mean()) if len(sm) else None,
            "median": float(np.median(sm)) if len(sm) else None,
        },
        "prob_ch_true_ch": {
            "mean": float(ch.mean()) if len(ch) else None,
            "median": float(np.median(ch)) if len(ch) else None,
        },
        "true_ch_rate": float(y.mean()),
    }


def ranking_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "n": int(len(y)),
    }


def prevalence_matched_yhat(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Top-k prevalence-matched predictions (k = number of true positives)."""
    k = int(np.asarray(y).sum())
    yhat = np.zeros(len(y), dtype=int)
    if k > 0:
        yhat[np.argsort(-np.asarray(prob, dtype=float))[:k]] = 1
    return yhat


def attach_cosmic_haem_flags(
    df: pd.DataFrame,
    *,
    cosmic_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Ensure ``n_haem`` / ``haem_ge3`` columns exist (join COSMIC haem≥1 table)."""
    out = df.copy()
    out["variant_id"] = out["variant_id"].astype(str)
    if cosmic_csv is None:
        # sm_ch_finetune/src/eval -> parents[2] = sm_ch_finetune
        cosmic_csv = (
            Path(__file__).resolve().parents[2]
            / "data/arch_mane_noncoding/arch_mane_noncoding_somatic_cosmic_ncv_haem_min1.csv"
        )
    cos = pd.read_csv(cosmic_csv, usecols=["variant_id", "n_haem"])
    cos["variant_id"] = cos["variant_id"].astype(str)
    out = out.drop(columns=[c for c in ("n_haem", "haem_ge3") if c in out.columns])
    out = out.merge(cos, on="variant_id", how="left")
    out["haem_ge3"] = (out["n_haem"].fillna(0).astype(float) >= 3).astype(int)
    return out


def haem_ge3_membership(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extra membership mask for source_sensitivity_pm (requires haem_ge3 col)."""
    if "haem_ge3" not in df.columns:
        raise KeyError("haem_ge3 missing; call attach_cosmic_haem_flags first")
    return {"cosmic_haem_ge3": df["haem_ge3"].fillna(0).astype(int).astype(bool).values}


def source_sensitivity_pm(
    y: np.ndarray,
    prob: np.ndarray,
    source,
    *,
    extra_membership: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Sensitivity under prevalence-matched threshold, overall + membership groups.

    Default membership among true somatic (y==1):
    - ``gnomad``: source contains ``gnomad_aspab`` or ``paper_mw``
    - ``cosmic``: source contains ``cosmic``
    Optional ``extra_membership`` adds more groups (e.g. ``cosmic_haem_ge3``).
    Alleles may belong to multiple groups.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    src = np.asarray([str(s) if s is not None else "" for s in source], dtype=object)
    pm = prevalence_matched_table(y, prob)
    yhat = prevalence_matched_yhat(y, prob)
    rows: list[dict] = [
        {
            "scope": "overall",
            "source": "all",
            "n": int(len(y)),
            "n_true_pos": int(y.sum()),
            "n_pred_pos": int(yhat.sum()),
            "sensitivity": pm.get("sensitivity"),
            "specificity": pm.get("specificity"),
            "tp": pm.get("tp"),
            "fp": pm.get("fp"),
            "fn": pm.get("fn"),
            "tn": pm.get("tn"),
        }
    ]
    som_mask = y == 1
    src_l = np.array([s.lower() for s in src], dtype=object)
    membership: dict[str, np.ndarray] = {
        "gnomad": np.array(["paper_mw" in s for s in src_l], dtype=bool),
        "cosmic": np.array(["cosmic" in s for s in src_l], dtype=bool),
    }
    if extra_membership:
        for tag, mask in extra_membership.items():
            membership[tag] = np.asarray(mask, dtype=bool)
    for tag, tag_mask in membership.items():
        m = som_mask & tag_mask
        n = int(m.sum())
        rows.append(
            {
                "scope": "source_membership",
                "source": tag,
                "n": n,
                "n_true_pos": n,
                "n_pred_pos": int(yhat[m].sum()) if n else 0,
                "sensitivity": float(yhat[m].mean()) if n else None,
                "specificity": None,
                "tp": int(yhat[m].sum()) if n else 0,
                "fp": None,
                "fn": int((1 - yhat[m]).sum()) if n else 0,
                "tn": None,
            }
        )
    return rows



def print_contingency(title: str, table: dict) -> None:
    print(f"\n{title}")
    if table.get("method") == "prevalence_matched_top_k":
        print(f"  (top {table['k']} scores predicted CH; implicit threshold={table['threshold']:.4f})")
    elif table.get("method") == "saved_pred_ch":
        print("  (from saved pred_ch column)")
    elif table.get("threshold") is not None and not np.isnan(table["threshold"]):
        print(f"  threshold={table['threshold']:.4f}")
    print(f"              pred SM   pred CH")
    print(f"  true SM       {table['tn']:4d}      {table['fp']:4d}")
    print(f"  true CH       {table['fn']:4d}      {table['tp']:4d}")
    print(
        f"  sens={table['sensitivity']:.3f} spec={table['specificity']:.3f} "
        f"F1={table['f1']:.3f} bal_acc={table['balanced_accuracy']:.3f}"
    )
