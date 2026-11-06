#!/usr/bin/env python3
"""Publication-oriented CH-gene germline vs somatic analysis (plus no-age SM).

Steps:
  1) Load pilots + API; ensure AF/CADD/VEP (gnomAD extracts; paper CADD fallback)
  2) Pooled univariate associations (logistic OR / AUROC) for each feature
  3) Per-gene multivariate CV: baseline / API / both (LR + RF)
  4) Sensitivity / specificity per gene
  5) Stratified performance (pooled + by gene where n allows)
  6) Repeat (3–5) after excluding synonymous variants
  7) Top-3 RF importances (API alone; API+baseline)
  8) SHAP (TreeExplainer) for pooled RF: mean|SHAP| tables + beeswarm / dependence plots

Baseline features: mutation_type, vep_consequence, cadd_phred, log10(AF).
API: 10 CD34+ AlphaGenome track scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from eval.categorical_features import fit_one_hot
from eval.classification_metrics import (
    contingency_table,
    prevalence_matched_table,
    ranking_metrics,
    top_fraction_table,
)
from paths import project_subpath, resolve_config
from scoring.track_classifiers import make_track_classifier

warnings.filterwarnings("ignore", category=UserWarning)

AF_EPS = 1e-8
SHORT = {
    "RNA_SEQ__CL_0001059_polyA_plus_RNA_seq__GeneMaskLFC": "RNA_SEQ",
    "SPLICE_SITE_USAGE__usage_CL_0001059_polyA_plus_RNA_seq__GeneMaskSplicing": "SPLICE_USAGE",
    "SPLICE_JUNCTIONS__junction_CL_0001059_polyA_plus_RNA_seq__SpliceJunction": "SPLICE_JUNC",
    "DNASE__CL_0001059_DNase_seq__CenterMask": "DNASE",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K27ac__CenterMask": "H3K27ac",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K27me3__CenterMask": "H3K27me3",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K36me3__CenterMask": "H3K36me3",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K4me1__CenterMask": "H3K4me1",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K4me3__CenterMask": "H3K4me3",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K9me3__CenterMask": "H3K9me3",
}
SYN_LIKE = {
    "synonymous_variant",
    "synonymous_variant&splice_region_variant",
    "splice_region_variant&synonymous_variant",
}


def _feat_cols(df: pd.DataFrame) -> list[str]:
    meta = {
        "variant_id", "label", "gene", "func", "exonic_func", "split", "af", "cadd_phred",
        "label_str", "class_source", "mutation_type", "vep_consequence", "consequence",
        "chrom", "pos", "ref", "alt", "ab_median_approx", "y", "log10_af",
    }
    return [
        c for c in df.columns
        if c not in meta and not str(c).endswith("__quantile") and pd.api.types.is_numeric_dtype(df[c])
        and any(k in c for k in ("RNA_SEQ", "SPLICE", "DNASE", "CHIP_HISTONE", "ATAC", "CAGE", "PROCAP", "CHIP_TF"))
    ]


def _primary_cons(s: str) -> str:
    return str(s).split("&")[0] if pd.notna(s) else "unknown"


def _is_synonymous(s: str) -> bool:
    t = str(s).lower()
    if t in SYN_LIKE:
        return True
    # primary consequence synonymous
    return _primary_cons(t) == "synonymous_variant"


def _load_all(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    pilots = project_subpath(PROJECT, cfg["data"]["pilots_dir"])
    scores_root = project_subpath(PROJECT, cfg["api"]["scores_root"])
    germ_dir = project_subpath(PROJECT, cfg["data"]["germline_cadd_dir"])
    parts = []
    for gdir in sorted(pilots.iterdir()):
        if not gdir.is_dir() or not (gdir / "train.parquet").exists():
            continue
        gene = gdir.name
        meta = pd.concat(
            [pd.read_parquet(gdir / f"{s}.parquet") for s in ("train", "valid", "test") if (gdir / f"{s}.parquet").exists()],
            ignore_index=True,
        )
        meta["variant_id"] = meta["variant_id"].astype(str)
        meta["gene"] = gene
        allp = germ_dir / f"{gene}_all_sites_cadd.parquet"
        if allp.exists():
            lu = pd.read_parquet(
                allp, columns=["variant_id", "af", "cadd_phred", "vep_consequence"]
            ).drop_duplicates("variant_id")
            lu["variant_id"] = lu["variant_id"].astype(str)
            lu = lu.rename(
                columns={
                    "af": "af_gnomad",
                    "cadd_phred": "cadd_gnomad",
                    "vep_consequence": "vep_gnomad",
                }
            )
            meta = meta.merge(lu, on="variant_id", how="left")
            meta["af"] = pd.to_numeric(meta.get("af"), errors="coerce")
            meta["af"] = meta["af"].fillna(pd.to_numeric(meta["af_gnomad"], errors="coerce"))
            meta["cadd_phred"] = pd.to_numeric(meta.get("cadd_phred"), errors="coerce")
            meta["cadd_phred"] = meta["cadd_phred"].fillna(
                pd.to_numeric(meta["cadd_gnomad"], errors="coerce")
            )
            unk = meta["vep_consequence"].isna() | (meta["vep_consequence"].astype(str) == "unknown")
            meta.loc[unk, "vep_consequence"] = meta.loc[unk, "vep_gnomad"]
            meta = meta.drop(columns=["af_gnomad", "cadd_gnomad", "vep_gnomad"], errors="ignore")

        sd = scores_root / gene
        if gene == "DNMT3A" and not (sd / "train_features_wide.parquet").exists():
            sd = project_subpath(PROJECT, "results/track_scores_dnmt3a_germline_vs_somatic")
        api_parts = []
        for s in ("train", "valid", "test"):
            p = sd / f"{s}_features_wide.parquet"
            if p.exists():
                api_parts.append(pd.read_parquet(p))
        if not api_parts:
            continue
        api = pd.concat(api_parts, ignore_index=True)
        api["variant_id"] = api["variant_id"].astype(str)
        feats = _feat_cols(api)
        api = api.drop_duplicates("variant_id")
        m = meta.merge(api[["variant_id"] + feats], on="variant_id", how="inner")
        parts.append(m)

    df = pd.concat(parts, ignore_index=True).drop_duplicates(["gene", "variant_id"])
    df["y"] = df["label"].astype(int)
    df["mutation_type"] = df["mutation_type"].fillna("unknown").astype(str)
    df["vep_consequence"] = df["vep_consequence"].fillna("unknown").astype(str)
    df["cons_primary"] = df["vep_consequence"].map(_primary_cons)
    df["cadd_phred"] = pd.to_numeric(df["cadd_phred"], errors="coerce")
    df["af"] = pd.to_numeric(df["af"], errors="coerce")
    df["log10_af"] = np.log10(np.clip(df["af"].astype(float), 0, None) + AF_EPS)
    df.loc[~np.isfinite(df["log10_af"]), "log10_af"] = np.nan
    feat_cols = _feat_cols(df)
    return df, feat_cols


def _load_sm_ch_map(age_dir: Path) -> pd.DataFrame:
    """variant_id -> SM/CH from age-filtered cohort parquet."""
    parts = []
    for split in ("train", "valid", "test"):
        p = age_dir / f"{split}.parquet"
        if not p.exists():
            continue
        parts.append(pd.read_parquet(p, columns=["variant_id", "label_str"]))
    if not parts:
        return pd.DataFrame(columns=["variant_id", "sm_ch"])
    cohort = pd.concat(parts, ignore_index=True)
    cohort["variant_id"] = cohort["variant_id"].astype(str)
    cohort = cohort[cohort["label_str"].isin(["SM", "CH"])].drop_duplicates("variant_id")
    return cohort[["variant_id", "label_str"]].rename(columns={"label_str": "sm_ch"})


def _prevalence_matched_yhat(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    k = int(np.asarray(y).sum())
    yhat = np.zeros(len(y), dtype=int)
    if k > 0:
        yhat[np.argsort(-prob)[:k]] = 1
    return yhat


def _top_frac_yhat(prob: np.ndarray, frac: float) -> np.ndarray:
    n = len(prob)
    k = max(1, int(np.ceil(float(frac) * n))) if n else 0
    yhat = np.zeros(n, dtype=int)
    if k > 0:
        yhat[np.argsort(-prob)[:k]] = 1
    return yhat


def _subtype_sensitivity(
    y: np.ndarray,
    prob: np.ndarray,
    subtype: np.ndarray,
    label: str,
    *,
    yhat: np.ndarray | None = None,
) -> dict:
    """Sensitivity among true class `label` (e.g. CH) at a fixed positive call set."""
    if yhat is None:
        yhat = _prevalence_matched_yhat(y, prob)
    mask = (np.asarray(subtype).astype(str) == label) & (np.asarray(y).astype(int) == 1)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "sensitivity": None}
    return {"n": n, "sensitivity": float(np.asarray(yhat)[mask].mean())}


def _logit_wald(X, y, class_weight: str = "balanced"):
    """Fit logistic regression; return coef, se, z, p, or_ (length = n_features, no intercept in return)."""
    from scipy.stats import norm

    clf = LogisticRegression(max_iter=2000, class_weight=class_weight, solver="lbfgs")
    clf.fit(X, y)
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])
    # design with intercept for Hessian
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.concatenate([[intercept], coef])
    p = 1.0 / (1.0 + np.exp(-Xd @ beta))
    w = p * (1.0 - p)
    # optional class reweight to match balanced LR roughly for SE
    if class_weight == "balanced":
        n_pos = max(float(y.sum()), 1.0)
        n_neg = max(float(len(y) - y.sum()), 1.0)
        sw = np.where(y == 1, 0.5 * len(y) / n_pos, 0.5 * len(y) / n_neg)
        w = w * sw
    XtW = Xd.T * w
    try:
        cov = np.linalg.pinv(XtW @ Xd)
    except np.linalg.LinAlgError:
        cov = np.eye(Xd.shape[1]) * np.nan
    se = np.sqrt(np.clip(np.diag(cov)[1:], 1e-16, None))
    z = coef / se
    pval = 2.0 * norm.sf(np.abs(z))
    lo = coef - 1.96 * se
    hi = coef + 1.96 * se
    return coef, se, z, pval, lo, hi, np.exp(coef), np.exp(lo), np.exp(hi)


def _univariate_continuous(y: np.ndarray, x: np.ndarray, name: str) -> dict:
    mask = np.isfinite(x)
    y2, x2 = y[mask], x[mask]
    if len(y2) < 50 or y2.sum() < 5 or (len(y2) - y2.sum()) < 5:
        return {
            "feature": name,
            "n": int(mask.sum()),
            "coef": None,
            "coef_ci_lo": None,
            "coef_ci_hi": None,
            "or_per_sd": None,
            "or_ci_lo": None,
            "or_ci_hi": None,
            "p_value": None,
            "auroc": None,
        }
    xs = StandardScaler().fit_transform(x2.reshape(-1, 1))
    coef, se, z, pval, lo, hi, ors, or_lo, or_hi = _logit_wald(xs, y2)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(xs, y2)
    prob = clf.predict_proba(xs)[:, 1]
    auroc = float(roc_auc_score(y2, prob))
    auroc_dir = float(roc_auc_score(y2, x2 if coef[0] >= 0 else -x2))
    return {
        "feature": name,
        "n": int(len(y2)),
        "n_pos": int(y2.sum()),
        "coef": float(coef[0]),
        "coef_ci_lo": float(lo[0]),
        "coef_ci_hi": float(hi[0]),
        "or_per_sd": float(ors[0]),
        "or_ci_lo": float(or_lo[0]),
        "or_ci_hi": float(or_hi[0]),
        "p_value": float(pval[0]),
        "auroc_model": auroc,
        "auroc_raw": auroc_dir,
        "mean_germ": float(x2[y2 == 0].mean()),
        "mean_som": float(x2[y2 == 1].mean()),
    }


def _univariate_categorical(y: np.ndarray, cat: pd.Series, name: str, min_n: int = 30) -> list[dict]:
    rows = []
    dummies = pd.get_dummies(cat.fillna("unknown").astype(str), prefix=name)
    keep = [c for c in dummies.columns if dummies[c].sum() >= min_n]
    if len(keep) < 2:
        return rows
    # reference = most frequent level
    ref = max(keep, key=lambda c: int(dummies[c].sum()))
    others = [c for c in keep if c != ref]
    X = dummies[others].values.astype(float)
    coef, se, z, pval, lo, hi, ors, or_lo, or_hi = _logit_wald(X, y)
    for i, nm in enumerate(others):
        rows.append(
            {
                "feature": nm,
                "group": name,
                "reference": ref,
                "n": int(dummies[nm].sum()),
                "coef": float(coef[i]),
                "or": float(ors[i]),
                "or_ci_lo": float(or_lo[i]),
                "or_ci_hi": float(or_hi[i]),
                "p_value": float(pval[i]),
            }
        )
    tab = pd.crosstab(cat.fillna("unknown"), y)
    for level in tab.index:
        n = int(tab.loc[level].sum())
        if n < min_n:
            continue
        n_pos = int(tab.loc[level, 1]) if 1 in tab.columns else 0
        rows.append(
            {
                "feature": f"{name}={level}",
                "group": f"{name}_rate",
                "n": n,
                "n_pos": n_pos,
                "somatic_rate": float(n_pos / n) if n else None,
            }
        )
    return rows


def _baseline_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    include_af: bool = True,
    include_gene: bool = False,
):
    # Cap rare levels — full VEP strings explode on pooled ~20k variants and stall LR.
    mut_tr, mut_te = fit_one_hot(
        train["mutation_type"], test["mutation_type"], col_name="mutation_type", top_k=20
    )[:2]
    cons_col = "cons_primary" if "cons_primary" in train.columns else "vep_consequence"
    cons_tr, cons_te = fit_one_hot(
        train[cons_col], test[cons_col], col_name=cons_col, top_k=25
    )[:2]
    med_c = float(np.nanmedian(train["cadd_phred"].astype(float)))
    if not np.isfinite(med_c):
        med_c = 0.0
    cadd_tr = train["cadd_phred"].astype(float).fillna(med_c).values.reshape(-1, 1)
    cadd_te = test["cadd_phred"].astype(float).fillna(med_c).values.reshape(-1, 1)
    parts_tr = [mut_tr, cons_tr, cadd_tr]
    parts_te = [mut_te, cons_te, cadd_te]
    names = (
        [f"mut={c}" for c in range(mut_tr.shape[1])]
        + [f"cons={c}" for c in range(cons_tr.shape[1])]
        + ["cadd_phred"]
    )
    if include_af:
        med_a = float(np.nanmedian(train["log10_af"].astype(float)))
        if not np.isfinite(med_a):
            med_a = float(np.log10(AF_EPS))
        af_tr = train["log10_af"].astype(float).fillna(med_a).values.reshape(-1, 1)
        af_te = test["log10_af"].astype(float).fillna(med_a).values.reshape(-1, 1)
        parts_tr.append(af_tr)
        parts_te.append(af_te)
        names = names + ["log10_af"]
    if include_gene:
        gene_tr, gene_te, gene_names = fit_one_hot(
            train["gene"], test["gene"], col_name="gene", top_k=None
        )
        parts_tr.append(gene_tr)
        parts_te.append(gene_te)
        names = names + gene_names
    return (np.hstack(parts_tr), np.hstack(parts_te), names)


def _cv_models(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    n_splits: int,
    seed: int,
    classifiers: tuple[str, ...] = ("lr", "rf"),
    feature_sets: tuple[str, ...] = ("baseline", "api", "both"),
    include_af: bool = True,
    include_gene: bool = False,
    quiet: bool = False,
) -> dict:
    y = df["y"].values
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos < n_splits or n_neg < n_splits:
        return {"status": "too_few", "n": len(df), "n_pos": n_pos}

    imp = SimpleImputer(strategy="median")
    X_api = imp.fit_transform(df[feat_cols].replace([np.inf, -np.inf], np.nan).values)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model_specs = []
    for which in feature_sets:
        for kind in classifiers:
            prefix = {
                "baseline": "baseline_plus_gene" if include_gene else "baseline",
                "api": "api",
                "both": (
                    "api_plus_baseline_plus_gene"
                    if include_gene
                    else "api_plus_baseline"
                ),
            }[which]
            model_specs.append((f"{prefix}_{kind}", kind, which))
    oof = {m: np.full(len(df), np.nan) for m, _, _ in model_specs}
    fold_metrics = {m: [] for m, _, _ in model_specs}
    use_saga = len(df) >= 5000

    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(df)), y), 1):
        if not quiet:
            print(f"    fold {fold_i}/{n_splits} n_train={len(tr)} …", flush=True)
        train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        y_tr, y_te = train["y"].values, test["y"].values
        base_tr, base_te, _ = _baseline_matrix(
            train, test, include_af=include_af, include_gene=include_gene
        )
        api_tr, api_te = X_api[tr], X_api[te]
        mats = {
            "baseline": (base_tr, base_te),
            "api": (api_tr, api_te),
            "both": (np.hstack([api_tr, base_tr]), np.hstack([api_te, base_te])),
        }
        for name, kind, which in model_specs:
            X_tr, X_te = mats[which]
            if kind == "lr":
                clf = LogisticRegression(
                    max_iter=500 if use_saga else 2000,
                    C=0.5,
                    class_weight="balanced",
                    solver="saga" if use_saga else "lbfgs",
                    tol=1e-3,
                )
            elif kind in ("rf", "xgb", "xgboost"):
                kind_key = "xgb" if kind in ("xgb", "xgboost") else "rf"
                n_pos_tr = int(y_tr.sum())
                n_neg_tr = int(len(y_tr) - n_pos_tr)
                spw = (n_neg_tr / n_pos_tr) if (kind_key == "xgb" and n_pos_tr > 0) else None
                clf = make_track_classifier(
                    kind_key, random_state=seed + fold_i, scale_pos_weight=spw
                )
            else:
                raise ValueError(f"Unknown classifier: {kind!r}")
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            m = ranking_metrics(y_te, prob)
            fold_metrics[name].append(m["auroc"])
            oof[name][te] = prob
            if not quiet:
                print(f"      {name} AUROC={m['auroc']:.3f}", flush=True)

    out_models = []
    subtype = None
    if "sm_ch" in df.columns:
        subtype = df["sm_ch"].fillna("unknown").astype(str).values
    for name, kind, which in model_specs:
        m = ranking_metrics(y, oof[name])
        prev = prevalence_matched_table(y, oof[name])
        t05 = contingency_table(y, oof[name], threshold=0.5)
        top5 = top_fraction_table(y, oof[name], frac=0.05)
        top10 = top_fraction_table(y, oof[name], frac=0.10)
        row = {
            "model": name,
            "classifier": kind,
            "features": which,
            "auroc_mean": float(np.mean(fold_metrics[name])),
            "auroc_std": float(np.std(fold_metrics[name], ddof=0)),
            "oof_auroc": m["auroc"],
            "oof_auprc": m["auprc"],
            "sens_prev": prev["sensitivity"],
            "spec_prev": prev["specificity"],
            "sens_top5": top5["sensitivity"],
            "spec_top5": top5["specificity"],
            "k_top5": top5["k"],
            "sens_top10": top10["sensitivity"],
            "spec_top10": top10["specificity"],
            "k_top10": top10["k"],
            "sens_0.5": t05["sensitivity"],
            "spec_0.5": t05["specificity"],
        }
        if subtype is not None:
            ch = _subtype_sensitivity(y, oof[name], subtype, "CH")
            sm = _subtype_sensitivity(y, oof[name], subtype, "SM")
            row["n_CH"] = ch["n"]
            row["sens_CH_prev"] = ch["sensitivity"]
            row["n_SM"] = sm["n"]
            row["sens_SM_prev"] = sm["sensitivity"]
            yhat5 = _top_frac_yhat(oof[name], 0.05)
            yhat10 = _top_frac_yhat(oof[name], 0.10)
            ch5 = _subtype_sensitivity(y, oof[name], subtype, "CH", yhat=yhat5)
            ch10 = _subtype_sensitivity(y, oof[name], subtype, "CH", yhat=yhat10)
            row["sens_CH_top5"] = ch5["sensitivity"]
            row["sens_CH_top10"] = ch10["sensitivity"]
        out_models.append(row)
    return {
        "status": "ok",
        "n": int(len(df)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "models": out_models,
        "oof": {k: v.tolist() for k, v in oof.items()},
        "y": y.tolist(),
    }


def _top_k_series(s: pd.Series, k: int) -> pd.Series:
    s = s.fillna("unknown").astype(str)
    keep = set(s.value_counts().head(k).index)
    return s.where(s.isin(keep), "__other__")


def _design_matrices(df: pd.DataFrame, feat_cols: list[str], *, include_af: bool = True) -> dict:
    """Full-data design matrices with stable feature names (for importance / SHAP)."""
    imp = SimpleImputer(strategy="median")
    X_api = imp.fit_transform(df[feat_cols].replace([np.inf, -np.inf], np.nan).values)
    enc_m = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc_c = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xm = enc_m.fit_transform(_top_k_series(df["mutation_type"], 20).values.reshape(-1, 1))
    cons_col = "cons_primary" if "cons_primary" in df.columns else "vep_consequence"
    Xc = enc_c.fit_transform(_top_k_series(df[cons_col], 25).values.reshape(-1, 1))
    med_c = float(np.nanmedian(df["cadd_phred"]))
    if not np.isfinite(med_c):
        med_c = 0.0
    cadd = df["cadd_phred"].astype(float).fillna(med_c).values.reshape(-1, 1)
    parts = [Xm, Xc, cadd]
    base_names = (
        [f"mut={v}" for v in enc_m.categories_[0]]
        + [f"cons={v}" for v in enc_c.categories_[0]]
        + ["cadd_phred"]
    )
    if include_af:
        med_a = float(np.nanmedian(df["log10_af"]))
        if not np.isfinite(med_a):
            med_a = float(np.log10(AF_EPS))
        laf = df["log10_af"].astype(float).fillna(med_a).values.reshape(-1, 1)
        parts.append(laf)
        base_names = base_names + ["log10_af"]
    X_base = np.hstack(parts)
    api_names = [SHORT.get(c, c) for c in feat_cols]
    return {
        "y": df["y"].values.astype(int),
        "X_api": X_api,
        "X_base": X_base,
        "X_both": np.hstack([X_api, X_base]),
        "api_names": api_names,
        "base_names": base_names,
        "both_names": api_names + base_names,
    }


def _feature_importance(
    df: pd.DataFrame, feat_cols: list[str], *, seed: int, n_splits: int = 5, include_af: bool = True
) -> dict:
    y = df["y"].values
    if y.sum() < n_splits or (len(y) - y.sum()) < n_splits:
        return {}
    dm = _design_matrices(df, feat_cols, include_af=include_af)
    X_api, X_both = dm["X_api"], dm["X_both"]
    api_names, both_names = dm["api_names"], dm["both_names"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    api_rf, both_rf, api_xgb, both_xgb = [], [], [], []
    for fold_i, (tr, _) in enumerate(skf.split(np.zeros(len(df)), y), 1):
        y_tr = y[tr]
        n_pos_tr = int(y_tr.sum())
        n_neg_tr = int(len(y_tr) - n_pos_tr)
        spw = (n_neg_tr / n_pos_tr) if n_pos_tr > 0 else 1.0
        clf_a = make_track_classifier("rf", random_state=seed + fold_i)
        clf_a.fit(X_api[tr], y_tr)
        api_rf.append(clf_a.feature_importances_)
        clf_b = make_track_classifier("rf", random_state=seed + fold_i)
        clf_b.fit(X_both[tr], y_tr)
        both_rf.append(clf_b.feature_importances_)
        clf_xa = make_track_classifier("xgb", random_state=seed + fold_i, scale_pos_weight=spw)
        clf_xa.fit(X_api[tr], y_tr)
        api_xgb.append(clf_xa.feature_importances_)
        clf_xb = make_track_classifier("xgb", random_state=seed + fold_i, scale_pos_weight=spw)
        clf_xb.fit(X_both[tr], y_tr)
        both_xgb.append(clf_xb.feature_importances_)
    api_rf_m = np.mean(api_rf, axis=0)
    both_rf_m = np.mean(both_rf, axis=0)
    api_xgb_m = np.mean(api_xgb, axis=0)
    both_xgb_m = np.mean(both_xgb, axis=0)

    def topk(names, vals, k=3):
        order = np.argsort(-vals)
        return [{"rank": i + 1, "feature": names[j], "importance": float(vals[j])} for i, j in enumerate(order[:k])]

    return {
        "api_rf_top3": topk(api_names, api_rf_m, 3),
        "api_plus_baseline_rf_top3": topk(both_names, both_rf_m, 3),
        "api_rf_all": topk(api_names, api_rf_m, len(api_names)),
        "api_plus_baseline_rf_all": topk(both_names, both_rf_m, min(20, len(both_names))),
        "api_xgb_top3": topk(api_names, api_xgb_m, 3),
        "api_plus_baseline_xgb_top3": topk(both_names, both_xgb_m, 3),
        "api_xgb_all": topk(api_names, api_xgb_m, len(api_names)),
        "api_plus_baseline_xgb_all": topk(both_names, both_xgb_m, min(20, len(both_names))),
    }


def _shap_class1(values) -> np.ndarray:
    """Return (n, p) SHAP matrix for the positive (somatic) class."""
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
    return arr


def _shap_tree_block(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    seed: int,
    max_samples: int,
    plot_dir: Path,
    tag: str,
    model_key: str,
    clf_kind: str = "rf",
) -> dict:
    """TreeExplainer SHAP for RF or XGB (somatic = class 1)."""
    import shap

    n = len(y)
    if n > max_samples:
        idx, _ = train_test_split(
            np.arange(n), train_size=max_samples, stratify=y, random_state=seed
        )
        idx = np.sort(idx)
    else:
        idx = np.arange(n)
    X_s, y_s = X[idx], y[idx]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = (n_neg / n_pos) if (clf_kind == "xgb" and n_pos > 0) else None
    clf = make_track_classifier(clf_kind, random_state=seed, scale_pos_weight=spw)
    clf.fit(X, y)
    explainer = shap.TreeExplainer(clf)
    sv = _shap_class1(explainer.shap_values(X_s))
    # XGB sometimes returns (n, p) already for binary; ensure 2d
    sv = np.asarray(sv)
    if sv.ndim == 1:
        sv = sv.reshape(-1, 1)
    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_signed = np.mean(sv, axis=0)
    order = np.argsort(-mean_abs)
    rows = [
        {
            "rank": r + 1,
            "feature": names[j],
            "mean_abs_shap": float(mean_abs[j]),
            "mean_shap": float(mean_signed[j]),
        }
        for r, j in enumerate(order)
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(plot_dir / f"shap_summary_{model_key}_{tag}.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, X_s, feature_names=names, max_display=20, show=False)
    plt.tight_layout()
    bee = plot_dir / f"shap_beeswarm_{model_key}_{tag}.png"
    plt.savefig(bee, dpi=150, bbox_inches="tight")
    plt.close("all")

    dep_feats = []
    for j in order:
        name = names[j]
        if name.startswith("mut=") or name.startswith("cons="):
            continue
        dep_feats.append(j)
        if len(dep_feats) >= 3:
            break
    if not dep_feats:
        dep_feats = list(order[:3])
    for rank, j in enumerate(dep_feats, 1):
        plt.figure()
        shap.dependence_plot(j, sv, X_s, feature_names=names, show=False)
        plt.tight_layout()
        plt.savefig(
            plot_dir / f"shap_dependence_{model_key}_top{rank}_{names[j].replace('/', '_')}_{tag}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")

    proba = clf.predict_proba(X_s)[:, 1]
    examples = {
        "highest_somatic_prob": int(np.argmax(proba)),
        "lowest_somatic_prob": int(np.argmin(proba)),
    }
    pos = np.where(y_s == 1)[0]
    neg = np.where(y_s == 0)[0]
    if len(pos):
        examples["true_somatic"] = int(pos[np.argmax(proba[pos])])
    if len(neg):
        examples["true_germline"] = int(neg[np.argmin(proba[neg])])
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.asarray(expected).ravel()[-1])
    else:
        expected = float(expected)
    for label, i in examples.items():
        try:
            exp = shap.Explanation(
                values=sv[i],
                base_values=expected,
                data=X_s[i],
                feature_names=names,
            )
            plt.figure()
            shap.plots.waterfall(exp, max_display=12, show=False)
            plt.tight_layout()
            plt.savefig(
                plot_dir / f"shap_waterfall_{model_key}_{label}_{tag}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")
        except Exception as e:
            print(f"  waterfall {model_key}/{label} skipped: {e}")

    print(
        f"  SHAP {model_key}: n_explain={len(idx)} top3="
        + str([{"feature": r["feature"], "mean_abs_shap": round(r["mean_abs_shap"], 4)} for r in rows[:3]])
    )
    return {
        "model": model_key,
        "clf_kind": clf_kind,
        "n_fit": int(n),
        "n_explain": int(len(idx)),
        "top10": rows[:10],
        "beeswarm": str(bee.name),
    }


# backwards-compatible alias
def _shap_rf_block(*args, **kwargs):
    kwargs.setdefault("clf_kind", "rf")
    return _shap_tree_block(*args, **kwargs)


def _shap_analysis(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    seed: int,
    include_af: bool,
    max_samples: int,
    out_dir: Path,
    tag: str,
    label: str = "",
) -> dict:
    """Fit pooled RF + XGB; TreeExplainer SHAP on a stratified subsample."""
    y = df["y"].values
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return {"status": "too_few", "n": int(len(df)), "n_pos": int(y.sum())}
    dm = _design_matrices(df, feat_cols, include_af=include_af)
    plot_dir = out_dir / f"shap_{tag}" / label if label else out_dir / f"shap_{tag}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    file_tag = f"{label}_{tag}" if label else tag
    blocks = {}
    for key, X, names, kind in (
        ("api_rf", dm["X_api"], dm["api_names"], "rf"),
        ("api_plus_baseline_rf", dm["X_both"], dm["both_names"], "rf"),
        ("api_xgb", dm["X_api"], dm["api_names"], "xgb"),
        ("api_plus_baseline_xgb", dm["X_both"], dm["both_names"], "xgb"),
    ):
        blocks[key] = _shap_tree_block(
            X,
            dm["y"],
            names,
            seed=seed,
            max_samples=max_samples,
            plot_dir=plot_dir,
            tag=file_tag,
            model_key=key,
            clf_kind=kind,
        )
    with open(plot_dir / f"shap_summary_{file_tag}.json", "w") as f:
        json.dump(blocks, f, indent=2)
    return {"status": "ok", "plot_dir": str(plot_dir), "models": blocks}


def _strata_eval(df: pd.DataFrame, oof_prob: np.ndarray, strata_col: str, min_n: int = 30) -> list[dict]:
    rows = []
    y = df["y"].values
    for level, idx in df.groupby(strata_col).groups.items():
        idx = list(idx)
        if len(idx) < min_n:
            continue
        ys = y[idx]
        if len(set(ys)) < 2:
            continue
        ps = oof_prob[idx]
        try:
            auroc = float(roc_auc_score(ys, ps))
        except ValueError:
            continue
        prev = prevalence_matched_table(ys, ps)
        rows.append(
            {
                "stratum": str(level),
                "n": len(idx),
                "n_pos": int(ys.sum()),
                "auroc": auroc,
                "sens_prev": prev["sensitivity"],
                "spec_prev": prev["specificity"],
            }
        )
    return sorted(rows, key=lambda r: -(r["auroc"] or 0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/ch_genes_germline_vs_somatic_plus_noage.yaml")
    p.add_argument("--out-dir", default="results/ch_genes_gvs_publication")
    p.add_argument("--exclude-genes", default="", help="Comma genes to drop from all analyses (default: none)")
    p.add_argument(
        "--min-somatic-gene",
        type=int,
        default=5,
        help="Skip gene-level CV/importance when somatic count < this (pooled keeps all genes)",
    )
    p.add_argument(
        "--steps",
        default="uni",
        help="Comma steps: uni | multi | strata | importance | shap | all. "
        "multi includes gene-level sens/spec. strata needs multi (same run). "
        "shap = pooled RF TreeExplainer (beeswarm / dependence / waterfall).",
    )
    p.add_argument(
        "--shap-max-samples",
        type=int,
        default=2000,
        help="Max stratified variants to explain with SHAP (fit still uses all).",
    )
    p.add_argument(
        "--suite",
        default="all",
        choices=["all", "nonsyn", "both"],
        help="Variant set: all | nonsyn (drop synonymous) | both",
    )
    p.add_argument(
        "--classifiers",
        default="lr,rf",
        help="Comma: lr, rf, and/or xgb (xgb needs: ./run_binemo_l40.sh -m pip install xgboost)",
    )
    p.add_argument(
        "--scope",
        default="pooled,per_gene",
        help="Comma: pooled and/or per_gene (multi/strata/importance/shap)",
    )
    p.add_argument(
        "--feature-sets",
        default="baseline,api,both",
        help="Comma: baseline, api, both. For strata alone use both.",
    )
    p.add_argument(
        "--max-af",
        type=float,
        default=None,
        help="Keep variants with AF < MAX_AF or AF missing (unreported). Example: 0.01",
    )
    p.add_argument(
        "--drop-af",
        action="store_true",
        help="Exclude log10_af from baseline / API+baseline models (and skip AF univariate).",
    )
    args = p.parse_args()

    steps = {s.strip().lower() for s in args.steps.split(",") if s.strip()}
    if "all" in steps:
        steps = {"uni", "multi", "strata", "importance", "shap"}
    classifiers = tuple(
        ("xgb" if c.strip().lower() in ("xgb", "xgboost") else c.strip().lower())
        for c in args.classifiers.split(",")
        if c.strip()
    )
    scope = {s.strip().lower() for s in args.scope.split(",") if s.strip()}
    feature_sets = tuple(s.strip().lower() for s in args.feature_sets.split(",") if s.strip())
    include_af = not args.drop_af
    min_som_gene = int(args.min_somatic_gene)
    shap_max = int(args.shap_max_samples)
    suites = {"all": "all_variants", "nonsyn": "nonsynonymous_excl"}
    suite_keys = list(suites.values()) if args.suite == "both" else [suites[args.suite]]

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    out_dir = project_subpath(PROJECT, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feat_cols = _load_all(cfg)
    excl = {g.strip().upper() for g in args.exclude_genes.split(",") if g.strip()}
    if excl:
        df = df[~df["gene"].str.upper().isin(excl)].reset_index(drop=True)

    # Attach SM/CH subtype for CH-specific sensitivity (age-filtered cohort labels)
    age_dir = project_subpath(PROJECT, cfg.get("data", {}).get("age_filtered_dir", "data/processed_age_filtered"))
    sm_ch_map = _load_sm_ch_map(age_dir)
    if len(sm_ch_map):
        df = df.merge(sm_ch_map, on="variant_id", how="left")
        # no-age SM stay SM; germline stays NA
        noage = df.get("class_source") == "sm_no_age"
        df.loc[noage & df["sm_ch"].isna(), "sm_ch"] = "SM"
        n_ch = int((df["sm_ch"] == "CH").sum())
        n_sm = int((df["sm_ch"] == "SM").sum())
        print(f"SM/CH labels joined: CH={n_ch} SM={n_sm} (of somatic={int(df['y'].sum())})")
    else:
        print("WARNING: no SM/CH map found; sens_CH will be omitted")
        df["sm_ch"] = np.nan

    n_before = len(df)
    if args.max_af is not None:
        af = pd.to_numeric(df["af"], errors="coerce")
        keep = af.isna() | (af < float(args.max_af))
        df = df.loc[keep].reset_index(drop=True)
        print(
            f"AF filter: keep AF < {args.max_af} or missing → "
            f"n={len(df)}/{n_before} (dropped {n_before - len(df)}); "
            f"somatic={int(df['y'].sum())}"
        )

    n_splits = int(cfg.get("cv", {}).get("n_splits", 5))
    seed = int(cfg.get("data", {}).get("seed", 42))

    baseline_feats = ["mutation_type", "vep_consequence", "cadd_phred"]
    if include_af:
        baseline_feats.append("log10_af")
    ann = {
        "n_variants": int(len(df)),
        "n_genes": int(df["gene"].nunique()),
        "n_somatic": int(df["y"].sum()),
        "n_germline": int((df["y"] == 0).sum()),
        "n_sm_no_age": int((df.get("class_source") == "sm_no_age").sum()) if "class_source" in df.columns else None,
        "n_CH": int((df.get("sm_ch") == "CH").sum()) if "sm_ch" in df.columns else None,
        "n_SM": int((df.get("sm_ch") == "SM").sum()) if "sm_ch" in df.columns else None,
        "af_coverage": float(df["af"].notna().mean()),
        "cadd_coverage": float(df["cadd_phred"].notna().mean()),
        "max_af": args.max_af,
        "include_af_feature": include_af,
        "min_somatic_gene": min_som_gene,
        "n_before_af_filter": n_before,
        "api_features": [SHORT.get(c, c) for c in feat_cols],
        "baseline_features": baseline_feats,
        "note": (
            "AF/VEP from gnomAD v2 gene extracts; CADD from gnomAD v4.1 join in "
            "*_all_sites_cadd.parquet. Outcome binary → univariate uses logistic regression "
            "(OR per 1 SD), not linear regression."
        ),
    }
    print(
        f"Loaded n={ann['n_variants']} genes={ann['n_genes']} somatic={ann['n_somatic']} "
        f"api={len(feat_cols)} steps={sorted(steps)} suite={args.suite} clf={classifiers} "
        f"include_af={include_af} max_af={args.max_af}"
    )
    with open(out_dir / "annotation.json", "w") as f:
        json.dump(ann, f, indent=2)

    uni: list = []
    if "uni" in steps:
        y = df["y"].values
        uni = [_univariate_continuous(y, df["cadd_phred"].values.astype(float), "cadd_phred")]
        if include_af:
            uni.append(_univariate_continuous(y, df["log10_af"].values.astype(float), "log10_af"))
        for c in feat_cols:
            uni.append(_univariate_continuous(y, df[c].values.astype(float), SHORT.get(c, c)))
        uni_cat = []
        uni_cat.extend(_univariate_categorical(y, df["mutation_type"], "mutation_type"))
        uni_cat.extend(_univariate_categorical(y, df["cons_primary"], "cons_primary"))
        pd.DataFrame(uni).to_csv(out_dir / "univariate_continuous.csv", index=False)
        pd.DataFrame(uni_cat).to_csv(out_dir / "univariate_categorical.csv", index=False)
        print("Univariate continuous (AUROC raw, OR/SD):")
        for r in sorted(uni, key=lambda x: -(x.get("auroc_raw") or 0)):
            if r.get("auroc_raw") is None:
                continue
            print(f"  {r['feature']:12s} AUROC={r['auroc_raw']:.3f}  OR/SD={r['or_per_sd']:.3f}")

    def frame_for(tag: str) -> pd.DataFrame:
        if tag == "nonsynonymous_excl":
            return df[~df["vep_consequence"].map(_is_synonymous)].reset_index(drop=True)
        return df

    def run_suite(tag: str, frame: pd.DataFrame) -> dict:
        print(f"\n=== Suite: {tag} n={len(frame)} somatic={int(frame['y'].sum())} ===")
        out: dict = {"n": int(len(frame)), "n_pos": int(frame["y"].sum())}
        pooled: dict = {}

        if "multi" in steps or "strata" in steps:
            if "pooled" in scope or "strata" in steps:
                pooled = _cv_models(
                    frame,
                    feat_cols,
                    n_splits=n_splits,
                    seed=seed,
                    classifiers=classifiers,
                    feature_sets=feature_sets,
                    include_af=include_af,
                    quiet=False,
                )
                pooled_pub = {k: v for k, v in pooled.items() if k not in ("oof", "y")}
                out["pooled"] = pooled_pub
                print("Pooled models:")
                for m in pooled.get("models", []):
                    chs = m.get("sens_CH_prev")
                    chs_s = f"{chs:.3f}" if chs is not None else "NA"
                    print(
                        f"  {m['model']:28s} AUROC={m['auroc_mean']:.3f}±{m['auroc_std']:.3f}  "
                        f"sens={m['sens_prev']:.3f} top5={m.get('sens_top5', float('nan')):.3f} "
                        f"top10={m.get('sens_top10', float('nan')):.3f} "
                        f"sens_CH={chs_s} "
                        f"CH5={m.get('sens_CH_top5')} CH10={m.get('sens_CH_top10')} "
                        f"spec={m['spec_prev']:.3f}"
                    )
                with open(out_dir / f"pooled_cv_{tag}.json", "w") as f:
                    json.dump(pooled_pub, f, indent=2)

            if "multi" in steps and "per_gene" in scope:
                gene_rows = []
                for gene, gdf in frame.groupby("gene"):
                    gdf = gdf.reset_index(drop=True)
                    n_pos_g = int(gdf["y"].sum())
                    if n_pos_g < min_som_gene:
                        print(f"  {gene:8s} skipped gene-level (somatic={n_pos_g} < {min_som_gene})")
                        gene_rows.append(
                            {
                                "gene": gene,
                                "status": "too_few_somatic",
                                "n": len(gdf),
                                "n_pos": n_pos_g,
                            }
                        )
                        continue
                    res = _cv_models(
                        gdf,
                        feat_cols,
                        n_splits=n_splits,
                        seed=seed,
                        classifiers=classifiers,
                        feature_sets=feature_sets,
                        include_af=include_af,
                        quiet=True,
                    )
                    row = {
                        "gene": gene,
                        "status": res["status"],
                        "n": res.get("n"),
                        "n_pos": res.get("n_pos"),
                    }
                    if res["status"] == "ok":
                        for m in res["models"]:
                            row[f"{m['model']}_auroc"] = m["auroc_mean"]
                            row[f"{m['model']}_auroc_std"] = m["auroc_std"]
                            row[f"{m['model']}_sens"] = m["sens_prev"]
                            row[f"{m['model']}_sens_CH"] = m.get("sens_CH_prev")
                            row[f"{m['model']}_n_CH"] = m.get("n_CH")
                            row[f"{m['model']}_spec"] = m["spec_prev"]
                        best = max(res["models"], key=lambda m: m["auroc_mean"])
                        chs = best.get("sens_CH_prev")
                        chs_s = f"{chs:.3f}" if chs is not None else "NA"
                        print(
                            f"  {gene:8s} best={best['model']} {best['auroc_mean']:.3f}  "
                            f"sens={best['sens_prev']:.3f} sens_CH={chs_s} spec={best['spec_prev']:.3f}"
                        )
                    gene_rows.append(row)
                gdf_out = pd.DataFrame(gene_rows)
                gdf_out.to_csv(out_dir / f"per_gene_cv_{tag}.csv", index=False)
                # wide AUROC / sens / spec tables for publishing
                auroc_cols = ["gene", "status", "n", "n_pos"] + [
                    c for c in gdf_out.columns if c.endswith("_auroc") and not c.endswith("_auroc_std")
                ]
                sens_cols = ["gene", "status", "n", "n_pos"] + [
                    c for c in gdf_out.columns if c.endswith("_sens") or c.endswith("_sens_CH") or c.endswith("_n_CH")
                ]
                # de-dupe while preserving order
                sens_cols = list(dict.fromkeys(sens_cols))
                spec_cols = ["gene", "status", "n", "n_pos"] + [c for c in gdf_out.columns if c.endswith("_spec")]
                gdf_out[[c for c in auroc_cols if c in gdf_out.columns]].to_csv(
                    out_dir / f"per_gene_auroc_{tag}.csv", index=False
                )
                gdf_out[[c for c in sens_cols if c in gdf_out.columns]].to_csv(
                    out_dir / f"per_gene_sens_{tag}.csv", index=False
                )
                # dedicated CH sensitivity table
                ch_cols = ["gene", "status", "n", "n_pos"] + [
                    c for c in gdf_out.columns if c.endswith("_sens_CH") or c.endswith("_n_CH")
                ]
                ch_cols = list(dict.fromkeys(ch_cols))
                gdf_out[[c for c in ch_cols if c in gdf_out.columns]].to_csv(
                    out_dir / f"per_gene_sens_CH_{tag}.csv", index=False
                )
                gdf_out[[c for c in spec_cols if c in gdf_out.columns]].to_csv(
                    out_dir / f"per_gene_spec_{tag}.csv", index=False
                )
                out["per_gene"] = gene_rows

        if "strata" in steps and pooled.get("status") == "ok":
            strata = {}
            for model_name, oof_list in pooled.get("oof", {}).items():
                oof = np.array(oof_list)
                for sk in ("cons_primary", "mutation_type"):
                    rows = _strata_eval(frame, oof, sk, min_n=30)
                    for r in rows:
                        r["model"] = model_name
                    key = f"{sk}__{model_name}"
                    strata[key] = rows
                    pd.DataFrame(rows).to_csv(
                        out_dir / f"strata_{sk}_{model_name}_{tag}.csv", index=False
                    )
                if "class_source" in frame.columns:
                    rows = _strata_eval(frame, oof, "class_source", min_n=20)
                    for r in rows:
                        r["model"] = model_name
                    strata[f"class_source__{model_name}"] = rows
                    pd.DataFrame(rows).to_csv(
                        out_dir / f"strata_class_source_{model_name}_{tag}.csv", index=False
                    )
            # stacked long table for publishing
            long_rows = [r for rows in strata.values() for r in rows]
            if long_rows:
                pd.DataFrame(long_rows).to_csv(out_dir / f"strata_all_models_{tag}.csv", index=False)
            out["strata"] = strata
            print(f"Strata written for models: {sorted(pooled.get('oof', {}).keys())}")

        if "importance" in steps:
            # default: pooled if scope empty of both; respect --scope
            do_pooled = "pooled" in scope or scope == set()
            do_gene = "per_gene" in scope
            if not do_pooled and not do_gene:
                do_pooled = True
            if do_pooled:
                imp = _feature_importance(
                    frame, feat_cols, seed=seed, n_splits=n_splits, include_af=include_af
                )
                out["feature_importance"] = imp
                with open(out_dir / f"feature_importance_{tag}.json", "w") as f:
                    json.dump(imp, f, indent=2)
                print("Pooled Top3 API RF:", imp.get("api_rf_top3"))
                print("Pooled Top3 API+baseline RF:", imp.get("api_plus_baseline_rf_top3"))
                print("Pooled Top3 API XGB:", imp.get("api_xgb_top3"))
                print("Pooled Top3 API+baseline XGB:", imp.get("api_plus_baseline_xgb_top3"))
            if do_gene:
                gene_imp_rows = []
                gene_imp = {}
                for gene, gdf in frame.groupby("gene"):
                    gdf = gdf.reset_index(drop=True)
                    n_pos = int(gdf["y"].sum())
                    n_neg = len(gdf) - n_pos
                    if n_pos < min_som_gene or n_pos < n_splits or n_neg < n_splits:
                        print(f"  {gene:8s} importance skipped (n_pos={n_pos})")
                        gene_imp[gene] = {"status": "too_few", "n": len(gdf), "n_pos": n_pos}
                        continue
                    gi = _feature_importance(
                        gdf, feat_cols, seed=seed, n_splits=n_splits, include_af=include_af
                    )
                    gene_imp[gene] = {"status": "ok", "n": len(gdf), "n_pos": n_pos, **gi}
                    print(
                        f"  {gene:8s} RF API={gi.get('api_rf_top3')} both={gi.get('api_plus_baseline_rf_top3')}  "
                        f"XGB API={gi.get('api_xgb_top3')} both={gi.get('api_plus_baseline_xgb_top3')}"
                    )
                    for model_key, top_key in (
                        ("api_rf", "api_rf_top3"),
                        ("api_plus_baseline_rf", "api_plus_baseline_rf_top3"),
                        ("api_xgb", "api_xgb_top3"),
                        ("api_plus_baseline_xgb", "api_plus_baseline_xgb_top3"),
                    ):
                        for item in gi.get(top_key) or []:
                            gene_imp_rows.append(
                                {
                                    "gene": gene,
                                    "model": model_key,
                                    "rank": item["rank"],
                                    "feature": item["feature"],
                                    "importance": item["importance"],
                                    "n": len(gdf),
                                    "n_pos": n_pos,
                                }
                            )
                out["feature_importance_per_gene"] = gene_imp
                with open(out_dir / f"feature_importance_per_gene_{tag}.json", "w") as f:
                    json.dump(gene_imp, f, indent=2)
                pd.DataFrame(gene_imp_rows).to_csv(
                    out_dir / f"feature_importance_per_gene_{tag}.csv", index=False
                )

        if "shap" in steps:
            do_pooled = "pooled" in scope or scope == set()
            do_gene = "per_gene" in scope
            if not do_pooled and not do_gene:
                do_pooled = True
            print(f"SHAP (TreeExplainer RF, max_samples={shap_max}) …", flush=True)
            shap_out: dict = {}
            if do_pooled:
                shap_res = _shap_analysis(
                    frame,
                    feat_cols,
                    seed=seed,
                    include_af=include_af,
                    max_samples=shap_max,
                    out_dir=out_dir,
                    tag=tag,
                )
                shap_out["pooled"] = shap_res
                # keep top-level summary for pooled (compat)
                if shap_res.get("status") == "ok":
                    with open(out_dir / f"shap_summary_{tag}.json", "w") as f:
                        json.dump(shap_res.get("models", {}), f, indent=2)
                print(f"SHAP pooled plots -> {shap_res.get('plot_dir')}")
            if do_gene:
                gene_shap = {}
                for gene, gdf in frame.groupby("gene"):
                    gdf = gdf.reset_index(drop=True)
                    n_pos = int(gdf["y"].sum())
                    n_neg = len(gdf) - n_pos
                    if n_pos < min_som_gene or n_pos < 2 or n_neg < 2:
                        print(f"  {gene:8s} SHAP skipped (n_pos={n_pos})")
                        gene_shap[gene] = {"status": "too_few", "n": len(gdf), "n_pos": n_pos}
                        continue
                    print(f"  {gene:8s} SHAP n={len(gdf)} somatic={n_pos} …", flush=True)
                    gene_shap[gene] = _shap_analysis(
                        gdf,
                        feat_cols,
                        seed=seed,
                        include_af=include_af,
                        max_samples=min(shap_max, len(gdf)),
                        out_dir=out_dir,
                        tag=tag,
                        label=str(gene),
                    )
                shap_out["per_gene"] = gene_shap
                with open(out_dir / f"shap_summary_per_gene_{tag}.json", "w") as f:
                    json.dump(gene_shap, f, indent=2)
                print(f"SHAP per-gene plots -> {out_dir / f'shap_{tag}'}/<GENE>/")
            out["shap"] = shap_out

        return out

    suite_res = {}
    need_suite = steps & {"multi", "strata", "importance", "shap"}
    if need_suite:
        for tag in suite_keys:
            suite_res[tag] = run_suite(tag, frame_for(tag))

    payload = {
        "task": "germline_vs_somatic_publication",
        "dataset": "pilot_ch_genes_gvs_plus_noage_sm",
        "excluded_genes": sorted(excl),
        "steps": sorted(steps),
        "suite": args.suite,
        "classifiers": list(classifiers),
        "annotation": ann,
        "univariate_continuous": uni if uni else None,
        "suites": suite_res,
        "synonymous_note": (
            "Excluding VEP synonymous_variant (and synonymous&splice_region) is a common "
            "coding/NGS filter for pathogenicity analyses. Germline sets are enriched for "
            "synonymous sites, so this changes class balance and should be reported as a "
            "sensitivity analysis."
        ),
    }
    out_json = out_dir / f"summary_{'_'.join(sorted(steps))}_{args.suite}.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_json}")
    print(f"CSVs under {out_dir}")


if __name__ == "__main__":
    main()
