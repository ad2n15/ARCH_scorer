#!/usr/bin/env python3
"""Fit full-cohort models for ARCH_scorer (not 5-fold OOF).

Exports:
  a) emb + baseline XGB  (+ gene one-hot) — production
  b) api + baseline XGB  (+ gene one-hot) — production, no local emb
  c) api-only RF — interpretability (per-variant SHAP ranks of 10 tracks)

Thresholds come from published 5-fold OOF pm_threshold (honest operating point),
while classifiers / transforms are fit on ALL training alleles.

Run (binemo env with xgboost):
  ./run_binemo_AMD.sh ARCH_scorer/scripts/export_models_from_sm_ch_finetune.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SM = ROOT / "sm_ch_finetune"
sys.path.insert(0, str(SM / "src"))
sys.path.insert(0, str(SM / "scripts"))

from data.variant_table import mutation_type_feature  # noqa: E402
from scoring.track_classifiers import make_track_classifier  # noqa: E402

import importlib.util


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_m106 = _load_mod("cv106", SM / "scripts" / "106_cv5_pooled_arch_genes.py")
_m102 = _load_mod("cv102", SM / "scripts" / "102_cv5_fold1_embeddings.py")

ARCH21 = [
    "ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1", "IDH2",
    "JAK2", "KIT", "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1", "SETBP1", "SF3B1",
    "SRSF2", "TET2", "TP53",
]
SCORES_ROOT = SM / "results" / "track_scores_arch_mane_noncoding_pooled_sm_by_gene"
EMB_OOF = (
    SM / "results"
    / "pooled_arch21_cv5_fold1_emb_baseline_emb_diff_pca64_stratify_gene_func_genefeat_onehot"
)
API_OOF = SM / "results" / "pooled_arch21_cv5_stratify_gene_func_genefeat_onehot"
OUT_DIR = ROOT / "ARCH_scorer" / "artifacts"
SEED = 42
PCA_K = 64

API_PREFIXES = (
    "RNA_SEQ__",
    "SPLICE_SITE_USAGE__",
    "SPLICE_JUNCTIONS__",
    "DNASE__",
    "CHIP_HISTONE__",
)


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _oof_thr(summary_csv: Path, model: str) -> dict:
    s = pd.read_csv(summary_csv)
    row = s.loc[s["model"] == model].iloc[0]
    return {
        "pm_threshold": float(row["pm_threshold"]),
        "oof_auroc": float(row["auroc"]),
        "oof_pm_sensitivity": float(row["pm_sensitivity"]),
        "oof_pm_specificity": float(row["pm_specificity"]),
    }


def load_emb_pooled() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows = []
    emb_dim = None
    for gene in ARCH21:
        emb_dir = SM / "results" / "fold1_embeddings_pooled_sm" / gene
        meta, X_emb, _ = _m102._load_X(emb_dir, "diff")
        emb_dim = int(X_emb.shape[1])
        meta = meta.copy()
        meta["variant_id"] = meta["variant_id"].astype(str)
        meta["gene"] = gene
        if "y" not in meta.columns:
            meta["y"] = (meta["label"].astype(str).str.lower() == "somatic").astype(int)
        cadd = pd.read_csv(_m106._resolve_cadd(gene))
        cadd["variant_id"] = cadd["variant_id"].astype(str)
        cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
        df = meta.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")
        ann_p = _m106._resolve_annovar(gene)
        if Path(ann_p).is_file():
            df = df.merge(_m106._load_annovar(ann_p), on="variant_id", how="left")
            df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
        else:
            df["Func.refGene"] = "unknown"
        df["mutation_type"] = [
            mutation_type_feature(str(r), str(a), "sbs6")
            for r, a in zip(df["ref"], df["alt"])
        ]
        emb_block = pd.DataFrame(
            X_emb, columns=[f"e{i}" for i in range(emb_dim)], index=df.index
        )
        df = pd.concat([df, emb_block], axis=1)
        df = df.loc[df["cadd_phred"].notna()].drop_duplicates("variant_id")
        rows.append(df)
        print(f"  emb {gene}: n={len(df)} som={int(df.y.sum())}")
    df = pd.concat(rows, ignore_index=True)
    return df, df["y"].astype(int).to_numpy(), [f"e{i}" for i in range(emb_dim)]


def load_api_pooled() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows = []
    api_feats: list[str] | None = None
    for gene in ARCH21:
        cohort = SM / "data" / "arch_mane_noncoding_pooled_sm" / "by_gene" / f"{gene}.csv"
        df = pd.read_csv(cohort)
        df["variant_id"] = df["variant_id"].astype(str)
        df["gene"] = gene
        df["y"] = (df["label"].astype(str).str.lower() == "somatic").astype(int)
        sc = pd.read_parquet(SCORES_ROOT / gene / "test_features_wide.parquet")
        sc["variant_id"] = sc["variant_id"].astype(str)
        feats = [c for c in sc.columns if any(c.startswith(p) for p in API_PREFIXES)]
        if api_feats is None:
            api_feats = sorted(feats)
        df = df.merge(sc[["variant_id"] + api_feats], on="variant_id", how="inner")
        df = df.merge(_m106._load_annovar(_m106._resolve_annovar(gene)), on="variant_id", how="left")
        df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
        cadd = pd.read_csv(_m106._resolve_cadd(gene))
        cadd["variant_id"] = cadd["variant_id"].astype(str)
        cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
        df = df.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")
        if "ref" in df.columns and "alt" in df.columns:
            df["mutation_type"] = [
                mutation_type_feature(str(r), str(a), "sbs6")
                for r, a in zip(df["ref"], df["alt"])
            ]
        else:
            parts = df["variant_id"].str.split("-", n=3, expand=True)
            df["mutation_type"] = [
                mutation_type_feature(str(r), str(a), "sbs6")
                for r, a in zip(parts[2], parts[3])
            ]
        df = df.loc[df["cadd_phred"].notna()].drop_duplicates("variant_id")
        rows.append(df)
        print(f"  api {gene}: n={len(df)} som={int(df.y.sum())}")
    df = pd.concat(rows, ignore_index=True)
    assert api_feats is not None
    return df, df["y"].astype(int).to_numpy(), api_feats


def fit_emb(df: pd.DataFrame, y: np.ndarray, emb_cols: list[str], oof: dict) -> Path:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    emb_pca = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=PCA_K, random_state=SEED)),
        ]
    )
    Z_emb = emb_pca.fit_transform(np.asarray(df[emb_cols], dtype=np.float64))
    base_cats = ["mutation_type", "Func.refGene", "gene"]
    base_ct = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
            ("cat", _make_ohe(), base_cats),
        ],
        remainder="drop",
    )
    Z_base = np.asarray(base_ct.fit_transform(df[["cadd_phred"] + base_cats]), dtype=np.float64)
    Z = np.hstack([Z_emb, Z_base])
    clf = make_track_classifier("xgb", random_state=SEED, scale_pos_weight=n_neg / max(n_pos, 1))
    clf.fit(Z, y)
    print(f"emb+baseline XGB fit Z={Z.shape}")
    art = {
        "model_name": "emb_baseline_xgb",
        "feature_set": "emb_baseline",
        "emb_mode": "diff",
        "emb_dim": len(emb_cols),
        "pca_k": PCA_K,
        "gene_feature": "onehot",
        "baseline_cats": base_cats,
        "api_feats": None,
        "emb_pca": emb_pca,
        "api_imp": None,
        "base_ct": base_ct,
        "clf": clf,
        "pm_method": "prevalence_matched_top_k_from_oof",
        "genes": ARCH21,
        "n_train": int(len(df)),
        "n_somatic": n_pos,
        "seed": SEED,
        **oof,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "emb_baseline_xgb_gene_onehot.joblib"
    art["clf"].save_model(OUT_DIR / "emb_baseline_xgb_gene_onehot.xgb.json")  # version-independent booster
    art["clf"] = "emb_baseline_xgb_gene_onehot.xgb.json"
    joblib.dump(art, path)
    meta = {k: v for k, v in art.items() if k not in ("emb_pca", "api_imp", "base_ct", "clf")}
    (OUT_DIR / "emb_baseline_xgb_gene_onehot.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {path}")
    return path


def fit_api(df: pd.DataFrame, y: np.ndarray, api_feats: list[str], oof: dict) -> Path:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    api_imp = Pipeline([("imp", SimpleImputer(strategy="median"))])
    Z_api = np.asarray(api_imp.fit_transform(df[api_feats]), dtype=np.float64)
    base_cats = ["mutation_type", "Func.refGene", "gene"]
    base_ct = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
            ("cat", _make_ohe(), base_cats),
        ],
        remainder="drop",
    )
    Z_base = np.asarray(base_ct.fit_transform(df[["cadd_phred"] + base_cats]), dtype=np.float64)
    Z = np.hstack([Z_api, Z_base])
    clf = make_track_classifier("xgb", random_state=SEED, scale_pos_weight=n_neg / max(n_pos, 1))
    clf.fit(Z, y)
    print(f"api+baseline XGB fit Z={Z.shape}")
    art = {
        "model_name": "api_baseline_xgb",
        "feature_set": "api_baseline",
        "emb_mode": None,
        "emb_dim": None,
        "pca_k": None,
        "gene_feature": "onehot",
        "baseline_cats": base_cats,
        "api_feats": api_feats,
        "emb_pca": None,
        "api_imp": api_imp,
        "base_ct": base_ct,
        "clf": clf,
        "pm_method": "prevalence_matched_top_k_from_oof",
        "genes": ARCH21,
        "n_train": int(len(df)),
        "n_somatic": n_pos,
        "seed": SEED,
        **oof,
    }
    path = OUT_DIR / "api_baseline_xgb_gene_onehot.joblib"
    art["clf"].save_model(OUT_DIR / "api_baseline_xgb_gene_onehot.xgb.json")  # version-independent booster
    art["clf"] = "api_baseline_xgb_gene_onehot.xgb.json"
    joblib.dump(art, path)
    meta = {k: v for k, v in art.items() if k not in ("emb_pca", "api_imp", "base_ct", "clf")}
    (OUT_DIR / "api_baseline_xgb_gene_onehot.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {path}")
    return path


def fit_api_only_rf(df: pd.DataFrame, y: np.ndarray, api_feats: list[str], oof: dict) -> Path:
    """API-track-only RF for per-variant SHAP ranking (interpretability)."""
    n_pos = int(y.sum())
    api_imp = Pipeline([("imp", SimpleImputer(strategy="median"))])
    Z = np.asarray(api_imp.fit_transform(df[api_feats]), dtype=np.float64)
    clf = make_track_classifier("rf", random_state=SEED)
    clf.fit(Z, y)
    print(f"api-only RF fit Z={Z.shape}")
    art = {
        "model_name": "api_only_rf",
        "feature_set": "api",
        "emb_mode": None,
        "emb_dim": None,
        "pca_k": None,
        "gene_feature": "none",
        "baseline_cats": None,
        "api_feats": api_feats,
        "emb_pca": None,
        "api_imp": api_imp,
        "base_ct": None,
        "clf": clf,
        "pm_method": "prevalence_matched_top_k_from_oof",
        "genes": ARCH21,
        "n_train": int(len(df)),
        "n_somatic": n_pos,
        "seed": SEED,
        **oof,
    }
    path = OUT_DIR / "api_only_rf.joblib"
    joblib.dump(art, path)
    meta = {k: v for k, v in art.items() if k not in ("emb_pca", "api_imp", "base_ct", "clf")}
    (OUT_DIR / "api_only_rf.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {path}")
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        choices=("all", "emb_baseline", "api_baseline", "api_only"),
        default="all",
    )
    args = ap.parse_args()

    need_emb = args.only in ("all", "emb_baseline")
    need_api = args.only in ("all", "api_baseline", "api_only")

    df_a = y_a = api_feats = None
    if need_api:
        print("=== load api pooled ===")
        df_a, y_a, api_feats = load_api_pooled()

    if need_emb:
        emb_oof = _oof_thr(EMB_OOF / "oof_summary.csv", "emb_base_xgb")
        print("=== emb+baseline (full fit) ===")
        print(f"OOF thr={emb_oof['pm_threshold']:.6f} auroc={emb_oof['oof_auroc']:.4f}")
        df_e, y_e, emb_cols = load_emb_pooled()
        fit_emb(df_e, y_e, emb_cols, emb_oof)

    if args.only in ("all", "api_baseline"):
        api_oof = _oof_thr(API_OOF / "oof_summary.csv", "api_baseline_xgb")
        print("=== api+baseline (full fit) ===")
        print(f"OOF thr={api_oof['pm_threshold']:.6f} auroc={api_oof['oof_auroc']:.4f}")
        assert df_a is not None and api_feats is not None
        fit_api(df_a, y_a, api_feats, api_oof)

    if args.only in ("all", "api_only"):
        api_rf_oof = _oof_thr(API_OOF / "oof_summary.csv", "api_rf")
        print("=== api-only RF (interpretability) ===")
        print(f"OOF thr={api_rf_oof['pm_threshold']:.6f} auroc={api_rf_oof['oof_auroc']:.4f}")
        assert df_a is not None and api_feats is not None
        fit_api_only_rf(df_a, y_a, api_feats, api_rf_oof)


if __name__ == "__main__":
    main()
