#!/usr/bin/env python3
"""Pooled ARCH21 NC bake-off: CADD-only vs API-only vs emb-only (matched CV).

Uses the intersection of alleles with CADD + API tracks + fold_1 embeddings so
the three arms are scored on the same variants.

Default CV: stratify_gene_func (+ optional gene onehot).
Classifiers: lr, rf, xgb (emb uses PCA64; CADD/API use raw numeric features).

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/116_cv5_pooled_cadd_api_emb_bakeoff.py \\
      --cv-mode stratify_gene_func --gene-feature onehot --pca 64
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import mutation_type_feature
from eval.classification_metrics import (
    attach_cosmic_haem_flags,
    haem_ge3_membership,
    prevalence_matched_table,
    ranking_metrics,
    source_sensitivity_pm,
)
from paths import project_subpath
from scoring.track_classifiers import make_track_classifier

_s102 = spec_from_file_location("cv5_emb", PROJECT / "scripts/102_cv5_fold1_embeddings.py")
_m102 = module_from_spec(_s102)
assert _s102.loader is not None
_s102.loader.exec_module(_m102)
_load_X = _m102._load_X

_s106 = spec_from_file_location("cv5_pooled", PROJECT / "scripts/106_cv5_pooled_arch_genes.py")
_m106 = module_from_spec(_s106)
assert _s106.loader is not None
_s106.loader.exec_module(_m106)
ARCH21 = _m106.ARCH21
_api_cols = _m106._api_cols
_load_annovar = _m106._load_annovar
_resolve_cadd = _m106._resolve_cadd
_resolve_annovar = _m106._resolve_annovar
_make_ohe = _m106._make_ohe

_s109 = spec_from_file_location("cv5_lof", PROJECT / "scripts/109_cv5_pooled_lof_emb_baseline.py")
_m109 = module_from_spec(_s109)
assert _s109.loader is not None
_s109.loader.exec_module(_m109)
_make_splits = _m109._make_splits


def _fit_predict(
    kind: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
    fold_i: int,
) -> np.ndarray:
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    spw = (n_neg / n_pos) if (kind == "xgb" and n_pos > 0) else None
    if kind == "lr":
        clf = LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"
        )
    else:
        clf = make_track_classifier(kind, random_state=seed + fold_i, scale_pos_weight=spw)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes", default=",".join(ARCH21))
    ap.add_argument("--feature-mode", default="diff", choices=("alt", "ref", "diff", "alt_diff", "concat"))
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--classifiers", default="lr,rf,xgb")
    ap.add_argument(
        "--cv-mode",
        default="stratify_gene_func",
        choices=(
            "stratify_label",
            "stratify_gene",
            "stratify_func",
            "stratify_sbs6",
            "stratify_gene_func",
            "stratify_func_sbs6",
            "stratify_gene_func_sbs6",
        ),
    )
    ap.add_argument("--gene-feature", choices=("none", "onehot"), default="onehot")
    ap.add_argument("--mutation-scheme", choices=("raw", "sbs6"), default="sbs6")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())
    rows = []
    emb_dim = None
    feat_tag = None
    api_col_sets: list[set[str]] = []

    for gene in genes:
        emb_dir = project_subpath(PROJECT, f"results/fold1_embeddings_pooled_sm/{gene}")
        if not (emb_dir / "variants_meta.parquet").is_file():
            raise SystemExit(f"Missing embeddings: {emb_dir}")
        meta, X_emb, tag = _load_X(emb_dir, args.feature_mode)
        feat_tag = tag
        emb_dim = int(X_emb.shape[1])
        meta = meta.copy()
        meta["variant_id"] = meta["variant_id"].astype(str)
        meta["gene"] = gene
        if "y" not in meta.columns:
            meta["y"] = (meta["label"].astype(str).str.lower() == "somatic").astype(int)
        if "source" not in meta.columns:
            meta["source"] = np.where(meta["y"] == 1, "cosmic_ncv", "gnomad")

        sc_path = project_subpath(
            PROJECT,
            f"results/track_scores_arch_mane_noncoding_pooled_sm_by_gene/{gene}/"
            "test_features_wide.parquet",
        )
        if not sc_path.is_file():
            raise SystemExit(f"Missing API scores: {sc_path}")
        sc = pd.read_parquet(sc_path)
        sc["variant_id"] = sc["variant_id"].astype(str)
        api_cols = _api_cols(sc)
        api_col_sets.append(set(api_cols))

        cadd_p = _resolve_cadd(gene)
        if cadd_p is None:
            raise SystemExit(f"Missing CADD for {gene}")
        cadd = pd.read_csv(cadd_p)
        cadd["variant_id"] = cadd["variant_id"].astype(str)
        cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")

        ann = _load_annovar(_resolve_annovar(gene))
        df = meta.merge(sc[["variant_id"] + api_cols], on="variant_id", how="inner")
        df = df.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="inner")
        df = df.merge(ann, on="variant_id", how="left")
        df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
        df = df.loc[df["cadd_phred"].notna()].copy()
        if "ref" not in df.columns or "alt" not in df.columns:
            raise SystemExit(f"{gene}: missing ref/alt")

        df["mutation_type"] = [
            mutation_type_feature(str(r), str(a), args.mutation_scheme)
            for r, a in zip(df["ref"], df["alt"])
        ]
        emb_map = {vid: X_emb[i] for i, vid in enumerate(meta["variant_id"].astype(str))}
        E = np.vstack([emb_map[v] for v in df["variant_id"].astype(str)])
        emb_block = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])
        df = pd.concat([df.reset_index(drop=True), emb_block], axis=1)
        df = df.drop_duplicates("variant_id")
        rows.append(df)
        print(
            f"  {gene}: n={len(df)} somatic={int(df.y.sum())} api={len(api_cols)}",
            flush=True,
        )

    assert emb_dim is not None and api_col_sets
    api_cols = sorted(set.intersection(*api_col_sets))
    if not api_cols:
        raise SystemExit("No shared API columns across genes")
    df = pd.concat(rows, ignore_index=True)
    df = attach_cosmic_haem_flags(df)
    y = df["y"].astype(int).to_numpy()
    emb_cols = [f"e{i}" for i in range(emb_dim)]

    print(
        f"Bake-off n={len(df)} genes={df['gene'].nunique()} somatic={int(y.sum())} "
        f"mode={feat_tag} pca={args.pca} cv={args.cv_mode} gene_feature={args.gene_feature}",
        flush=True,
    )

    splits = _make_splits(
        args.cv_mode,
        y,
        genes=df["gene"].astype(str).to_numpy(),
        funcs=df["Func.refGene"].astype(str).to_numpy(),
        muts=df["mutation_type"].astype(str).to_numpy(),
        n_splits=int(args.n_splits),
        seed=int(args.seed),
    )

    gene_ct = None
    if args.gene_feature == "onehot":
        gene_ct = ColumnTransformer(
            [("cat", _make_ohe(), ["gene"])],
            remainder="drop",
        )

    arms = ("cadd", "api", "emb")
    model_specs = [(f"{arm}_{k}", arm, k) for arm in arms for k in classifiers]
    oof = {m: np.full(len(df), np.nan) for m, _, _ in model_specs}
    fold_metrics: dict[str, list] = {m: [] for m, _, _ in model_specs}

    for fold_i, (tr, te) in enumerate(splits, 1):
        y_tr, y_te = y[tr], y[te]
        print(
            f"fold {fold_i}/{len(splits)} n_train={len(tr)} n_test={len(te)} "
            f"pos_te={int(y_te.sum())}",
            flush=True,
        )
        gene_tr = gene_te = None
        if gene_ct is not None:
            gene_tr = np.asarray(gene_ct.fit_transform(df.iloc[tr][["gene"]]), dtype=float)
            gene_te = np.asarray(gene_ct.transform(df.iloc[te][["gene"]]), dtype=float)

        # CADD
        cadd_pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
        )
        C_tr = cadd_pipe.fit_transform(df.iloc[tr][["cadd_phred"]])
        C_te = cadd_pipe.transform(df.iloc[te][["cadd_phred"]])
        if gene_tr is not None:
            C_tr = np.hstack([C_tr, gene_tr])
            C_te = np.hstack([C_te, gene_te])

        # API
        api_pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
        )
        A_tr = api_pipe.fit_transform(df.iloc[tr][api_cols])
        A_te = api_pipe.transform(df.iloc[te][api_cols])
        if gene_tr is not None:
            A_tr = np.hstack([A_tr, gene_tr])
            A_te = np.hstack([A_te, gene_te])

        # emb PCA
        E_tr = np.asarray(df.iloc[tr][emb_cols], dtype=np.float64)
        E_te = np.asarray(df.iloc[te][emb_cols], dtype=np.float64)
        n_comp = min(int(args.pca), len(tr) - 1, E_tr.shape[1]) if args.pca > 0 else 0
        if n_comp > 0:
            emb_pipe = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("pca", PCA(n_components=n_comp, random_state=args.seed + fold_i)),
                ]
            )
        else:
            emb_pipe = Pipeline(
                [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
            )
        Z_tr = emb_pipe.fit_transform(E_tr)
        Z_te = emb_pipe.transform(E_te)
        if gene_tr is not None:
            Z_tr = np.hstack([np.asarray(Z_tr), gene_tr])
            Z_te = np.hstack([np.asarray(Z_te), gene_te])

        mats = {"cadd": (C_tr, C_te), "api": (A_tr, A_te), "emb": (Z_tr, Z_te)}
        for name, arm, kind in model_specs:
            Xtr, Xte = mats[arm]
            p = _fit_predict(kind, Xtr, y_tr, Xte, seed=args.seed, fold_i=fold_i)
            oof[name][te] = p
            fold_metrics[name].append(ranking_metrics(y_te, p)["auroc"])

    out = project_subpath(
        PROJECT,
        args.out_dir
        or (
            f"results/pooled_arch21_nc_bakeoff_cadd_api_emb_"
            f"{feat_tag}_pca{args.pca}_{args.cv_mode}_genefeat_{args.gene_feature}"
        ),
    )
    out.mkdir(parents=True, exist_ok=True)

    pred = df[
        ["variant_id", "gene", "label", "source", "y", "Func.refGene", "cadd_phred"]
    ].copy()
    for m, _, _ in model_specs:
        pred[f"p_{m}"] = oof[m]
    pred.to_parquet(out / "oof_predictions.parquet", index=False)

    rows_sum = []
    rows_gene = []
    for name, arm, kind in model_specs:
        p = oof[name]
        m = ranking_metrics(y, p)
        pm = prevalence_matched_table(y, p)
        fold_au = np.asarray(fold_metrics[name], dtype=float)
        rows_sum.append(
            {
                "model": name,
                "feature_set": arm,
                "classifier": kind,
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "n": len(df),
                "pm_sensitivity": pm.get("sensitivity"),
                "pm_specificity": pm.get("specificity"),
                "fold_auroc_mean": float(np.nanmean(fold_au)),
                "fold_auroc_sd": float(np.nanstd(fold_au, ddof=1)) if len(fold_au) > 1 else 0.0,
                "cv_mode": args.cv_mode,
                "gene_feature": args.gene_feature,
            }
        )
        print(
            f"{name}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
            f"PM_sens={pm.get('sensitivity')}",
            flush=True,
        )
        for g, sub in pred.groupby("gene"):
            yg = sub["y"].to_numpy()
            if yg.min() == yg.max():
                continue
            mg = ranking_metrics(yg, sub[f"p_{name}"].to_numpy())
            pmg = prevalence_matched_table(yg, sub[f"p_{name}"].to_numpy())
            rows_gene.append(
                {
                    "model": name,
                    "feature_set": arm,
                    "classifier": kind,
                    "gene": g,
                    "n": len(sub),
                    "n_somatic": int(yg.sum()),
                    "auroc": mg["auroc"],
                    "auprc": mg["auprc"],
                    "pm_sensitivity": pmg.get("sensitivity"),
                }
            )

    summary = pd.DataFrame(rows_sum)
    summary.to_csv(out / "oof_summary.csv", index=False)
    pd.DataFrame(rows_gene).to_csv(out / "per_gene_oof.csv", index=False)

    # source sens for each xgb arm
    src_rows = []
    for arm in arms:
        name = f"{arm}_xgb"
        if name not in oof:
            continue
        for r in source_sensitivity_pm(
            y, oof[name], df["source"].values, extra_membership=haem_ge3_membership(df)
        ):
            src_rows.append({"model": name, **r})
    if src_rows:
        pd.DataFrame(src_rows).to_csv(out / "source_sensitivity_pm.csv", index=False)

    # compact bake-off table (best AUROC per arm + all xgb)
    bake = summary.sort_values(["feature_set", "auroc"], ascending=[True, False])
    bake.to_csv(out / "bakeoff_by_arm.csv", index=False)

    meta = {
        "genes": genes,
        "n": int(len(df)),
        "n_somatic": int(y.sum()),
        "n_api": len(api_cols),
        "emb_dim": emb_dim,
        "feature_mode": feat_tag,
        "pca": int(args.pca),
        "cv_mode": args.cv_mode,
        "gene_feature": args.gene_feature,
        "arms": list(arms),
        "classifiers": list(classifiers),
        "note": "Intersection cohort with CADD+API+emb; Func/SBS6 NOT in features",
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
