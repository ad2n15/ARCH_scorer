#!/usr/bin/env python3
"""Pooled ARCH21 coding∪noncoding bake-off: CADD vs API vs emb (matched CV).

Same design as scripts/116 (NC-only): intersection of alleles with CADD +
API (CD34+ CMP) + fold_1 emb DIFF; Func/SBS6 not in features.
Reports overall + per-region (coding / noncoding) AUROC and prevalence-matched
sens/spec.

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/142_cv5_pooled_cnc_cadd_api_emb_bakeoff.py \\
      --cv-mode stratify_gene_func --gene-feature onehot --pca 64 --classifiers lr
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
    prevalence_matched_table,
    ranking_metrics,
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
_make_ohe = _m106._make_ohe

_s109 = spec_from_file_location("cv5_lof", PROJECT / "scripts/109_cv5_pooled_lof_emb_baseline.py")
_m109 = module_from_spec(_s109)
assert _s109.loader is not None
_s109.loader.exec_module(_m109)
_make_splits = _m109._make_splits

_s112 = spec_from_file_location("cv5_tp53_cnc", PROJECT / "scripts/112_cv5_tp53_coding_noncoding.py")
_m112 = module_from_spec(_s112)
assert _s112.loader is not None
_s112.loader.exec_module(_m112)
_load_arm = _m112._load_arm


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


def _slice_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = ranking_metrics(y, p)
    pm = prevalence_matched_table(y, p)
    return {
        "n": int(len(y)),
        "n_somatic": int(y.sum()),
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "pm_sensitivity": pm.get("sensitivity"),
        "pm_specificity": pm.get("specificity"),
    }


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
        for kind, cohort, scores, emb_dir in (
            (
                "noncoding",
                project_subpath(
                    PROJECT, f"data/arch_mane_noncoding_pooled_sm/by_gene/{gene}.csv"
                ),
                project_subpath(
                    PROJECT,
                    f"results/track_scores_arch_mane_noncoding_pooled_sm_by_gene/{gene}/"
                    "test_features_wide.parquet",
                ),
                project_subpath(PROJECT, f"results/fold1_embeddings_pooled_sm/{gene}"),
            ),
            (
                "coding",
                project_subpath(
                    PROJECT, f"data/arch_mane_coding_pooled_sm/by_gene/{gene}.csv"
                ),
                project_subpath(
                    PROJECT,
                    f"results/track_scores_arch_mane_coding_pooled_sm_by_gene/{gene}/"
                    "test_features_wide.parquet",
                ),
                project_subpath(
                    PROJECT, f"results/fold1_embeddings_coding_pooled_sm/{gene}"
                ),
            ),
        ):
            if not cohort.is_file():
                print(f"  skip {gene}/{kind}: missing cohort", flush=True)
                continue
            df, X_emb, api_feats = _load_arm(
                gene=gene,
                kind=kind,
                cohort=cohort,
                scores=scores,
                emb_dir=emb_dir,
                feature_mode=args.feature_mode,
            )
            # _load_arm already attached emb alignment via return X; attach columns
            emb_block = pd.DataFrame(
                X_emb, columns=[f"e{i}" for i in range(X_emb.shape[1])]
            )
            df = pd.concat([df.reset_index(drop=True), emb_block], axis=1)
            # recompute mutation_type already set; ensure scheme consistency
            if "mutation_type" not in df.columns:
                df["mutation_type"] = [
                    mutation_type_feature(str(r), str(a), "sbs6")
                    for r, a in zip(df["ref"], df["alt"])
                ]
            emb_dim = int(X_emb.shape[1])
            feat_tag = f"emb_{args.feature_mode}"
            api_col_sets.append(set(api_feats))
            rows.append(df)
            print(
                f"  {gene}/{kind}: n={len(df)} somatic={int(df.y.sum())}",
                flush=True,
            )

    assert emb_dim is not None and api_col_sets
    api_cols = sorted(set.intersection(*api_col_sets))
    if not api_cols:
        raise SystemExit("No shared API columns across genes/regions")
    df = pd.concat(rows, ignore_index=True)
    df = attach_cosmic_haem_flags(df)
    y = df["y"].astype(int).to_numpy()
    emb_cols = [f"e{i}" for i in range(emb_dim)]

    print(
        f"CNC bake-off n={len(df)} coding={(df.region_class=='coding').sum()} "
        f"nc={(df.region_class=='noncoding').sum()} somatic={int(y.sum())} "
        f"pca={args.pca} cv={args.cv_mode} gene_feature={args.gene_feature}",
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

    for fold_i, (tr, te) in enumerate(splits, 1):
        y_tr = y[tr]
        print(
            f"fold {fold_i}/{len(splits)} n_train={len(tr)} n_test={len(te)} "
            f"pos_te={int(y[te].sum())}",
            flush=True,
        )
        gene_tr = gene_te = None
        if gene_ct is not None:
            gene_tr = np.asarray(gene_ct.fit_transform(df.iloc[tr][["gene"]]), dtype=float)
            gene_te = np.asarray(gene_ct.transform(df.iloc[te][["gene"]]), dtype=float)

        cadd_pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
        )
        C_tr = cadd_pipe.fit_transform(df.iloc[tr][["cadd_phred"]])
        C_te = cadd_pipe.transform(df.iloc[te][["cadd_phred"]])

        api_pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
        )
        A_tr = api_pipe.fit_transform(df.iloc[tr][api_cols])
        A_te = api_pipe.transform(df.iloc[te][api_cols])

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
            C_tr = np.hstack([C_tr, gene_tr])
            C_te = np.hstack([C_te, gene_te])
            A_tr = np.hstack([A_tr, gene_tr])
            A_te = np.hstack([A_te, gene_te])
            Z_tr = np.hstack([np.asarray(Z_tr), gene_tr])
            Z_te = np.hstack([np.asarray(Z_te), gene_te])

        mats = {"cadd": (C_tr, C_te), "api": (A_tr, A_te), "emb": (Z_tr, Z_te)}
        for name, arm, kind in model_specs:
            Xtr, Xte = mats[arm]
            oof[name][te] = _fit_predict(
                kind, Xtr, y_tr, Xte, seed=args.seed, fold_i=fold_i
            )

    out = project_subpath(
        PROJECT,
        args.out_dir
        or (
            f"results/pooled_arch21_cnc_bakeoff_cadd_api_emb_"
            f"{feat_tag}_pca{args.pca}_{args.cv_mode}_genefeat_{args.gene_feature}"
        ),
    )
    out.mkdir(parents=True, exist_ok=True)

    pred = df[
        [
            "variant_id",
            "gene",
            "region_class",
            "label",
            "source",
            "y",
            "Func.refGene",
            "cadd_phred",
        ]
    ].copy()
    for m, _, _ in model_specs:
        pred[f"p_{m}"] = oof[m]
    pred.to_parquet(out / "oof_predictions.parquet", index=False)

    rows_sum = []
    rows_region = []
    for name, arm, kind in model_specs:
        p = oof[name]
        overall = _slice_metrics(y, p)
        rows_sum.append(
            {
                "model": name,
                "feature_set": arm,
                "classifier": kind,
                "region": "all",
                **overall,
                "cv_mode": args.cv_mode,
                "gene_feature": args.gene_feature,
            }
        )
        print(
            f"{name} all: AUROC={overall['auroc']:.4f} "
            f"PM_sens={overall['pm_sensitivity']:.4f} "
            f"PM_spec={overall['pm_specificity']:.4f}",
            flush=True,
        )
        for rc in ("coding", "noncoding"):
            mask = df["region_class"].to_numpy() == rc
            if mask.sum() == 0 or y[mask].min() == y[mask].max():
                continue
            sm = _slice_metrics(y[mask], p[mask])
            rows_region.append(
                {
                    "model": name,
                    "feature_set": arm,
                    "classifier": kind,
                    "region": rc,
                    **sm,
                    "cv_mode": args.cv_mode,
                    "gene_feature": args.gene_feature,
                }
            )
            print(
                f"{name} {rc}: AUROC={sm['auroc']:.4f} "
                f"PM_sens={sm['pm_sensitivity']:.4f} "
                f"PM_spec={sm['pm_specificity']:.4f}",
                flush=True,
            )

    summary = pd.DataFrame(rows_sum)
    region = pd.DataFrame(rows_region)
    summary.to_csv(out / "oof_summary.csv", index=False)
    region.to_csv(out / "oof_by_region.csv", index=False)
    pd.concat([summary, region], ignore_index=True).to_csv(
        out / "bakeoff_overall_and_region.csv", index=False
    )

    # LR-focused publication table
    lr = region[region["classifier"] == "lr"].copy()
    if len(lr):
        pub = lr.pivot_table(
            index=["feature_set"],
            columns="region",
            values=["auroc", "pm_sensitivity", "pm_specificity"],
            aggfunc="first",
        )
        pub.columns = [f"{a}_{b}" for a, b in pub.columns]
        pub = pub.reset_index()
        pub.to_csv(out / "lr_by_region_publication.csv", index=False)

    meta = {
        "genes": genes,
        "n": int(len(df)),
        "n_coding": int((df.region_class == "coding").sum()),
        "n_noncoding": int((df.region_class == "noncoding").sum()),
        "n_somatic": int(y.sum()),
        "n_api": len(api_cols),
        "emb_dim": emb_dim,
        "feature_mode": args.feature_mode,
        "pca": int(args.pca),
        "cv_mode": args.cv_mode,
        "gene_feature": args.gene_feature,
        "arms": list(arms),
        "classifiers": list(classifiers),
        "note": (
            "CNC intersection CADD+API+emb; Func/SBS6 NOT in features; "
            "PM sens/spec = prevalence-matched"
        ),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
