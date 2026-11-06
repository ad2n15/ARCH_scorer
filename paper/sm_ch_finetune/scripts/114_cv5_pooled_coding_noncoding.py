#!/usr/bin/env python3
"""Multi-gene pooled 5-fold CV on coding ∪ noncoding (ARCH21 default).

Based on scripts/112_cv5_tp53_coding_noncoding.py; uses gene×Func splits from 109.

Feature sets (default five; skips api_emb_baseline for speed):
  baseline | api | emb | api_baseline | emb_baseline

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/114_cv5_pooled_coding_noncoding.py \\
      --cv-mode stratify_gene_func --gene-feature onehot --feature-mode diff --pca 64
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from eval.classification_metrics import (
    attach_cosmic_haem_flags,
    haem_ge3_membership,
    prevalence_matched_table,
    ranking_metrics,
    source_sensitivity_pm,
)
from paths import project_subpath
from scoring.track_classifiers import make_track_classifier

_s106 = spec_from_file_location("cv5_pooled", PROJECT / "scripts" / "106_cv5_pooled_arch_genes.py")
_m106 = module_from_spec(_s106)
assert _s106.loader is not None
_s106.loader.exec_module(_m106)
ARCH21 = _m106.ARCH21
_make_ohe = _m106._make_ohe

_s109 = spec_from_file_location(
    "cv5_lof_emb", PROJECT / "scripts" / "109_cv5_pooled_lof_emb_baseline.py"
)
_m109 = module_from_spec(_s109)
assert _s109.loader is not None
_s109.loader.exec_module(_m109)
_make_splits = _m109._make_splits

_s112 = spec_from_file_location(
    "cv5_tp53_cnc", PROJECT / "scripts" / "112_cv5_tp53_coding_noncoding.py"
)
_m112 = module_from_spec(_s112)
assert _s112.loader is not None
_s112.loader.exec_module(_m112)
_load_arm = _m112._load_arm
_fit_predict = _m112._fit_predict


def _cohort_n(kind: str, gene: str) -> int | None:
    if kind == "coding":
        p = project_subpath(PROJECT, f"data/arch_mane_coding_pooled_sm/by_gene/{gene}.csv")
    else:
        p = project_subpath(
            PROJECT, f"data/arch_mane_noncoding_pooled_sm/by_gene/{gene}.csv"
        )
    if not p.is_file():
        return None
    return len(pd.read_csv(p))


def _api_n(kind: str, gene: str) -> int | None:
    if kind == "coding":
        p = project_subpath(
            PROJECT,
            f"results/track_scores_arch_mane_coding_pooled_sm_by_gene/{gene}/"
            "test_features_wide.parquet",
        )
    else:
        p = project_subpath(
            PROJECT,
            f"results/track_scores_arch_mane_noncoding_pooled_sm_by_gene/{gene}/"
            "test_features_wide.parquet",
        )
    if not p.is_file():
        return None
    return len(pd.read_parquet(p, columns=["variant_id"]))


def _check_api_complete(genes: list[str]) -> None:
    gaps: list[str] = []
    for gene in genes:
        for kind in ("coding", "noncoding"):
            n_coh = _cohort_n(kind, gene)
            n_api = _api_n(kind, gene)
            if n_coh is None:
                gaps.append(f"{gene}/{kind}: missing cohort CSV")
            elif n_api is None:
                gaps.append(f"{gene}/{kind}: missing API scores (cohort n={n_coh})")
            elif n_api != n_coh:
                gaps.append(
                    f"{gene}/{kind}: API incomplete {n_api}/{n_coh}"
                )
    if gaps:
        msg = "Incomplete coding/noncoding API scores (--require-api):\n  " + "\n  ".join(
            gaps
        )
        raise SystemExit(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes", default=",".join(ARCH21))
    ap.add_argument(
        "--feature-sets",
        default="baseline,api,emb,api_baseline,emb_baseline",
        help="Comma list; default skips api_emb_baseline",
    )
    ap.add_argument("--classifiers", default="lr,rf,xgb")
    ap.add_argument(
        "--feature-mode", default="diff", choices=("alt", "ref", "diff", "alt_diff")
    )
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=500)
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
    ap.add_argument(
        "--require-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit if any gene lacks complete coding or noncoding API (default: true)",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    feature_sets = tuple(x.strip() for x in args.feature_sets.split(",") if x.strip())
    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())

    if args.require_api:
        _check_api_complete(genes)

    dfs, Xs, api_feats = [], [], None
    for gene in genes:
        arms = [
            (
                "noncoding",
                project_subpath(
                    PROJECT, f"data/arch_mane_noncoding_pooled_sm/by_gene/{gene}.csv"
                ),
                project_subpath(
                    PROJECT,
                    f"results/track_scores_arch_mane_noncoding_pooled_sm_by_gene/"
                    f"{gene}/test_features_wide.parquet",
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
                    f"results/track_scores_arch_mane_coding_pooled_sm_by_gene/"
                    f"{gene}/test_features_wide.parquet",
                ),
                project_subpath(
                    PROJECT, f"results/fold1_embeddings_coding_pooled_sm/{gene}"
                ),
            ),
        ]
        for kind, cohort, scores, emb_dir in arms:
            df_a, X_a, api_a = _load_arm(
                gene=gene,
                kind=kind,
                cohort=cohort,
                scores=scores,
                emb_dir=emb_dir,
                feature_mode=args.feature_mode,
            )
            if api_feats is None:
                api_feats = api_a
            else:
                api_feats = [c for c in api_feats if c in api_a]
            dfs.append(df_a)
            Xs.append(X_a)

    df = pd.concat(dfs, ignore_index=True)
    X_emb = np.vstack(Xs)
    api_feats = [c for c in (api_feats or []) if c in df.columns]
    emb_cols = [f"e{i}" for i in range(X_emb.shape[1])]
    emb_block = pd.DataFrame(X_emb, columns=emb_cols, index=df.index)
    df = pd.concat([df, emb_block], axis=1)

    df = attach_cosmic_haem_flags(df)
    y = df["y"].astype(int).values
    print(
        f"Pooled coding∪noncoding: n={len(df)} genes={df['gene'].nunique()} "
        f"germline={(y == 0).sum()} somatic={int(y.sum())} api={len(api_feats)} "
        f"emb={X_emb.shape[1]} cv={args.cv_mode} gene_feature={args.gene_feature}"
    )
    print(
        df.groupby(["region_class"])["y"]
        .agg(["size", "sum"])
        .rename(columns={"sum": "n_som"})
        .to_string()
    )

    splits = _make_splits(
        args.cv_mode,
        y,
        genes=df["gene"].astype(str).values,
        funcs=df["Func.refGene"].astype(str).values,
        muts=df["mutation_type"].astype(str).values,
        n_splits=int(args.n_splits),
        seed=int(args.seed),
    )
    print(f"n_folds={len(splits)}")

    pca_k = int(args.pca)
    n_pos_all = max(int(y.sum()), 1)
    n_neg_all = max(int(len(y) - n_pos_all), 1)
    w_pos = 0.5 * len(y) / n_pos_all
    w_neg = 0.5 * len(y) / n_neg_all

    base_cats = ["mutation_type", "Func.refGene"]
    if args.gene_feature == "onehot":
        base_cats = base_cats + ["gene"]
    base_ct = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
            ("cat", _make_ohe(), base_cats),
        ]
    )
    gene_ct = None
    if args.gene_feature == "onehot":
        gene_ct = ColumnTransformer([("cat", _make_ohe(), ["gene"])], remainder="drop")
    api_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    )

    out_dir = project_subpath(
        PROJECT,
        args.out_dir
        or (
            f"results/pooled_arch{len(genes)}_coding_noncoding_cv5_"
            f"{args.cv_mode}_emb_{args.feature_mode}_pca{pca_k}_"
            f"genefeat_{args.gene_feature}"
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = df[
        [
            c
            for c in [
                "variant_id",
                "gene",
                "region_class",
                "label",
                "source",
                "y",
                "n_haem",
                "haem_ge3",
                "Func.refGene",
                "mutation_type",
                "cadd_phred",
            ]
            if c in df.columns
        ]
    ].copy()
    summary_rows, src_rows, region_rows, per_gene_rows = [], [], [], []

    for fs in feature_sets:
        use_mlp = "emb" in fs
        clfs = classifiers if use_mlp else tuple(c for c in classifiers if c != "mlp")
        for kind in clfs:
            name = f"{fs}_{kind}"
            oof = np.full(len(df), np.nan)
            fold_aurocs = []
            for fold_i, (tr, te) in enumerate(splits, 1):
                y_tr, y_te = y[tr], y[te]
                print(
                    f"{name} fold {fold_i}/{len(splits)} "
                    f"n_train={len(tr)} n_test={len(te)} pos_te={int(y_te.sum())}"
                )
                blocks_tr, blocks_te = [], []
                blocks_tr_mlp, blocks_te_mlp = [], []

                if fs in ("emb", "emb_baseline", "api_emb_baseline"):
                    E_tr = np.asarray(df.iloc[tr][emb_cols], dtype=np.float64)
                    E_te = np.asarray(df.iloc[te][emb_cols], dtype=np.float64)
                    emb_full = Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                        ]
                    )
                    emb_tr_full = emb_full.fit_transform(E_tr)
                    emb_te_full = emb_full.transform(E_te)
                    if pca_k > 0:
                        n_comp = min(pca_k, len(tr) - 1, E_tr.shape[1])
                        emb_pca = Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler()),
                                (
                                    "pca",
                                    PCA(
                                        n_components=n_comp,
                                        random_state=args.seed + fold_i,
                                    ),
                                ),
                            ]
                        )
                        emb_tr_tree = emb_pca.fit_transform(E_tr)
                        emb_te_tree = emb_pca.transform(E_te)
                    else:
                        emb_tr_tree, emb_te_tree = emb_tr_full, emb_te_full
                    blocks_tr.append(np.asarray(emb_tr_tree))
                    blocks_te.append(np.asarray(emb_te_tree))
                    blocks_tr_mlp.append(np.asarray(emb_tr_full))
                    blocks_te_mlp.append(np.asarray(emb_te_full))

                if fs in ("api", "api_baseline", "api_emb_baseline"):
                    atr = api_pipe.fit_transform(df.iloc[tr][api_feats])
                    ate = api_pipe.transform(df.iloc[te][api_feats])
                    blocks_tr.append(np.asarray(atr, dtype=float))
                    blocks_te.append(np.asarray(ate, dtype=float))
                    blocks_tr_mlp.append(np.asarray(atr, dtype=float))
                    blocks_te_mlp.append(np.asarray(ate, dtype=float))

                if fs in ("baseline", "api_baseline", "emb_baseline", "api_emb_baseline"):
                    bcols = ["cadd_phred", "mutation_type", "Func.refGene"]
                    if args.gene_feature == "onehot":
                        bcols = bcols + ["gene"]
                    btr = base_ct.fit_transform(df.iloc[tr][bcols])
                    bte = base_ct.transform(df.iloc[te][bcols])
                    blocks_tr.append(np.asarray(btr, dtype=float))
                    blocks_te.append(np.asarray(bte, dtype=float))
                    blocks_tr_mlp.append(np.asarray(btr, dtype=float))
                    blocks_te_mlp.append(np.asarray(bte, dtype=float))
                elif fs in ("emb", "api") and gene_ct is not None:
                    gtr = gene_ct.fit_transform(df.iloc[tr][["gene"]])
                    gte = gene_ct.transform(df.iloc[te][["gene"]])
                    blocks_tr.append(np.asarray(gtr, dtype=float))
                    blocks_te.append(np.asarray(gte, dtype=float))
                    blocks_tr_mlp.append(np.asarray(gtr, dtype=float))
                    blocks_te_mlp.append(np.asarray(gte, dtype=float))

                Z_tr = np.hstack(blocks_tr)
                Z_te = np.hstack(blocks_te)
                if kind == "mlp":
                    Z_tr = np.hstack(blocks_tr_mlp)
                    Z_te = np.hstack(blocks_te_mlp)

                prob = _fit_predict(
                    kind,
                    Z_tr,
                    y_tr,
                    Z_te,
                    seed=args.seed,
                    fold_i=fold_i,
                    hidden_dim=args.hidden_dim,
                    max_iter=args.max_iter,
                    w_pos=w_pos,
                    w_neg=w_neg,
                )
                oof[te] = prob
                fold_aurocs.append(ranking_metrics(y_te, prob)["auroc"])

            pred[f"p_{name}"] = oof
            m = ranking_metrics(y, oof)
            pm = prevalence_matched_table(y, oof)
            print(
                f"\n{name} OOF: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                f"PM_sens={pm.get('sensitivity')}"
            )
            summary_rows.append(
                {
                    "model": name,
                    "feature_set": fs,
                    "classifier": kind,
                    **m,
                    **{f"pm_{k}": v for k, v in pm.items()},
                    "cv_mode": args.cv_mode,
                    "gene_feature": args.gene_feature,
                    "fold_auroc_mean": float(np.mean(fold_aurocs)),
                    "fold_auroc_sd": float(np.std(fold_aurocs)),
                }
            )
            for r in source_sensitivity_pm(
                y, oof, df["source"].values, extra_membership=haem_ge3_membership(df)
            ):
                src_rows.append({"model": name, **r})
            for rc in sorted(df["region_class"].unique()):
                mask = df["region_class"].values == rc
                if y[mask].sum() == 0 or (y[mask] == 0).sum() == 0:
                    continue
                gm = ranking_metrics(y[mask], oof[mask])
                gpm = prevalence_matched_table(y[mask], oof[mask])
                region_rows.append(
                    {
                        "model": name,
                        "region_class": rc,
                        "n": int(mask.sum()),
                        "n_somatic": int(y[mask].sum()),
                        **gm,
                        "pm_sensitivity": gpm.get("sensitivity"),
                    }
                )
            for g in sorted(df["gene"].unique()):
                mask = df["gene"].values == g
                if y[mask].sum() == 0 or (y[mask] == 0).sum() == 0:
                    continue
                gm = ranking_metrics(y[mask], oof[mask])
                gpm = prevalence_matched_table(y[mask], oof[mask])
                per_gene_rows.append(
                    {
                        "model": name,
                        "gene": g,
                        "n": int(mask.sum()),
                        "n_somatic": int(y[mask].sum()),
                        **gm,
                        "pm_sensitivity": gpm.get("sensitivity"),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(out_dir / "oof_summary.csv", index=False)
    pd.DataFrame(src_rows).to_csv(out_dir / "source_sensitivity_pm.csv", index=False)
    pd.DataFrame(region_rows).to_csv(out_dir / "per_region_oof.csv", index=False)
    pd.DataFrame(per_gene_rows).to_csv(out_dir / "per_gene_oof.csv", index=False)
    pred.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "genes": genes,
                "n": len(df),
                "n_coding": int((df.region_class == "coding").sum()),
                "n_noncoding": int((df.region_class == "noncoding").sum()),
                "cv_mode": args.cv_mode,
                "gene_feature": args.gene_feature,
                "feature_sets": list(feature_sets),
                "classifiers": list(classifiers),
                "feature_mode": args.feature_mode,
                "pca": pca_k,
                "n_api": len(api_feats),
                "emb_dim": int(X_emb.shape[1]),
                "require_api": bool(args.require_api),
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
