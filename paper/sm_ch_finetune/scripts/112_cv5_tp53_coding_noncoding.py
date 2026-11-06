#!/usr/bin/env python3
"""TP53 5-fold CV on coding ∪ noncoding with Func×SBS6 stratification.

Feature sets (default all six):
  baseline | api | emb | api_baseline | emb_baseline | api_emb_baseline

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/112_cv5_tp53_coding_noncoding.py \\
      --cv-mode stratify_func_sbs6 --feature-mode diff --pca 64
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

_s102 = spec_from_file_location("cv5_emb", PROJECT / "scripts" / "102_cv5_fold1_embeddings.py")
_m102 = module_from_spec(_s102)
assert _s102.loader is not None
_s102.loader.exec_module(_m102)
_load_X = _m102._load_X

_s106 = spec_from_file_location("cv5_pooled", PROJECT / "scripts" / "106_cv5_pooled_arch_genes.py")
_m106 = module_from_spec(_s106)
assert _s106.loader is not None
_s106.loader.exec_module(_m106)
_api_cols = _m106._api_cols
_load_annovar = _m106._load_annovar
_make_ohe = _m106._make_ohe

_s108 = spec_from_file_location(
    "cv5_gene_strat", PROJECT / "scripts" / "108_cv5_gene_stratify_func_sbs6.py"
)
_m108 = module_from_spec(_s108)
assert _s108.loader is not None
_s108.loader.exec_module(_m108)
_make_splits = _m108._make_splits


def _annovar_path(kind: str, gene: str) -> Path:
    if kind == "coding":
        return project_subpath(
            PROJECT,
            f"data/arch_mane_coding_pooled_sm/annovar_by_gene/{gene}_annovar.hg38_multianno.csv",
        )
    return project_subpath(
        PROJECT,
        f"data/arch_mane_noncoding_pooled_sm/annovar_by_gene/{gene}_annovar.hg38_multianno.csv",
    )


def _cadd_path(kind: str, gene: str) -> Path:
    if kind == "coding":
        api = project_subpath(
            PROJECT,
            f"data/arch_mane_coding_pooled_sm/cadd_by_gene/{gene}_cadd_phred_api.csv",
        )
        local = project_subpath(
            PROJECT,
            f"data/arch_mane_coding_pooled_sm/cadd_by_gene/{gene}_cadd_phred_local.csv",
        )
    else:
        local = project_subpath(
            PROJECT,
            f"data/arch_mane_noncoding_pooled_sm/cadd_by_gene/{gene}_cadd_phred_local.csv",
        )
        api = project_subpath(
            PROJECT,
            f"data/arch_mane_noncoding_pooled_sm/cadd_by_gene/{gene}_cadd_phred_api.csv",
        )
    if local.is_file():
        return local
    if api.is_file():
        return api
    raise SystemExit(f"Missing CADD for {kind}/{gene}")


def _load_arm(
    *,
    gene: str,
    kind: str,
    cohort: Path,
    scores: Path,
    emb_dir: Path,
    feature_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(cohort)
    df["variant_id"] = df["variant_id"].astype(str)
    df["gene"] = gene
    df["region_class"] = kind
    df["y"] = (df["label"].astype(str).str.lower() == "somatic").astype(int)
    if "source" not in df.columns:
        df["source"] = np.where(df["y"] == 1, "cosmic_ncv", "gnomad")

    sc = pd.read_parquet(scores)
    sc["variant_id"] = sc["variant_id"].astype(str)
    api_feats = _api_cols(sc)
    df = df.merge(sc[["variant_id"] + api_feats], on="variant_id", how="inner")

    ann = _load_annovar(_annovar_path(kind, gene))
    df = df.merge(ann, on="variant_id", how="left")
    df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)

    cadd = pd.read_csv(_cadd_path(kind, gene))
    cadd["variant_id"] = cadd["variant_id"].astype(str)
    cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
    df = df.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")

    if "ref" not in df.columns or "alt" not in df.columns:
        raise SystemExit(f"{kind}: missing ref/alt")
    df["mutation_type"] = [
        mutation_type_feature(str(r), str(a), "sbs6")
        for r, a in zip(df["ref"], df["alt"])
    ]

    meta, X_emb, feat_tag = _load_X(emb_dir, feature_mode)
    emb_map = {
        str(v): X_emb[i] for i, v in enumerate(meta["variant_id"].astype(str))
    }
    miss = [v for v in df["variant_id"].astype(str) if v not in emb_map]
    if miss:
        raise SystemExit(f"{kind}: {len(miss)} variants missing embeddings e.g. {miss[:3]}")
    X = np.stack([emb_map[v] for v in df["variant_id"].astype(str)])
    df = df.loc[df["cadd_phred"].notna()].drop_duplicates("variant_id").reset_index(drop=True)
    # re-align emb after cadd drop
    X = np.stack([emb_map[v] for v in df["variant_id"].astype(str)])
    print(
        f"  {kind}: n={len(df)} somatic={int(df.y.sum())} api={len(api_feats)} "
        f"emb={X.shape} tag={feat_tag}"
    )
    return df, X, api_feats


def _fit_predict(kind, X_tr, y_tr, X_te, *, seed, fold_i, hidden_dim, max_iter, w_pos, w_neg):
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    spw = (n_neg / n_pos) if (kind == "xgb" and n_pos > 0) else None
    if kind == "mlp":
        sw = np.where(y_tr == 1, w_pos, w_neg)
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=min(256, max(32, len(y_tr) // 10)),
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed + fold_i,
        )
        clf.fit(X_tr, y_tr, sample_weight=sw)
        return clf.predict_proba(X_te)[:, 1]
    if kind == "lr":
        clf = LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    clf = make_track_classifier(kind, random_state=seed + fold_i, scale_pos_weight=spw)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene", default="TP53")
    ap.add_argument(
        "--feature-sets",
        default="baseline,api,emb,api_baseline,emb_baseline,api_emb_baseline",
    )
    ap.add_argument("--classifiers", default="lr,rf,xgb,mlp")
    ap.add_argument("--feature-mode", default="diff", choices=("alt", "ref", "diff", "alt_diff"))
    ap.add_argument("--pca", type=int, default=64, help="PCA on emb for LR/RF/XGB; MLP full")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument(
        "--cv-mode",
        default="stratify_func_sbs6",
        choices=("stratify_label", "stratify_func", "stratify_sbs6", "stratify_func_sbs6"),
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    gene = args.gene.strip().upper()
    feature_sets = tuple(x.strip() for x in args.feature_sets.split(",") if x.strip())
    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())

    arms = [
        (
            "noncoding",
            project_subpath(PROJECT, f"data/arch_mane_noncoding_pooled_sm/by_gene/{gene}.csv"),
            project_subpath(
                PROJECT,
                f"results/track_scores_arch_mane_noncoding_pooled_sm_by_gene/{gene}/test_features_wide.parquet",
            ),
            project_subpath(PROJECT, f"results/fold1_embeddings_pooled_sm/{gene}"),
        ),
        (
            "coding",
            project_subpath(PROJECT, f"data/arch_mane_coding_pooled_sm/by_gene/{gene}.csv"),
            project_subpath(
                PROJECT,
                f"results/track_scores_arch_mane_coding_pooled_sm_by_gene/{gene}/test_features_wide.parquet",
            ),
            project_subpath(PROJECT, f"results/fold1_embeddings_coding_pooled_sm/{gene}"),
        ),
    ]

    dfs, Xs, api_feats = [], [], None
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
        elif api_a != api_feats:
            # intersect to common columns
            api_feats = [c for c in api_feats if c in api_a]
        dfs.append(df_a)
        Xs.append(X_a)

    df = pd.concat(dfs, ignore_index=True)
    X_emb = np.vstack(Xs)
    # keep only shared API cols
    keep_api = [c for c in api_feats if c in df.columns]
    api_feats = keep_api
    emb_cols = [f"e{i}" for i in range(X_emb.shape[1])]
    for i, c in enumerate(emb_cols):
        df[c] = X_emb[:, i]

    df = attach_cosmic_haem_flags(df)
    y = df["y"].astype(int).values
    print(
        f"{gene} coding∪noncoding: n={len(df)} germline={(y == 0).sum()} "
        f"somatic={int(y.sum())} api={len(api_feats)} emb={X_emb.shape[1]} "
        f"cv={args.cv_mode}"
    )
    print(df.groupby("region_class")["y"].agg(["size", "sum"]).rename(columns={"sum": "n_som"}))

    splits = _make_splits(
        args.cv_mode,
        y,
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

    base_ct = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
            ("cat", _make_ohe(), ["mutation_type", "Func.refGene"]),
        ]
    )
    api_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    )

    out_dir = project_subpath(
        PROJECT,
        args.out_dir
        or (
            f"results/{gene.lower()}_coding_noncoding_cv5_"
            f"{args.cv_mode}_emb_{args.feature_mode}_pca{pca_k}"
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
    summary_rows, src_rows, region_rows = [], [], []

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
                    btr = base_ct.fit_transform(
                        df.iloc[tr][["cadd_phred", "mutation_type", "Func.refGene"]]
                    )
                    bte = base_ct.transform(
                        df.iloc[te][["cadd_phred", "mutation_type", "Func.refGene"]]
                    )
                    blocks_tr.append(np.asarray(btr, dtype=float))
                    blocks_te.append(np.asarray(bte, dtype=float))
                    blocks_tr_mlp.append(np.asarray(btr, dtype=float))
                    blocks_te_mlp.append(np.asarray(bte, dtype=float))

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
                    "fold_auroc_mean": float(np.mean(fold_aurocs)),
                    "fold_auroc_sd": float(np.std(fold_aurocs)),
                }
            )
            for r in source_sensitivity_pm(
                y, oof, df["source"].values, extra_membership=haem_ge3_membership(df)
            ):
                src_rows.append({"model": name, **r})
            for rc in sorted(df["region_class"].unique()):
                mask = (df["region_class"].values == rc)
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

    pd.DataFrame(summary_rows).to_csv(out_dir / "oof_summary.csv", index=False)
    pd.DataFrame(src_rows).to_csv(out_dir / "source_sensitivity_pm.csv", index=False)
    pd.DataFrame(region_rows).to_csv(out_dir / "per_region_oof.csv", index=False)
    pred.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "gene": gene,
                "n": len(df),
                "n_coding": int((df.region_class == "coding").sum()),
                "n_noncoding": int((df.region_class == "noncoding").sum()),
                "cv_mode": args.cv_mode,
                "feature_sets": list(feature_sets),
                "classifiers": list(classifiers),
                "feature_mode": args.feature_mode,
                "pca": pca_k,
                "n_api": len(api_feats),
                "emb_dim": X_emb.shape[1],
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
