#!/usr/bin/env python3
"""Pooled ARCH 5-fold CV across genes: API / baseline / API+baseline (3×3).

Combines all genes (default: 21 pooled-SM by_gene cohorts). Options:

  --cv-mode stratify_gene        Stratify folds on gene×label (default)
  --cv-mode stratify_func        Stratify on Func.refGene×label
  --cv-mode stratify_gene_func   Stratify on gene×Func.refGene×label
  --cv-mode group_gene           Hold out whole genes
  --gene-feature none|onehot

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/106_cv5_pooled_arch_genes.py \\
      --cv-mode stratify_gene_func --gene-feature none
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

META_DROP = {"variant_id", "label", "gene", "source", "y", "chrom", "pos", "ref", "alt"}

ARCH21 = [
    "ASXL1",
    "ASXL2",
    "CBL",
    "DNMT3A",
    "ETNK1",
    "GNAS",
    "GNB1",
    "IDH1",
    "IDH2",
    "JAK2",
    "KIT",
    "KRAS",
    "MPL",
    "NRAS",
    "PPM1D",
    "RUNX1",
    "SETBP1",
    "SF3B1",
    "SRSF2",
    "TET2",
    "TP53",
]


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _load_annovar(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(path)
    ann["variant_id"] = (
        ann["Chr"].astype(str)
        + "-"
        + ann["Start"].astype(str)
        + "-"
        + ann["Ref"].astype(str)
        + "-"
        + ann["Alt"].astype(str)
    )
    return ann[["variant_id", "Func.refGene"]].copy()


def _resolve_cadd(gene: str) -> Path | None:
    local = PROJECT / f"data/arch_mane_noncoding_pooled_sm/cadd_by_gene/{gene}_cadd_phred_local.csv"
    api = PROJECT / f"data/arch_mane_noncoding_pooled_sm/cadd_by_gene/{gene}_cadd_phred_api.csv"
    if local.is_file():
        return local
    if api.is_file():
        return api
    return None


def _resolve_annovar(gene: str) -> Path:
    g = PROJECT / (
        f"data/arch_mane_noncoding_pooled_sm/annovar_by_gene/{gene}_annovar.hg38_multianno.csv"
    )
    pool = PROJECT / (
        "data/arch_mane_noncoding_pooled_sm/"
        "arch_mane_noncoding_pooled_annovar.hg38_multianno.csv"
    )
    return g if g.is_file() else pool


def _api_cols(sc: pd.DataFrame) -> list[str]:
    return [
        c
        for c in sc.columns
        if c not in META_DROP and c != "variant_id" and pd.api.types.is_numeric_dtype(sc[c])
    ]


def _collapse_rare_strata(strata: np.ndarray, y: np.ndarray, n_splits: int) -> np.ndarray:
    """Merge strata with count < n_splits; fall back to label-only if still rare."""
    strata = np.asarray(strata, dtype=object)
    vc = pd.Series(strata).value_counts()
    rare = set(vc[vc < n_splits].index.astype(str))
    if not rare:
        return strata
    out = np.array(
        [s if s not in rare else f"_lab_{yi}" for s, yi in zip(strata, y)],
        dtype=object,
    )
    vc2 = pd.Series(out).value_counts()
    rare2 = set(vc2[vc2 < n_splits].index.astype(str))
    if rare2:
        out = np.array(
            [s if s not in rare2 else str(int(yi)) for s, yi in zip(out, y)],
            dtype=object,
        )
    return out


def _make_splits(
    cv_mode: str,
    y: np.ndarray,
    genes: np.ndarray,
    n_splits: int,
    seed: int,
    funcs: np.ndarray | None = None,
):
    if cv_mode == "stratify_gene":
        strata = np.array([f"{g}_{yi}" for g, yi in zip(genes, y)], dtype=object)
        strata = _collapse_rare_strata(strata, y, n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), strata))
    if cv_mode == "stratify_func":
        if funcs is None:
            raise SystemExit("stratify_func requires Func.refGene")
        # primary Func token (before ';')
        f0 = np.array([str(f).split(";")[0] for f in funcs], dtype=object)
        strata = np.array([f"{fi}_{yi}" for fi, yi in zip(f0, y)], dtype=object)
        strata = _collapse_rare_strata(strata, y, n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), strata))
    if cv_mode == "stratify_gene_func":
        if funcs is None:
            raise SystemExit("stratify_gene_func requires Func.refGene")
        f0 = np.array([str(f).split(";")[0] for f in funcs], dtype=object)
        strata = np.array(
            [f"{g}_{fi}_{yi}" for g, fi, yi in zip(genes, f0, y)], dtype=object
        )
        strata = _collapse_rare_strata(strata, y, n_splits)
        n_strata = len(np.unique(strata))
        print(
            f"stratify_gene_func: {n_strata} strata after rare collapse "
            f"(gene×Func×label)"
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), strata))
    if cv_mode == "group_gene":
        n_groups = len(np.unique(genes))
        n_use = min(n_splits, n_groups)
        if n_use < 2:
            raise SystemExit("Need ≥2 genes for group_gene CV")
        try:
            cv = StratifiedGroupKFold(n_splits=n_use, shuffle=True, random_state=seed)
            return list(cv.split(np.zeros(len(y)), y, groups=genes))
        except Exception:
            cv = GroupKFold(n_splits=n_use)
            return list(cv.split(np.zeros(len(y)), y, groups=genes))
    raise SystemExit(f"Unknown --cv-mode {cv_mode}")


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
    ap.add_argument(
        "--genes",
        default=",".join(ARCH21),
        help="Comma-separated ARCH genes (default: 21)",
    )
    ap.add_argument(
        "--cohort-dir",
        default="data/arch_mane_noncoding_pooled_sm/by_gene",
    )
    ap.add_argument(
        "--scores-root",
        default="results/track_scores_arch_mane_noncoding_pooled_sm_by_gene",
    )
    ap.add_argument(
        "--feature-sets",
        default="api,baseline,api_baseline",
        help="Comma list: api, baseline, api_baseline",
    )
    ap.add_argument("--classifiers", default="lr,rf,xgb")
    ap.add_argument(
        "--cv-mode",
        choices=("stratify_gene", "stratify_func", "stratify_gene_func", "group_gene"),
        default="stratify_gene",
    )
    ap.add_argument(
        "--gene-feature",
        choices=("none", "onehot"),
        default="none",
    )
    ap.add_argument("--mutation-scheme", choices=("raw", "sbs6"), default="sbs6")
    ap.add_argument("--keep-missing-cadd", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    feature_sets = tuple(x.strip() for x in args.feature_sets.split(",") if x.strip())
    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())
    need_api = any(fs.startswith("api") for fs in feature_sets)
    need_base = any("baseline" in fs for fs in feature_sets)
    need_func = args.cv_mode in ("stratify_func", "stratify_gene_func") or need_base

    rows = []
    for gene in genes:
        cohort = project_subpath(PROJECT, args.cohort_dir) / f"{gene}.csv"
        if not cohort.is_file():
            print(f"SKIP {gene}: missing {cohort}")
            continue
        df = pd.read_csv(cohort)
        df["variant_id"] = df["variant_id"].astype(str)
        df["gene"] = gene
        df["y"] = (df["label"].astype(str).str.lower() == "somatic").astype(int)
        if "source" not in df.columns:
            df["source"] = np.where(df["y"] == 1, "cosmic_ncv", "gnomad")

        if need_api:
            sc_path = (
                project_subpath(PROJECT, args.scores_root)
                / gene
                / "test_features_wide.parquet"
            )
            if not sc_path.is_file():
                print(f"SKIP {gene}: missing API scores {sc_path}")
                continue
            sc = pd.read_parquet(sc_path)
            sc["variant_id"] = sc["variant_id"].astype(str)
            api_cols = _api_cols(sc)
            df = df.merge(sc[["variant_id"] + api_cols], on="variant_id", how="inner")

        if need_func:
            ann = _load_annovar(_resolve_annovar(gene))
            df = df.merge(ann, on="variant_id", how="left")
            df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)

        if need_base:
            cadd_p = _resolve_cadd(gene)
            if cadd_p is None:
                print(f"SKIP {gene}: missing CADD")
                continue
            cadd = pd.read_csv(cadd_p)
            cadd["variant_id"] = cadd["variant_id"].astype(str)
            cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
            df = df.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")
            if "Func.refGene" not in df.columns:
                df["Func.refGene"] = "unknown"
            if "ref" in df.columns and "alt" in df.columns:
                df["mutation_type"] = [
                    mutation_type_feature(str(r), str(a), args.mutation_scheme)
                    for r, a in zip(df["ref"], df["alt"])
                ]
            else:
                parts = df["variant_id"].str.split("-", n=3, expand=True)
                df["mutation_type"] = [
                    mutation_type_feature(str(r), str(a), args.mutation_scheme)
                    for r, a in zip(parts[2], parts[3])
                ]
            if not args.keep_missing_cadd:
                df = df.loc[df["cadd_phred"].notna()].copy()

        rows.append(df.drop_duplicates("variant_id"))

    if not rows:
        raise SystemExit("No genes loaded")
    df = pd.concat(rows, ignore_index=True)
    df = attach_cosmic_haem_flags(df)
    print(
        f"Pooled n={len(df)} genes={df['gene'].nunique()} "
        f"germline={(df.y == 0).sum()} somatic={(df.y == 1).sum()} "
        f"cv={args.cv_mode} gene_feature={args.gene_feature}"
    )
    print(df.groupby("gene")["y"].agg(["size", "sum"]).rename(columns={"sum": "n_som"}).to_string())

    y = df["y"].astype(int).values
    gene_arr = df["gene"].astype(str).values
    func_arr = (
        df["Func.refGene"].astype(str).values
        if "Func.refGene" in df.columns
        else None
    )
    splits = _make_splits(
        args.cv_mode, y, gene_arr, int(args.n_splits), args.seed, funcs=func_arr
    )
    print(f"n_folds={len(splits)}")

    api_feats = [
        c
        for c in df.columns
        if any(
            c.startswith(p)
            for p in (
                "RNA_SEQ__",
                "SPLICE_SITE_USAGE__",
                "SPLICE_JUNCTIONS__",
                "DNASE__",
                "CHIP_HISTONE__",
            )
        )
    ]
    if need_api and len(api_feats) != 10:
        print(f"WARNING: expected 10 API feats, got {len(api_feats)}: {api_feats}")

    cat_cols_base = ["mutation_type", "Func.refGene"]
    if args.gene_feature == "onehot":
        gene_cols = ["gene"]
    else:
        gene_cols = []

    def build_matrix(fs: str, tr_idx, te_idx):
        """Fit transforms on train, return Z_tr, Z_te."""
        blocks_tr, blocks_te = [], []
        if fs in ("api", "api_baseline"):
            pipe = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                ]
            )
            atr = pipe.fit_transform(df.iloc[tr_idx][api_feats])
            ate = pipe.transform(df.iloc[te_idx][api_feats])
            blocks_tr.append(np.asarray(atr, dtype=np.float64))
            blocks_te.append(np.asarray(ate, dtype=np.float64))
        if fs in ("baseline", "api_baseline"):
            ct_cols = ["cadd_phred"] + cat_cols_base
            ct = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
                    ("cat", _make_ohe(), cat_cols_base),
                ],
                remainder="drop",
            )
            btr = ct.fit_transform(df.iloc[tr_idx][ct_cols])
            bte = ct.transform(df.iloc[te_idx][ct_cols])
            blocks_tr.append(np.asarray(btr, dtype=np.float64))
            blocks_te.append(np.asarray(bte, dtype=np.float64))
        if gene_cols:
            g_ohe = _make_ohe()
            gtr = g_ohe.fit_transform(df.iloc[tr_idx][gene_cols])
            gte = g_ohe.transform(df.iloc[te_idx][gene_cols])
            blocks_tr.append(np.asarray(gtr, dtype=np.float64))
            blocks_te.append(np.asarray(gte, dtype=np.float64))
        return np.hstack(blocks_tr), np.hstack(blocks_te)

    tag = (
        f"pooled_arch21_cv5_{args.cv_mode}_genefeat_{args.gene_feature}"
    )
    out_dir = project_subpath(
        PROJECT, args.out_dir or f"results/{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    src_rows = []
    per_gene_rows = []
    pred = df[
        [c for c in ["variant_id", "gene", "label", "source", "y", "n_haem", "haem_ge3"] if c in df.columns]
    ].copy()

    for fs in feature_sets:
        for kind in classifiers:
            name = f"{fs}_{kind}"
            oof = np.full(len(df), np.nan)
            fold_metrics = []
            for fold_i, (tr, te) in enumerate(splits, 1):
                print(
                    f"{name} fold {fold_i}/{len(splits)} "
                    f"n_train={len(tr)} n_test={len(te)} "
                    f"genes_te={len(np.unique(gene_arr[te]))} pos_te={int(y[te].sum())}"
                )
                Z_tr, Z_te = build_matrix(fs, tr, te)
                prob = _fit_predict(
                    kind, Z_tr, y[tr], Z_te, seed=args.seed, fold_i=fold_i
                )
                oof[te] = prob
                fold_metrics.append(ranking_metrics(y[te], prob))
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
                    "fold_auroc_mean": float(np.mean([x["auroc"] for x in fold_metrics])),
                    "fold_auroc_sd": float(np.std([x["auroc"] for x in fold_metrics])),
                }
            )
            for r in source_sensitivity_pm(
                y, oof, df["source"].values, extra_membership=haem_ge3_membership(df)
            ):
                src_rows.append({"model": name, **r})
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
    pd.DataFrame(per_gene_rows).to_csv(out_dir / "per_gene_oof.csv", index=False)
    pred.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "genes": genes,
                "n": len(df),
                "n_genes": int(df["gene"].nunique()),
                "cv_mode": args.cv_mode,
                "gene_feature": args.gene_feature,
                "feature_sets": list(feature_sets),
                "classifiers": list(classifiers),
                "n_splits": len(splits),
                "mutation_scheme": args.mutation_scheme,
                "n_api_feats": len(api_feats),
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
