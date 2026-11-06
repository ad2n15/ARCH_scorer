#!/usr/bin/env python3
"""Single-gene 5-fold CV (9 models) with Func / SBS6 stratification.

Same feature sets as pooled 106: api, baseline, api_baseline × lr/rf/xgb.

CV modes:
  stratify_label       label only (current gene-model default)
  stratify_func        Func.refGene × label
  stratify_sbs6        mutation_type (SBS6) × label
  stratify_func_sbs6   Func.refGene × SBS6 × label

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/108_cv5_gene_stratify_func_sbs6.py \\
      --gene ASXL1 --cv-mode stratify_func_sbs6
  # also run label-only for a head-to-head:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/108_cv5_gene_stratify_func_sbs6.py \\
      --gene ASXL1 --cv-mode stratify_label,stratify_func_sbs6
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

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

from importlib.util import module_from_spec, spec_from_file_location

_spec = spec_from_file_location("cv5_pooled", PROJECT / "scripts" / "106_cv5_pooled_arch_genes.py")
_mod = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_api_cols = _mod._api_cols
_load_annovar = _mod._load_annovar
_resolve_cadd = _mod._resolve_cadd
_resolve_annovar = _mod._resolve_annovar
_make_ohe = _mod._make_ohe
_collapse_rare_strata = _mod._collapse_rare_strata


def _make_splits(
    cv_mode: str,
    y: np.ndarray,
    *,
    funcs: np.ndarray,
    muts: np.ndarray,
    n_splits: int,
    seed: int,
):
    f0 = np.array([str(f).split(";")[0] for f in funcs], dtype=object)
    m0 = np.array([str(m) for m in muts], dtype=object)
    if cv_mode == "stratify_label":
        strata = y.astype(str)
    elif cv_mode == "stratify_func":
        strata = np.array([f"{fi}_{yi}" for fi, yi in zip(f0, y)], dtype=object)
    elif cv_mode == "stratify_sbs6":
        strata = np.array([f"{mi}_{yi}" for mi, yi in zip(m0, y)], dtype=object)
    elif cv_mode == "stratify_func_sbs6":
        strata = np.array(
            [f"{fi}_{mi}_{yi}" for fi, mi, yi in zip(f0, m0, y)], dtype=object
        )
    else:
        raise SystemExit(f"Unknown --cv-mode {cv_mode}")
    strata = _collapse_rare_strata(strata, y, n_splits)
    n_strata = len(np.unique(strata))
    print(f"{cv_mode}: {n_strata} strata after rare collapse")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), strata))


def _fit_predict(kind, X_tr, y_tr, X_te, *, seed, fold_i):
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


def _run_one(
    *,
    df: pd.DataFrame,
    api_feats: list[str],
    feature_sets: tuple[str, ...],
    classifiers: tuple[str, ...],
    cv_mode: str,
    n_splits: int,
    seed: int,
    out_dir: Path,
) -> None:
    y = df["y"].astype(int).values
    splits = _make_splits(
        cv_mode,
        y,
        funcs=df["Func.refGene"].astype(str).values,
        muts=df["mutation_type"].astype(str).values,
        n_splits=n_splits,
        seed=seed,
    )
    print(f"n_folds={len(splits)}")
    cat_cols = ["mutation_type", "Func.refGene"]

    def build_matrix(fs, tr_idx, te_idx):
        blocks_tr, blocks_te = [], []
        if fs in ("api", "api_baseline"):
            pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
            blocks_tr.append(np.asarray(pipe.fit_transform(df.iloc[tr_idx][api_feats]), float))
            blocks_te.append(np.asarray(pipe.transform(df.iloc[te_idx][api_feats]), float))
        if fs in ("baseline", "api_baseline"):
            ct = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
                    ("cat", _make_ohe(), cat_cols),
                ]
            )
            cols = ["cadd_phred"] + cat_cols
            blocks_tr.append(np.asarray(ct.fit_transform(df.iloc[tr_idx][cols]), float))
            blocks_te.append(np.asarray(ct.transform(df.iloc[te_idx][cols]), float))
        return np.hstack(blocks_tr), np.hstack(blocks_te)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, src_rows, func_rows = [], [], []
    pred = df[
        [
            c
            for c in [
                "variant_id",
                "gene",
                "label",
                "source",
                "y",
                "n_haem",
                "haem_ge3",
                "Func.refGene",
                "mutation_type",
            ]
            if c in df.columns
        ]
    ].copy()

    for fs in feature_sets:
        for kind in classifiers:
            name = f"{fs}_{kind}"
            oof = np.full(len(df), np.nan)
            fold_aurocs = []
            for fold_i, (tr, te) in enumerate(splits, 1):
                print(
                    f"{name} fold {fold_i}/{len(splits)} "
                    f"n_train={len(tr)} n_test={len(te)} pos_te={int(y[te].sum())}"
                )
                Z_tr, Z_te = build_matrix(fs, tr, te)
                prob = _fit_predict(kind, Z_tr, y[tr], Z_te, seed=seed, fold_i=fold_i)
                oof[te] = prob
                fold_aurocs.append(ranking_metrics(y[te], prob)["auroc"])
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
                    "cv_mode": cv_mode,
                    "fold_auroc_mean": float(np.mean(fold_aurocs)),
                    "fold_auroc_sd": float(np.std(fold_aurocs)),
                }
            )
            for r in source_sensitivity_pm(
                y, oof, df["source"].values, extra_membership=haem_ge3_membership(df)
            ):
                src_rows.append({"model": name, **r})
            # per Func.refGene sensitivity at gene-local PM thr
            thr = float(pm.get("threshold") or np.nan)
            if np.isfinite(thr):
                for func, sub in df.groupby(df["Func.refGene"].astype(str).str.split(";").str[0]):
                    idx = sub.index.to_numpy()
                    ys = y[idx]
                    if ys.sum() == 0:
                        continue
                    called = int((oof[idx][ys == 1] >= thr).sum())
                    func_rows.append(
                        {
                            "model": name,
                            "Func.refGene": func,
                            "n": int(len(idx)),
                            "n_somatic": int(ys.sum()),
                            "n_called": called,
                            "sens_at_pm": called / float(ys.sum()),
                            "auroc": ranking_metrics(ys, oof[idx])["auroc"]
                            if (ys.min() != ys.max())
                            else np.nan,
                        }
                    )

    pd.DataFrame(summary_rows).to_csv(out_dir / "oof_summary.csv", index=False)
    pd.DataFrame(src_rows).to_csv(out_dir / "source_sensitivity_pm.csv", index=False)
    if func_rows:
        pd.DataFrame(func_rows).to_csv(out_dir / "per_func_oof.csv", index=False)
    pred.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "gene": str(df["gene"].iloc[0]),
                "n": len(df),
                "cv_mode": cv_mode,
                "feature_sets": list(feature_sets),
                "classifiers": list(classifiers),
                "n_splits": len(splits),
                "n_api_feats": len(api_feats),
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene", default="ASXL1")
    ap.add_argument(
        "--cv-mode",
        default="stratify_func_sbs6",
        help="Comma list of: stratify_label,stratify_func,stratify_sbs6,stratify_func_sbs6",
    )
    ap.add_argument("--feature-sets", default="api,baseline,api_baseline")
    ap.add_argument("--classifiers", default="lr,rf,xgb")
    ap.add_argument("--mutation-scheme", choices=("raw", "sbs6"), default="sbs6")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    gene = args.gene.strip().upper()
    cv_modes = [x.strip() for x in args.cv_mode.split(",") if x.strip()]
    feature_sets = tuple(x.strip() for x in args.feature_sets.split(",") if x.strip())
    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())
    allowed = {
        "stratify_label",
        "stratify_func",
        "stratify_sbs6",
        "stratify_func_sbs6",
    }
    bad = [m for m in cv_modes if m not in allowed]
    if bad:
        raise SystemExit(f"Unknown cv-mode(s): {bad}")

    cohort = project_subpath(
        PROJECT, f"data/arch_mane_noncoding_pooled_sm/by_gene/{gene}.csv"
    )
    sc_path = project_subpath(
        PROJECT,
        f"results/track_scores_arch_mane_noncoding_pooled_sm_by_gene/{gene}/test_features_wide.parquet",
    )
    cadd_p = _resolve_cadd(gene)
    if not cohort.is_file() or not sc_path.is_file() or cadd_p is None:
        raise SystemExit(f"Missing cohort/scores/CADD for {gene}")

    df = pd.read_csv(cohort)
    df["variant_id"] = df["variant_id"].astype(str)
    df["gene"] = gene
    df["y"] = (df["label"].astype(str).str.lower() == "somatic").astype(int)
    if "source" not in df.columns:
        df["source"] = np.where(df["y"] == 1, "cosmic_ncv", "gnomad")

    sc = pd.read_parquet(sc_path)
    sc["variant_id"] = sc["variant_id"].astype(str)
    api_feats = _api_cols(sc)
    df = df.merge(sc[["variant_id"] + api_feats], on="variant_id", how="inner")

    ann = _load_annovar(_resolve_annovar(gene))
    df = df.merge(ann, on="variant_id", how="left")
    df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)

    cadd = pd.read_csv(cadd_p)
    cadd["variant_id"] = cadd["variant_id"].astype(str)
    cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
    df = df.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")
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
    df = df.loc[df["cadd_phred"].notna()].drop_duplicates("variant_id").reset_index(drop=True)
    df = attach_cosmic_haem_flags(df)

    print(
        f"{gene}: n={len(df)} germline={(df.y == 0).sum()} somatic={(df.y == 1).sum()} "
        f"api={len(api_feats)} modes={cv_modes}"
    )
    print("Func × somatic:\n", pd.crosstab(df["Func.refGene"].str.split(";").str[0], df["y"]))
    print("SBS6 × somatic:\n", pd.crosstab(df["mutation_type"], df["y"]))

    for mode in cv_modes:
        out = project_subpath(
            PROJECT,
            args.out_dir
            or f"results/{gene.lower()}_pooled_sm_cv5_strat_{mode}",
        )
        # if multiple modes and custom out-dir, nest
        if args.out_dir and len(cv_modes) > 1:
            out = project_subpath(PROJECT, args.out_dir) / mode
        print(f"\n=== {gene} {mode} → {out} ===")
        _run_one(
            df=df,
            api_feats=api_feats,
            feature_sets=feature_sets,
            classifiers=classifiers,
            cv_mode=mode,
            n_splits=int(args.n_splits),
            seed=int(args.seed),
            out_dir=out,
        )


if __name__ == "__main__":
    main()
