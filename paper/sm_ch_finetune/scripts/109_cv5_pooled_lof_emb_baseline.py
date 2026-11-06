#!/usr/bin/env python3
"""Pooled LoF-gene 5-fold CV: fold_1 emb (± baseline).

Default genes (9): ASXL1,ASXL2,CBL,DNMT3A,ETNK1,PPM1D,RUNX1,TET2,TP53

--features:
  emb_baseline  emb + CADD + SBS6 + Func (+ optional gene onehot); models emb_base_*
  emb           emb only (+ optional gene onehot); models emb_*
                Still loads ANNOVAR Func.refGene for CV strata when needed;
                missing annovar → Func="unknown". Skips CADD requirement.

CV modes:
  stratify_label
  stratify_gene
  stratify_func
  stratify_sbs6
  stratify_gene_func
  stratify_func_sbs6
  stratify_gene_func_sbs6   (gene × Func × SBS6 × label)

LR/RF/XGB: PCA on emb (default 64) + optional baseline (CPU; default classifiers).
MLP: use scripts/113_cv5_pooled_emb_baseline_mlp_gpu.py on a GPU node.

Example (CPU trees/LR only):
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/109_cv5_pooled_lof_emb_baseline.py \\
      --cv-mode stratify_gene_func --gene-feature onehot --pca 64
  # ARCH21 emb-only:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/109_cv5_pooled_lof_emb_baseline.py \\
      --genes ASXL1,ASXL2,CBL,DNMT3A,ETNK1,GNAS,GNB1,IDH1,IDH2,JAK2,KIT,KRAS,MPL,NRAS,PPM1D,RUNX1,SETBP1,SF3B1,SRSF2,TET2,TP53 \\
      --features emb --cv-mode stratify_gene_func --gene-feature onehot --pca 64
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
from sklearn.model_selection import StratifiedKFold
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
_load_annovar = _m106._load_annovar
_resolve_cadd = _m106._resolve_cadd
_resolve_annovar = _m106._resolve_annovar
_make_ohe = _m106._make_ohe
_collapse_rare_strata = _m106._collapse_rare_strata

LOF9 = [
    "ASXL1",
    "ASXL2",
    "CBL",
    "DNMT3A",
    "ETNK1",
    "PPM1D",
    "RUNX1",
    "TET2",
    "TP53",
]


def _make_splits(
    cv_mode: str,
    y: np.ndarray,
    *,
    genes: np.ndarray,
    funcs: np.ndarray,
    muts: np.ndarray,
    n_splits: int,
    seed: int,
):
    f0 = np.array([str(f).split(";")[0] for f in funcs], dtype=object)
    m0 = np.array([str(m) for m in muts], dtype=object)
    g0 = np.array([str(g) for g in genes], dtype=object)
    if cv_mode == "stratify_label":
        strata = y.astype(str)
    elif cv_mode == "stratify_gene":
        strata = np.array([f"{g}_{yi}" for g, yi in zip(g0, y)], dtype=object)
    elif cv_mode == "stratify_func":
        strata = np.array([f"{fi}_{yi}" for fi, yi in zip(f0, y)], dtype=object)
    elif cv_mode == "stratify_sbs6":
        strata = np.array([f"{mi}_{yi}" for mi, yi in zip(m0, y)], dtype=object)
    elif cv_mode == "stratify_gene_func":
        strata = np.array(
            [f"{g}_{fi}_{yi}" for g, fi, yi in zip(g0, f0, y)], dtype=object
        )
    elif cv_mode == "stratify_func_sbs6":
        strata = np.array(
            [f"{fi}_{mi}_{yi}" for fi, mi, yi in zip(f0, m0, y)], dtype=object
        )
    elif cv_mode == "stratify_gene_func_sbs6":
        strata = np.array(
            [f"{g}_{fi}_{mi}_{yi}" for g, fi, mi, yi in zip(g0, f0, m0, y)],
            dtype=object,
        )
    else:
        raise SystemExit(f"Unknown --cv-mode {cv_mode}")
    strata = _collapse_rare_strata(strata, y, n_splits)
    print(f"{cv_mode}: {len(np.unique(strata))} strata after rare collapse")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), strata))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes", default=",".join(LOF9))
    ap.add_argument("--feature-mode", default="diff", choices=("alt", "ref", "diff", "alt_diff", "concat"))
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument(
        "--classifiers",
        default="lr,rf,xgb",
        help="Comma list. Default lr,rf,xgb (CPU). For MLP use script 113 on GPU.",
    )
    ap.add_argument(
        "--cv-mode",
        default="stratify_gene_func_sbs6",
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
    ap.add_argument(
        "--features",
        choices=("emb", "emb_baseline"),
        default="emb_baseline",
        help="emb: emb (± gene onehot); emb_baseline: emb+CADD+SBS6+Func (± gene)",
    )
    ap.add_argument("--gene-feature", choices=("none", "onehot"), default="none")
    ap.add_argument("--mutation-scheme", choices=("raw", "sbs6"), default="sbs6")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())
    use_baseline = args.features == "emb_baseline"
    rows = []
    feat_tag = None
    emb_dim = None

    for gene in genes:
        emb_dir = project_subpath(PROJECT, f"results/fold1_embeddings_pooled_sm/{gene}")
        if not (emb_dir / "variants_meta.parquet").is_file():
            raise SystemExit(f"Missing embeddings: {emb_dir}")
        meta, X_emb, tag = _load_X(emb_dir, args.feature_mode)
        feat_tag = tag
        emb_dim = X_emb.shape[1]
        meta = meta.copy()
        meta["variant_id"] = meta["variant_id"].astype(str)
        meta["gene"] = gene
        if "y" not in meta.columns:
            meta["y"] = (meta["label"].astype(str).str.lower() == "somatic").astype(int)
        if "source" not in meta.columns:
            meta["source"] = np.where(meta["y"] == 1, "cosmic_ncv", "gnomad")

        df = meta
        if use_baseline:
            cadd_p = _resolve_cadd(gene)
            if cadd_p is None:
                raise SystemExit(f"Missing CADD for {gene}")
            cadd = pd.read_csv(cadd_p)
            cadd["variant_id"] = cadd["variant_id"].astype(str)
            cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
            df = df.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")

        ann_p = _resolve_annovar(gene)
        if Path(ann_p).is_file():
            ann = _load_annovar(ann_p)
            df = df.merge(ann, on="variant_id", how="left")
            df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
        else:
            df["Func.refGene"] = "unknown"

        if "ref" not in df.columns or "alt" not in df.columns:
            raise SystemExit(f"{gene}: meta missing ref/alt")
        df["mutation_type"] = [
            mutation_type_feature(str(r), str(a), args.mutation_scheme)
            for r, a in zip(df["ref"], df["alt"])
        ]
        emb_block = pd.DataFrame(
            X_emb, columns=[f"e{i}" for i in range(X_emb.shape[1])], index=df.index
        )
        df = pd.concat([df, emb_block], axis=1)
        if use_baseline:
            df = df.loc[df["cadd_phred"].notna()].drop_duplicates("variant_id")
        else:
            df = df.drop_duplicates("variant_id")
        rows.append(df)
        print(f"  loaded {gene}: n={len(df)} somatic={int(df.y.sum())}")

    df = pd.concat(rows, ignore_index=True)
    df = attach_cosmic_haem_flags(df)
    emb_cols = [f"e{i}" for i in range(emb_dim)]
    y = df["y"].astype(int).values
    print(
        f"Pooled {args.features} n={len(df)} genes={df['gene'].nunique()} "
        f"somatic={int(y.sum())} mode={feat_tag} pca={args.pca} "
        f"cv={args.cv_mode} gene_feature={args.gene_feature}"
    )
    print(
        df.groupby("gene")["y"]
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

    use_cols = list(emb_cols)
    if use_baseline:
        use_cols = use_cols + ["cadd_phred", "mutation_type", "Func.refGene"]
    if args.gene_feature == "onehot":
        use_cols = use_cols + ["gene"]
    X_df = df[use_cols].copy()

    pca_k = int(args.pca)
    base_ct = None
    gene_ct = None
    if use_baseline:
        base_cats = ["mutation_type", "Func.refGene"]
        if args.gene_feature == "onehot":
            base_cats = base_cats + ["gene"]
        base_ct = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
                ("cat", _make_ohe(), base_cats),
            ],
            remainder="drop",
        )
    elif args.gene_feature == "onehot":
        gene_ct = ColumnTransformer(
            [("cat", _make_ohe(), ["gene"])],
            remainder="drop",
        )

    model_prefix = "emb_base" if use_baseline else "emb"
    model_specs = [(f"{model_prefix}_{k}", k) for k in classifiers]
    oof = {m: np.full(len(df), np.nan) for m, _ in model_specs}
    fold_metrics: dict[str, list] = {m: [] for m, _ in model_specs}
    n_pos_all = max(int(y.sum()), 1)
    n_neg_all = max(int(len(y) - n_pos_all), 1)
    w_pos = 0.5 * len(y) / n_pos_all
    w_neg = 0.5 * len(y) / n_neg_all

    for fold_i, (tr, te) in enumerate(splits, 1):
        y_tr, y_te = y[tr], y[te]
        X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
        print(
            f"fold {fold_i}/{len(splits)} n_train={len(tr)} n_test={len(te)} "
            f"genes_te={df.iloc[te]['gene'].nunique()} pos_te={int(y_te.sum())}"
        )
        E_tr = np.asarray(X_tr[emb_cols], dtype=np.float64)
        E_te = np.asarray(X_te[emb_cols], dtype=np.float64)
        emb_full = Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
        )
        emb_tr_full = emb_full.fit_transform(E_tr)
        emb_te_full = emb_full.transform(E_te)
        if pca_k > 0:
            n_comp = min(pca_k, len(tr) - 1, E_tr.shape[1])
            emb_pca = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("pca", PCA(n_components=n_comp, random_state=args.seed + fold_i)),
                ]
            )
            emb_tr_tree = emb_pca.fit_transform(E_tr)
            emb_te_tree = emb_pca.transform(E_te)
        else:
            emb_tr_tree, emb_te_tree = emb_tr_full, emb_te_full

        extra_tr = extra_te = None
        if base_ct is not None:
            extra_tr = np.asarray(base_ct.fit_transform(X_tr), dtype=float)
            extra_te = np.asarray(base_ct.transform(X_te), dtype=float)
        elif gene_ct is not None:
            extra_tr = np.asarray(gene_ct.fit_transform(X_tr), dtype=float)
            extra_te = np.asarray(gene_ct.transform(X_te), dtype=float)

        if extra_tr is not None:
            Z_tr_tree = np.hstack([np.asarray(emb_tr_tree), extra_tr])
            Z_te_tree = np.hstack([np.asarray(emb_te_tree), extra_te])
            Z_tr_mlp = np.hstack([np.asarray(emb_tr_full), extra_tr])
            Z_te_mlp = np.hstack([np.asarray(emb_te_full), extra_te])
        else:
            Z_tr_tree, Z_te_tree = np.asarray(emb_tr_tree), np.asarray(emb_te_tree)
            Z_tr_mlp, Z_te_mlp = np.asarray(emb_tr_full), np.asarray(emb_te_full)

        for name, kind in model_specs:
            spw = (n_neg_all / n_pos_all) if kind == "xgb" else None
            if kind == "mlp":
                sw = np.where(y_tr == 1, w_pos, w_neg)
                clf = MLPClassifier(
                    hidden_layer_sizes=(int(args.hidden_dim),),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=min(256, max(32, len(tr) // 10)),
                    learning_rate_init=1e-3,
                    max_iter=int(args.max_iter),
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=args.seed + fold_i,
                )
                clf.fit(Z_tr_mlp, y_tr, sample_weight=sw)
                prob = clf.predict_proba(Z_te_mlp)[:, 1]
            elif kind == "lr":
                clf = LogisticRegression(
                    max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"
                )
                clf.fit(Z_tr_tree, y_tr)
                prob = clf.predict_proba(Z_te_tree)[:, 1]
            else:
                clf = make_track_classifier(
                    kind, random_state=args.seed + fold_i, scale_pos_weight=spw
                )
                clf.fit(Z_tr_tree, y_tr)
                prob = clf.predict_proba(Z_te_tree)[:, 1]
            oof[name][te] = prob
            fold_metrics[name].append(ranking_metrics(y_te, prob))

    pca_tag = f"pca{pca_k}" if pca_k > 0 else "nopca"
    out_dir = project_subpath(
        PROJECT,
        args.out_dir
        or (
            f"results/pooled_lof{len(genes)}_cv5_fold1_{args.features}_"
            f"{feat_tag}_{pca_tag}_{args.cv_mode}_genefeat_{args.gene_feature}"
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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
                "cadd_phred",
            ]
            if c in df.columns
        ]
    ].copy()
    summary_rows, src_rows, per_gene_rows = [], [], []
    for name, _ in model_specs:
        pred[f"p_{name}"] = oof[name]
        m = ranking_metrics(y, oof[name])
        pm = prevalence_matched_table(y, oof[name])
        print(
            f"\n{name} OOF: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
            f"PM_sens={pm.get('sensitivity')}"
        )
        summary_rows.append(
            {
                "model": name,
                **m,
                **{f"pm_{k}": v for k, v in pm.items()},
                "features": args.features,
                "cv_mode": args.cv_mode,
                "gene_feature": args.gene_feature,
                "fold_auroc_mean": float(np.mean([x["auroc"] for x in fold_metrics[name]])),
                "fold_auroc_sd": float(np.std([x["auroc"] for x in fold_metrics[name]])),
            }
        )
        for r in source_sensitivity_pm(
            y, oof[name], df["source"].values, extra_membership=haem_ge3_membership(df)
        ):
            src_rows.append({"model": name, **r})
        for g in sorted(df["gene"].unique()):
            mask = (df["gene"].values == g)
            if y[mask].sum() == 0 or (y[mask] == 0).sum() == 0:
                continue
            gm = ranking_metrics(y[mask], oof[name][mask])
            gpm = prevalence_matched_table(y[mask], oof[name][mask])
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
                "features": args.features,
                "feature_mode": feat_tag,
                "emb_dim": emb_dim,
                "pca": pca_k,
                "cv_mode": args.cv_mode,
                "gene_feature": args.gene_feature,
                "classifiers": list(classifiers),
                "mutation_scheme": args.mutation_scheme,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
