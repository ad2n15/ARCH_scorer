#!/usr/bin/env python3
"""5-fold CV on fold_1 variant-site embeddings (1536-d ALT / REF / DIFF).

Input from scripts/101_extract_fold1_variant_embeddings.py:
  results/fold1_embeddings_pooled_sm/<GENE>/{variants_meta.parquet,emb_*.npy}

Example (CPU/AMD OK after extract):
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/102_cv5_fold1_embeddings.py \\
      --gene TP53 --feature-mode diff --pca 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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


def _load_X(
    emb_dir: Path, feature_mode: str
) -> tuple[pd.DataFrame, np.ndarray, str]:
    meta = pd.read_parquet(emb_dir / "variants_meta.parquet")
    meta["variant_id"] = meta["variant_id"].astype(str)
    if "y" not in meta.columns:
        meta["y"] = (meta["label"].astype(str).str.lower() == "somatic").astype(int)

    mode = feature_mode.strip().lower()
    if mode == "alt":
        X = np.load(emb_dir / "emb_alt.npy")
        tag = "emb_alt"
    elif mode == "ref":
        X = np.load(emb_dir / "emb_ref.npy")
        tag = "emb_ref"
    elif mode == "diff":
        p = emb_dir / "emb_diff.npy"
        if p.exists():
            X = np.load(p)
        else:
            X = np.load(emb_dir / "emb_alt.npy") - np.load(emb_dir / "emb_ref.npy")
        tag = "emb_diff"
    elif mode in {"alt_diff", "concat"}:
        alt = np.load(emb_dir / "emb_alt.npy")
        diff_p = emb_dir / "emb_diff.npy"
        if diff_p.exists():
            diff = np.load(diff_p)
        else:
            diff = alt - np.load(emb_dir / "emb_ref.npy")
        X = np.concatenate([alt, diff], axis=1)
        tag = "emb_alt_diff"
    else:
        raise SystemExit(
            f"Unknown --feature-mode {feature_mode!r}; use alt|ref|diff|alt_diff"
        )

    if len(meta) != len(X):
        raise SystemExit(f"meta n={len(meta)} != emb n={len(X)} in {emb_dir}")
    return meta, np.asarray(X, dtype=np.float32), tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene", default="TP53")
    ap.add_argument(
        "--emb-dir",
        default=None,
        help="Default: results/fold1_embeddings_pooled_sm/<GENE>",
    )
    ap.add_argument(
        "--feature-mode",
        default="diff",
        choices=("alt", "ref", "diff", "alt_diff", "concat"),
        help="Embedding features (default: ALT−REF diff)",
    )
    ap.add_argument(
        "--pca",
        type=int,
        default=64,
        help="PCA dims before classifier (0 = no PCA). Default 64.",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classifiers", default="lr,rf,xgb")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    gene = args.gene.strip().upper()
    emb_dir = project_subpath(
        PROJECT,
        args.emb_dir or f"results/fold1_embeddings_pooled_sm/{gene}",
    )
    if not (emb_dir / "variants_meta.parquet").is_file():
        raise SystemExit(
            f"Missing embeddings under {emb_dir}. Run 101 first:\n"
            f"  sbatch run_binemo_l40.sh sm_ch_finetune/scripts/"
            f"101_extract_fold1_variant_embeddings.py --gene {gene}"
        )

    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())
    meta, X_raw, feat_tag = _load_X(emb_dir, args.feature_mode)
    if "source" not in meta.columns:
        meta["source"] = np.where(meta["y"] == 1, "cosmic_ncv", "gnomad")
    meta = attach_cosmic_haem_flags(meta)
    y = meta["y"].astype(int).values

    print(
        f"{gene} fold_1 embeddings CV: n={len(meta)} "
        f"germline={(y == 0).sum()} somatic={(y == 1).sum()} "
        f"X={X_raw.shape} mode={feat_tag} pca={args.pca}"
    )
    print(meta["source"].value_counts().to_string())

    n_splits = int(args.n_splits)
    if (y == 1).sum() < n_splits or (y == 0).sum() < n_splits:
        raise SystemExit("Too few positives/negatives for stratified CV")

    X_raw = SimpleImputer(strategy="median").fit_transform(
        np.nan_to_num(X_raw, nan=np.nan, posinf=np.nan, neginf=np.nan)
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    model_specs = [(f"emb_{kind}", kind) for kind in classifiers]
    oof = {m: np.full(len(meta), np.nan) for m, _ in model_specs}
    fold_metrics: dict[str, list] = {m: [] for m, _ in model_specs}

    pca_k = int(args.pca)
    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(meta)), y), 1):
        y_tr, y_te = y[tr], y[te]
        X_tr, X_te = X_raw[tr], X_raw[te]
        print(
            f"fold {fold_i}/{n_splits} n_train={len(tr)} n_test={len(te)} "
            f"pos_te={int(y_te.sum())}"
        )
        for name, kind in model_specs:
            n_pos = int(y_tr.sum())
            n_neg = int(len(y_tr) - n_pos)
            spw = (n_neg / n_pos) if (kind == "xgb" and n_pos > 0) else None
            steps = [("scaler", StandardScaler())]
            if pca_k > 0:
                n_comp = min(pca_k, X_tr.shape[0] - 1, X_tr.shape[1])
                steps.append(
                    ("pca", PCA(n_components=n_comp, random_state=args.seed + fold_i))
                )
            if kind == "lr":
                clf = LogisticRegression(
                    max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"
                )
            else:
                clf = make_track_classifier(
                    kind, random_state=args.seed + fold_i, scale_pos_weight=spw
                )
            steps.append(("clf", clf))
            pipe = Pipeline(steps)
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_te)[:, 1]
            oof[name][te] = prob
            fold_metrics[name].append(ranking_metrics(y_te, prob))

    tag = gene.lower()
    pca_tag = f"pca{pca_k}" if pca_k > 0 else "nopca"
    out_dir = project_subpath(
        PROJECT,
        args.out_dir
        or f"results/{tag}_pooled_sm_cv5_fold1_emb_{feat_tag}_{pca_tag}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = meta[
        [c for c in ["variant_id", "chrom", "pos", "ref", "alt", "label", "source", "y"] if c in meta.columns]
    ].copy()
    summary_rows = []
    src_rows = []
    for name, _kind in model_specs:
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
                "feature_mode": feat_tag,
                "pca": pca_k,
            }
        )
        src_sens = source_sensitivity_pm(
            y,
            oof[name],
            meta["source"].values,
            extra_membership=haem_ge3_membership(meta),
        )
        for r in src_sens:
            src_rows.append({"model": name, **r})
            if r.get("scope") == "source_membership":
                print(f"  {r.get('source')}: sens={r.get('sensitivity')} n={r.get('n_true_pos')}")

    fold_df = []
    for name, rows in fold_metrics.items():
        for i, m in enumerate(rows, 1):
            fold_df.append({"model": name, "fold": i, **m})
    pd.DataFrame(fold_df).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "oof_summary.csv", index=False)
    if src_rows:
        pd.DataFrame(src_rows).to_csv(out_dir / "source_sensitivity_pm.csv", index=False)
    pred.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "gene": gene,
                "emb_dir": str(emb_dir),
                "feature_mode": feat_tag,
                "pca": pca_k,
                "n": len(meta),
                "n_splits": n_splits,
                "seed": args.seed,
                "classifiers": list(classifiers),
                "X_shape": list(X_raw.shape),
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
