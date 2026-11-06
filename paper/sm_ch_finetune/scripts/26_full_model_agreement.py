#!/usr/bin/env python3
"""Do covariates / AG / NTv3 / Borzoi rank and call the same variants?

Outputs pairwise Spearman of P(CH), prevalence-matched call-set Jaccard,
and fraction of shared top-k CH calls on the full-cohort held-out test set.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import mutation_type as ref_alt_mutation_type
from eval.categorical_features import fit_one_hot
from paths import project_subpath, resolve_config, resolve_data_split_dir
from scoring.track_classifiers import make_track_classifier

MODELS = ("covariates", "alphagenome", "ntv3", "borzoi")


def _load_probs(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["variant_id"] = df["variant_id"].astype(str)
    col = "prob_ch" if "prob_ch" in df.columns else "finetune_prob"
    return df[["variant_id", col]].rename(columns={col: name})


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_subpath(PROJECT, p)


def _covariate_blocks(train: pd.DataFrame, test: pd.DataFrame, ens: dict):
    blocks_tr, blocks_te = [], []
    if ens.get("include_cadd", True):
        col = ens.get("cadd_col", "cadd")
        med = float(train[col].median())
        blocks_tr.append(train[col].fillna(med).astype(float).values.reshape(-1, 1))
        blocks_te.append(test[col].fillna(med).astype(float).values.reshape(-1, 1))
    if ens.get("include_exonic_func", True):
        col = ens.get("exonic_func_col", "exonic_func")
        X_tr, X_te, _ = fit_one_hot(train[col], test[col], col_name=col)
        blocks_tr.append(X_tr)
        blocks_te.append(X_te)
    if ens.get("include_mutation_type", True):
        col = ens.get("mutation_type_col", "mutation_type")
        if train[col].isna().any() and "ref" in train.columns:
            for df in (train, test):
                df[col] = df[col].fillna(
                    df.apply(lambda r: ref_alt_mutation_type(r["ref"], r["alt"]), axis=1)
                )
        X_tr, X_te, _ = fit_one_hot(train[col], test[col], col_name=col)
        blocks_tr.append(X_tr)
        blocks_te.append(X_te)
    if ens.get("include_gene", True):
        col = ens.get("gene_col", "gene")
        X_tr, X_te, _ = fit_one_hot(
            train[col], test[col], col_name=col, top_k=ens.get("gene_top_k")
        )
        blocks_tr.append(X_tr)
        blocks_te.append(X_te)
    return np.hstack(blocks_tr), np.hstack(blocks_te)


def _topk_mask(prob: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros(len(prob), dtype=bool)
    if k > 0:
        m[np.argsort(-prob)[:k]] = True
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="config/full_age_filtered_multibackbone_ensembles.yaml",
    )
    p.add_argument("--classifier", default="lr", choices=("lr", "rf", "xgb"))
    p.add_argument("--out", default="results/full_model_agreement.json")
    p.add_argument(
        "--scores-out",
        default="results/full_model_agreement_scores.parquet",
        help="Per-variant scores + prev-matched call flags for plotting",
    )
    args = p.parse_args()

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)
    ens = cfg.get("ensemble", {})
    ft = cfg["finetune"]
    seed = int(cfg.get("data", {}).get("seed", 42))
    C = float(ens.get("lr_C", 0.5))
    max_iter = int(ens.get("lr_max_iter", 2000))

    data_dir = resolve_data_split_dir(PROJECT, cfg)
    cols = [
        "variant_id", "label", "cadd", "exonic_func", "gene", "mutation_type", "ref", "alt",
    ]
    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    train["variant_id"] = train["variant_id"].astype(str)
    test["variant_id"] = test["variant_id"].astype(str)
    train = train[[c for c in cols if c in train.columns]].copy()
    test = test[[c for c in cols if c in test.columns]].copy()

    for bb in ("alphagenome", "ntv3", "borzoi"):
        tr = _load_probs(_resolve(ft[bb]["train_predictions"]), bb)
        te = _load_probs(_resolve(ft[bb]["test_predictions"]), bb)
        train = train.merge(tr, on="variant_id", how="inner")
        test = test.merge(te, on="variant_id", how="inner")

    y_tr = train["label"].astype(int).values
    y_te = test["label"].astype(int).values
    cov_tr, cov_te = _covariate_blocks(train, test, ens)

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    spw = (n_neg / n_pos) if (args.classifier == "xgb" and n_pos > 0) else None
    clf = make_track_classifier(
        args.classifier, C=C, max_iter=max_iter, random_state=seed, scale_pos_weight=spw
    )
    clf.fit(cov_tr, y_tr)
    scores = {
        "covariates": clf.predict_proba(cov_te)[:, 1],
        "alphagenome": test["alphagenome"].values.astype(float),
        "ntv3": test["ntv3"].values.astype(float),
        "borzoi": test["borzoi"].values.astype(float),
    }

    k = int(y_te.sum())  # prevalence-matched
    calls = {m: _topk_mask(s, k) for m, s in scores.items()}

    spearman = {}
    jaccard = {}
    shared_pos = {}
    kappa = {}
    print(f"n_test={len(test)} n_CH={k} cov_clf={args.classifier}")
    print("\nPairwise Spearman (scores) / Jaccard (prev-matched CH calls) / shared+/k")
    for a, b in combinations(MODELS, 2):
        rho, _ = spearmanr(scores[a], scores[b])
        inter = int((calls[a] & calls[b]).sum())
        union = int((calls[a] | calls[b]).sum())
        jac = inter / union if union else 0.0
        # Cohen's kappa on binary calls
        p_a = calls[a]
        p_b = calls[b]
        po = float((p_a == p_b).mean())
        pe = float(p_a.mean() * p_b.mean() + (~p_a).mean() * (~p_b).mean())
        kap = (po - pe) / (1 - pe) if pe < 1 else 1.0
        key = f"{a}__{b}"
        spearman[key] = float(rho)
        jaccard[key] = float(jac)
        shared_pos[key] = {"n_shared": inter, "k": k, "frac_of_k": inter / k if k else None}
        kappa[key] = float(kap)
        print(
            f"  {a:12s} vs {b:12s}  ρ={rho:.3f}  Jac={jac:.3f}  "
            f"shared={inter}/{k} ({inter/k:.1%})  κ={kap:.3f}"
        )

    # How often all 4 agree on a positive call
    all4 = calls["covariates"] & calls["alphagenome"] & calls["ntv3"] & calls["borzoi"]
    any1 = calls["covariates"] | calls["alphagenome"] | calls["ntv3"] | calls["borzoi"]
    print(f"\nAll-4 call positive: {int(all4.sum())}/{k} of prevalence slot ({all4.sum()/k:.1%})")
    print(f"Any-1 call positive union: {int(any1.sum())} variants (k={k})")

    # Among true CH, how many models call them (of 4)
    true_ch = y_te == 1
    n_models_calling = sum(calls[m].astype(int) for m in MODELS)
    print("\nAmong true CH variants, # models that call them CH (prev-matched):")
    for n in range(5):
        c = int(((n_models_calling == n) & true_ch).sum())
        print(f"  called by {n}/4 models: {c} ({c/k:.1%} of CH)")

    out = {
        "n_test": len(test),
        "n_ch": k,
        "covariate_classifier": args.classifier,
        "spearman": spearman,
        "jaccard_prev_matched": jaccard,
        "shared_positive_calls": shared_pos,
        "cohen_kappa_prev_matched": kappa,
        "all4_positive": int(all4.sum()),
        "any1_positive_union": int(any1.sum()),
        "true_ch_by_n_models_calling": {
            str(n): int(((n_models_calling == n) & true_ch).sum()) for n in range(5)
        },
    }
    out_path = project_subpath(PROJECT, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")

    scores_path = project_subpath(PROJECT, args.scores_out)
    score_df = pd.DataFrame(
        {
            "variant_id": test["variant_id"].values,
            "label": y_te,
            **{f"score_{m}": scores[m] for m in MODELS},
            **{f"call_{m}": calls[m].astype(int) for m in MODELS},
            "n_models_calling": n_models_calling.astype(int),
        }
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_parquet(scores_path, index=False)
    print(f"Saved scores -> {scores_path}")


if __name__ == "__main__":
    main()
