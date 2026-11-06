#!/usr/bin/env python3
"""Evaluate saved fine-tune probs + tabular covariates (no API, CPU-only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import mutation_type as ref_alt_mutation_type
from eval.categorical_features import fit_one_hot
from eval.classification_metrics import prevalence_matched_table, print_contingency, ranking_metrics
from eval.stratified_metrics import (
    eval_by_genes,
    eval_by_stratum,
    print_stratum_table,
    stratum_payload,
)
from paths import project_subpath, resolve_config, resolve_data_split_dir


def _covariate_suffix(*, cadd: bool, exonic: bool, gene: bool, mutation: bool) -> str:
    suffix = ""
    if cadd:
        suffix += "_plus_cadd"
    if exonic:
        suffix += "_plus_exonic"
    if gene:
        suffix += "_plus_gene"
    if mutation:
        suffix += "_plus_mutation"
    return suffix


def _finetune_covariate_name(*, cadd: bool, exonic: bool, gene: bool, mutation: bool) -> str:
    return "ensemble_finetune" + _covariate_suffix(
        cadd=cadd, exonic=exonic, gene=gene, mutation=mutation,
    )


def _attach_meta(df: pd.DataFrame, meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = meta[["variant_id"] + [c for c in cols if c in meta.columns]].copy()
    m["variant_id"] = m["variant_id"].astype(str)
    m = m.drop_duplicates("variant_id")
    out = df.merge(m, on="variant_id", how="left", suffixes=("", "_meta"))
    for c in cols:
        if f"{c}_meta" in out.columns:
            out[c] = out[c].fillna(out[f"{c}_meta"])
            out = out.drop(columns=[f"{c}_meta"])
    return out


def _ensure_mutation_type(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise SystemExit(f"Missing {col!r} in split parquet")
    if df[col].notna().all():
        return df
    out = df.copy()
    if "ref" in out.columns and "alt" in out.columns:
        out[col] = out[col].fillna(
            out.apply(lambda r: ref_alt_mutation_type(r["ref"], r["alt"]), axis=1),
        )
    return out


def _cadd_matrix(train: pd.DataFrame, test: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    med = float(train[col].median())
    X_tr = train[col].fillna(med).astype(float).values.reshape(-1, 1)
    X_te = test[col].fillna(med).astype(float).values.reshape(-1, 1)
    return X_tr, X_te


def _covariate_blocks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ens: dict,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict]:
    extras_tr: list[np.ndarray] = []
    extras_te: list[np.ndarray] = []
    feature_info: dict = {}
    flags = {
        "cadd": bool(ens.get("include_cadd", False)),
        "exonic": bool(ens.get("include_exonic_func", False)),
        "gene": bool(ens.get("include_gene", False)),
        "mutation": bool(ens.get("include_mutation_type", False)),
    }

    if flags["cadd"]:
        col = ens.get("cadd_col", "cadd")
        X_tr, X_te = _cadd_matrix(train, test, col)
        extras_tr.append(X_tr)
        extras_te.append(X_te)
        feature_info["cadd_col"] = col

    if flags["exonic"]:
        col = ens.get("exonic_func_col", "exonic_func")
        X_tr, X_te, names = fit_one_hot(train[col], test[col], col_name=col)
        extras_tr.append(X_tr)
        extras_te.append(X_te)
        feature_info["n_exonic_func_features"] = len(names)

    if flags["gene"]:
        col = ens.get("gene_col", "gene")
        top_k = ens.get("gene_top_k", 100)
        X_tr, X_te, names = fit_one_hot(train[col], test[col], col_name=col, top_k=top_k)
        extras_tr.append(X_tr)
        extras_te.append(X_te)
        feature_info["gene_top_k"] = top_k
        feature_info["n_gene_features"] = len(names)

    if flags["mutation"]:
        col = ens.get("mutation_type_col", "mutation_type")
        train = _ensure_mutation_type(train, col)
        test = _ensure_mutation_type(test, col)
        X_tr, X_te, names = fit_one_hot(train[col], test[col], col_name=col)
        extras_tr.append(X_tr)
        extras_te.append(X_te)
        feature_info["n_mutation_type_features"] = len(names)

    return extras_tr, extras_te, flags, feature_info


def _eval_lr(name: str, X_tr, y_tr, X_te, y_te, *, C: float, max_iter: int) -> dict:
    lr = LogisticRegression(max_iter=max_iter, C=C)
    lr.fit(X_tr, y_tr)
    prob = lr.predict_proba(X_te)[:, 1]
    rank = ranking_metrics(y_te, prob)
    prev = prevalence_matched_table(y_te, prob)
    return {
        "model": name,
        "n_features": int(X_tr.shape[1]),
        "fusion": "late",
        **rank,
        "prevalence_matched": prev,
        "prob": prob,
    }


def _eval_probs(name: str, y_te, prob_te, *, n_features: int) -> dict:
    rank = ranking_metrics(y_te, prob_te)
    prev = prevalence_matched_table(y_te, prob_te)
    return {
        "model": name,
        "n_features": int(n_features),
        **rank,
        "prevalence_matched": prev,
        "prob": prob_te,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/full_age_filtered_covariates.yaml")
    p.add_argument("--train-predictions", default=None)
    p.add_argument("--test-predictions", default=None)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--min-stratum-n",
        type=int,
        default=None,
        help="Minimum test variants per stratum (default: ensemble.min_stratum_n)",
    )
    args = p.parse_args()

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    ft_cfg = cfg.get("finetune", {})
    train_pred = Path(args.train_predictions or ft_cfg["train_predictions"])
    test_pred = Path(args.test_predictions or ft_cfg["test_predictions"])
    if not train_pred.is_absolute():
        train_pred = project_subpath(PROJECT, train_pred)
    if not test_pred.is_absolute():
        test_pred = project_subpath(PROJECT, test_pred)
    if not train_pred.exists() or not test_pred.exists():
        raise SystemExit(
            f"Missing predictions:\n  {train_pred}\n  {test_pred}\n"
            "Run 04_evaluate.py --export-only --export-splits train test first."
        )

    data_dir = resolve_data_split_dir(PROJECT, cfg)
    meta_cols = [
        "cadd", "exonic_func", "gene", "mutation_type", "ref", "alt", "func",
        "mchip", "lchip",
    ]
    train_meta = pd.read_parquet(data_dir / "train.parquet")
    test_meta = pd.read_parquet(data_dir / "test.parquet")

    tr = pd.read_parquet(train_pred)
    te = pd.read_parquet(test_pred)
    tr["variant_id"] = tr["variant_id"].astype(str)
    te["variant_id"] = te["variant_id"].astype(str)
    prob_col = "prob_ch" if "prob_ch" in tr.columns else "finetune_prob"
    tr = tr.rename(columns={prob_col: "finetune_prob"})
    te = te.rename(columns={prob_col: "finetune_prob"})
    # Borzoi/NTv3 exports may be variant_id+prob only — take labels from split meta.
    if "label" not in tr.columns:
        tr = tr.merge(train_meta[["variant_id", "label"]], on="variant_id", how="inner")
    if "label" not in te.columns:
        te = te.merge(test_meta[["variant_id", "label"]], on="variant_id", how="inner")

    train = _attach_meta(tr[["variant_id", "label", "finetune_prob"]], train_meta, meta_cols)
    test = _attach_meta(te[["variant_id", "label", "finetune_prob"]], test_meta, meta_cols)
    print(f"Joined: train={len(train)} test={len(test)} from {data_dir}")

    ens = cfg.get("ensemble", {})
    C = float(ens.get("lr_C", 0.5))
    max_iter = int(ens.get("lr_max_iter", 2000))
    extras_tr, extras_te, flags, feature_info = _covariate_blocks(train, test, ens)
    y_tr = train["label"].astype(int).values
    y_te = test["label"].astype(int).values
    ft_tr = train["finetune_prob"].values.reshape(-1, 1)
    ft_te = test["finetune_prob"].values.reshape(-1, 1)

    results = [
        _eval_probs("finetune_center_only", y_te, test["finetune_prob"].values, n_features=1),
    ]

    if extras_tr:
        cov_tr = np.hstack(extras_tr)
        cov_te = np.hstack(extras_te)
        results.append(_eval_lr("covariates_only", cov_tr, y_tr, cov_te, y_te, C=C, max_iter=max_iter))
        results.append(_eval_lr(
            _finetune_covariate_name(**flags),
            np.hstack([ft_tr, cov_tr]),
            y_tr,
            np.hstack([ft_te, cov_te]),
            y_te,
            C=C,
            max_iter=max_iter,
        ))

    for r in results:
        print(f"\n{r['model']}: AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f} (p={r['n_features']})")
        print_contingency("  prevalence-matched", r["prevalence_matched"])

    stratify_cols = ens.get("stratify_cols", ["mchip", "lchip"])
    stratify_func = bool(ens.get("stratify_func", False))
    stratify_exonic = bool(ens.get("stratify_exonic_func", False))
    func_col = ens.get("func_col", "func")
    exonic_col = ens.get("exonic_func_col", "exonic_func")
    stratify_genes = list(ens.get("stratify_genes", ["DNMT3A", "TET2", "ASXL1"]))
    gene_col = ens.get("gene_col", "gene")
    min_stratum_n = (
        args.min_stratum_n
        if args.min_stratum_n is not None
        else int(ens.get("min_stratum_n", 30))
    )
    stratify_model = ens.get("stratify_model")
    if stratify_model is None:
        stratify_model = results[-1]["model"] if len(results) > 1 else results[0]["model"]
    model_probs = {r["model"]: r.pop("prob") for r in results}

    by_stratum: dict[str, dict] = {}
    by_func: dict[str, dict] = {}
    by_exonic_func: dict[str, dict] = {}
    attach_cols = []
    if stratify_func and func_col not in test.columns and func_col in test_meta.columns:
        attach_cols.append(func_col)
    if stratify_exonic and exonic_col not in test.columns and exonic_col in test_meta.columns:
        attach_cols.append(exonic_col)
    if attach_cols:
        test = _attach_meta(test, test_meta, attach_cols)
    if stratify_func:
        min_func_n = int(ens.get("min_func_stratum_n", min_stratum_n))
        for model_name, prob in model_probs.items():
            rows = eval_by_stratum(
                test, prob, stratify_col=func_col, min_n=min_func_n,
            )
            if not rows:
                continue
            print_stratum_table(model_name, func_col, rows, min_n=min_func_n)
            by_func[model_name] = stratum_payload(
                model_name, func_col, rows, min_n=min_func_n,
            )
    if stratify_exonic:
        min_exonic_n = int(ens.get("min_exonic_stratum_n", min_stratum_n))
        for model_name, prob in model_probs.items():
            rows = eval_by_stratum(
                test, prob, stratify_col=exonic_col, min_n=min_exonic_n,
            )
            if not rows:
                continue
            print_stratum_table(model_name, exonic_col, rows, min_n=min_exonic_n)
            by_exonic_func[model_name] = stratum_payload(
                model_name, exonic_col, rows, min_n=min_exonic_n,
            )

    if stratify_cols and stratify_model in model_probs:
        for col in stratify_cols:
            rows = eval_by_stratum(
                test, model_probs[stratify_model], stratify_col=col, min_n=min_stratum_n,
            )
            if rows:
                print_stratum_table(stratify_model, col, rows, min_n=min_stratum_n)
                by_stratum[col] = stratum_payload(
                    stratify_model, col, rows, min_n=min_stratum_n,
                )
        if stratify_genes:
            rows = eval_by_genes(
                test,
                model_probs[stratify_model],
                gene_col=gene_col,
                genes=stratify_genes,
                min_n=min_stratum_n,
            )
            if rows:
                print_stratum_table(stratify_model, gene_col, rows, min_n=min_stratum_n)
                by_stratum["stratify_genes"] = stratum_payload(
                    stratify_model, gene_col, rows, min_n=min_stratum_n,
                )
                by_stratum["stratify_genes"]["genes"] = stratify_genes

    out = Path(args.out) if args.out else project_subpath(
        PROJECT, cfg.get("results_file", "results/finetune_covariates_metrics.json"),
    )
    payload = {
        "data_dir": str(data_dir),
        "train_predictions": str(train_pred),
        "test_predictions": str(test_pred),
        "n_train": len(train),
        "n_test": len(test),
        "ch_rate_test": float(test["label"].mean()),
        "ensemble_type": "meta_lr_on_finetune_prob_and_covariates",
        "include_cadd": flags.get("cadd", False),
        "include_exonic_func": flags.get("exonic", False),
        "include_gene": flags.get("gene", False),
        "include_mutation_type": flags.get("mutation", False),
        **feature_info,
        "metrics": [{k: v for k, v in m.items() if k not in ("prevalence_matched", "prob")} for m in results],
        "prevalence_matched": {m["model"]: m["prevalence_matched"] for m in results},
    }
    if by_stratum:
        payload["by_stratum"] = by_stratum
        payload["stratify_model"] = stratify_model
    if by_func:
        payload["by_func"] = by_func
        payload["func_col"] = func_col
    if by_exonic_func:
        payload["by_exonic_func"] = by_exonic_func
        payload["exonic_func_col"] = exonic_col
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
