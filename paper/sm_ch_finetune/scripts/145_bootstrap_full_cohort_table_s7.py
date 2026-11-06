#!/usr/bin/env python3
"""Bootstrap 95% CI for Table S7 — genome-wide ARCH vs SM held-out test (full cohort finetune).

Rebuilds test-set probabilities as in scripts/08_evaluate_finetune_covariates.py (finetune /
covariates / ensemble meta-LR), then stratified bootstrap AUROC per Table S7 row.

Example:
  cd paper/sm_ch_finetune
  python3 scripts/145_bootstrap_full_cohort_table_s7.py --n-boot 10000 --seed 42
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from eval.classification_metrics import prevalence_matched_table, ranking_metrics  # noqa: E402
from paths import project_subpath, resolve_config, resolve_data_split_dir  # noqa: E402

_s08 = PROJECT / "scripts/08_evaluate_finetune_covariates.py"
_spec = importlib.util.spec_from_file_location("m08", _s08)
_m08 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_m08)

REF_CSV = PROJECT / "results/full_cohort_finetune_auroc_sens_spec_by_func.csv"

BACKBONE_PREDS = {
    "AlphaGenome": (
        PROJECT / "results/finetune_full_train_predictions.parquet",
        PROJECT / "results/finetune_full_test_predictions.parquet",
    ),
    "NTv3": (
        ROOT / "sm_ch_finetune_ntv3/results/ntv3_full_train_predictions.parquet",
        ROOT / "sm_ch_finetune_ntv3/results/ntv3_full_test_predictions.parquet",
    ),
    "Borzoi": (
        ROOT / "sm_ch_finetune_borzoi/results/borzoi_full_train_predictions.parquet",
        ROOT / "sm_ch_finetune_borzoi/results/borzoi_full_test_predictions.parquet",
    ),
}

INTERNAL_TO_TABLE = {
    "finetune_center_only": "finetune_only",
    "covariates_only": "covariates_only",
    "ensemble_finetune_plus_cadd_plus_exonic_plus_gene_plus_mutation": (
        "ensemble_ft_plus_covariates"
    ),
}


def stratified_bootstrap_auroc(
    y: np.ndarray,
    prob: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    y = np.asarray(y, int)
    prob = np.asarray(prob, float)
    if len(y) < 10 or y.sum() < 2 or (len(y) - y.sum()) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(y, prob))
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        ix = np.concatenate([b0, b1])
        if len(np.unique(y[ix])) < 2:
            continue
        boots.append(roc_auc_score(y[ix], prob[ix]))
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _build_all_backbones(cfg: dict) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """One covariate fit; per-backbone finetune + ensemble."""
    ens = cfg.get("ensemble", {})
    data_dir = resolve_data_split_dir(PROJECT, cfg)
    meta_cols = [
        "cadd", "exonic_func", "gene", "mutation_type", "ref", "alt", "func",
    ]
    train_meta = pd.read_parquet(data_dir / "train.parquet")
    test_meta = pd.read_parquet(data_dir / "test.parquet")

    C = float(ens.get("lr_C", 0.5))
    max_iter = int(ens.get("lr_max_iter", 2000))

    # Covariates-only: backbone-independent (same tabular X)
    print("Fitting shared covariates-only LR on train …", flush=True)
    tr0 = pd.read_parquet(BACKBONE_PREDS["AlphaGenome"][0])
    tr0["variant_id"] = tr0["variant_id"].astype(str)
    prob_col = "prob_ch" if "prob_ch" in tr0.columns else "finetune_prob"
    tr0 = tr0.rename(columns={prob_col: "finetune_prob"})
    if "label" not in tr0.columns:
        tr0 = tr0.merge(train_meta[["variant_id", "label"]], on="variant_id", how="inner")
    train0 = _m08._attach_meta(
        tr0[["variant_id", "label", "finetune_prob"]], train_meta, meta_cols
    )
    te0 = pd.read_parquet(BACKBONE_PREDS["AlphaGenome"][1])
    te0["variant_id"] = te0["variant_id"].astype(str)
    te0 = te0.rename(columns={prob_col: "finetune_prob"})
    if "label" not in te0.columns:
        te0 = te0.merge(test_meta[["variant_id", "label"]], on="variant_id", how="inner")
    test = _m08._attach_meta(
        te0[["variant_id", "label", "finetune_prob"]], test_meta, meta_cols
    )

    extras_tr, extras_te, flags, _fi = _m08._covariate_blocks(train0, test, ens)
    y_tr = train0["label"].astype(int).values
    y_te = test["label"].astype(int).values
    cov_tr = np.hstack(extras_tr)
    cov_te = np.hstack(extras_te)
    cov_res = _m08._eval_lr(
        "covariates_only", cov_tr, y_tr, cov_te, y_te, C=C, max_iter=max_iter
    )
    prob_cov = cov_res.pop("prob")

    out: dict[str, dict[str, np.ndarray]] = {}
    for bb, (tr_p, te_p) in BACKBONE_PREDS.items():
        if not te_p.is_file():
            raise SystemExit(f"Missing {te_p}")
        print(f"  {bb} finetune + ensemble …", flush=True)
        tr = pd.read_parquet(tr_p)
        te = pd.read_parquet(te_p)
        tr["variant_id"] = tr["variant_id"].astype(str)
        te["variant_id"] = te["variant_id"].astype(str)
        pc = "prob_ch" if "prob_ch" in tr.columns else "finetune_prob"
        tr = tr.rename(columns={pc: "finetune_prob"})
        te = te.rename(columns={pc: "finetune_prob"})
        if "label" not in tr.columns:
            tr = tr.merge(train_meta[["variant_id", "label"]], on="variant_id", how="inner")
        train = _m08._attach_meta(
            tr[["variant_id", "label", "finetune_prob"]], train_meta, meta_cols
        )
        ft_tr = train["finetune_prob"].values.reshape(-1, 1)
        ft_te = (
            test["variant_id"]
            .map(te.set_index("variant_id")["finetune_prob"])
            .astype(float)
            .values.reshape(-1, 1)
        )
        ens_res = _m08._eval_lr(
            _m08._finetune_covariate_name(**flags),
            np.hstack([ft_tr, cov_tr]),
            y_tr,
            np.hstack([ft_te, cov_te]),
            y_te,
            C=C,
            max_iter=max_iter,
        )
        out[bb] = {
            "finetune_only": ft_te.ravel(),
            "covariates_only": prob_cov,
            "ensemble_ft_plus_covariates": ens_res.pop("prob"),
        }
    return test, out


def _mask(test: pd.DataFrame, stratum_type: str, stratum: str) -> np.ndarray:
    if stratum_type == "overall":
        return np.ones(len(test), dtype=bool)
    if stratum_type == "func":
        return test["func"].astype(str).values == str(stratum)
    if stratum_type == "exonic_func":
        return test["exonic_func"].astype(str).values == str(stratum)
    raise ValueError(stratum_type)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference-csv", type=Path, default=REF_CSV)
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "results/full_cohort_finetune_table_s7_bootstrap.csv",
    )
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out

    with open(resolve_config("config/full_age_filtered_covariates.yaml", PROJECT)) as f:
        cfg = yaml.safe_load(f)

    ref = pd.read_csv(args.reference_csv)
    # baseline rows use covariates_only from any backbone (identical X)
    bundles: dict[str, dict[str, np.ndarray]] = {}
    print("Loading backbones + shared covariates …", flush=True)
    test_df, bundles = _build_all_backbones(cfg)

    baseline_probs = bundles["AlphaGenome"]["covariates_only"]

    tasks = list(ref.itertuples(index=False))
    iterator = tasks
    if tqdm is not None:
        iterator = tqdm(tasks, desc="Bootstrap slices", unit="row")

    rows = []
    seed = args.seed
    for row in iterator:
        bb = str(row.backbone)
        model = str(row.model)
        stype = str(row.stratum_type)
        strat = str(row.stratum)

        if bb == "baseline":
            if model != "covariates_only":
                seed += 1
                continue
            prob = baseline_probs
        else:
            if bb not in bundles or model not in bundles[bb]:
                seed += 1
                continue
            prob = bundles[bb][model]

        mask = _mask(test_df, stype, strat)
        y = test_df.loc[mask, "label"].astype(int).to_numpy()
        p = prob[mask]
        point, lo, hi = stratified_bootstrap_auroc(y, p, n_boot=args.n_boot, seed=seed)
        seed += 1
        rank = ranking_metrics(y, p) if len(y) and y.sum() and (len(y) - y.sum()) else {}
        pm = prevalence_matched_table(y, p) if len(y) else {}
        rows.append(
            {
                "backbone": bb,
                "model": model,
                "stratum_type": stype,
                "stratum": strat,
                "n": int(len(y)),
                "n_pos": int(y.sum()) if len(y) else 0,
                "auroc": point,
                "auroc_ci_lo": lo,
                "auroc_ci_hi": hi,
                "auprc": rank.get("auprc"),
                "pm_sensitivity": pm.get("sensitivity"),
                "pm_specificity": pm.get("specificity"),
                "n_bootstrap": args.n_boot,
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
