#!/usr/bin/env python3
"""Bootstrap 95% CI for Table S5 — 21-gene held-out finetune (all models × settings × regions).

Rebuilds test probabilities as in scripts/136_arch21_ft_vs_ft_baseline_lr.py, then stratified
bootstrap AUROC on each evaluation slice (no re-finetuning).

Example:
  cd paper/sm_ch_finetune
  python3 scripts/144_bootstrap_arch21_finetune_table_s5.py --n-boot 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import mutation_type_feature  # noqa: E402

_s136 = PROJECT / "scripts/136_arch21_ft_vs_ft_baseline_lr.py"
# Reuse JOBS / helpers from 136 via import
import importlib.util

_spec = importlib.util.spec_from_file_location("m136", _s136)
_m136 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_m136)

JOBS = _m136.JOBS
load_cadd = _m136.load_cadd
load_split = _m136.load_split
fit_meta = _m136.fit_meta
metrics_row = _m136.metrics_row


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "results/arch21_finetune_table_s5_bootstrap.csv",
    )
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out

    cadd = load_cadd()
    rows = []
    seed = args.seed
    for model, cohort, pilot, tr_p, te_p in JOBS:
        if not tr_p.is_file() or not te_p.is_file():
            print(f"SKIP {model} {cohort}: missing predictions", flush=True)
            continue
        train = load_split(pilot, tr_p, cadd, "train").rename(columns={"prob_ch": "finetune_prob"})
        test = load_split(pilot, te_p, cadd, "test").rename(columns={"prob_ch": "finetune_prob"})
        y = test["y"].values
        p_ft = test["finetune_prob"].values
        p_meta = fit_meta(train, test)

        def add_slice(mask, region: str, setting: str, prob: np.ndarray) -> None:
            nonlocal seed
            ym = y[mask]
            pm = prob[mask]
            base = metrics_row(
                ym, pm, model=model, cohort=cohort, setting=setting, region=region
            )
            point, lo, hi = stratified_bootstrap_auroc(
                ym, pm, n_boot=args.n_boot, seed=seed
            )
            seed += 1
            base["auroc_ci_lo"] = lo
            base["auroc_ci_hi"] = hi
            base["auroc_boot_n"] = args.n_boot
            rows.append(base)

        add_slice(np.ones(len(y), dtype=bool), "overall", "finetune_only", p_ft)
        add_slice(np.ones(len(y), dtype=bool), "overall", "finetune_plus_baseline_lr", p_meta)
        if cohort == "CNC":
            for region in ("coding", "noncoding"):
                mask = test["region_class"].values == region
                add_slice(mask, region, "finetune_only", p_ft)
                add_slice(mask, region, "finetune_plus_baseline_lr", p_meta)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[["model", "cohort", "setting", "region", "n", "n_pos", "auroc", "auroc_ci_lo", "auroc_ci_hi"]].to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
