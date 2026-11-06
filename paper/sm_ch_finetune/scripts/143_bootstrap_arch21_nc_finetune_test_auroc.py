#!/usr/bin/env python3
"""Bootstrap 95% CI for held-out test AUROC — 21-gene NC germline vs somatic finetune.

Uses saved test predictions (no re-finetuning). Default: AlphaGenome, NTv3, Borzoi
on pilot_arch21_mane_noncoding_gvs_finetune test split.

Optional --with-baseline: logistic baseline (CADD + SBS6 + Func + gene) on same split.

Example:
  cd paper/sm_ch_finetune
  python3 scripts/143_bootstrap_arch21_nc_finetune_test_auroc.py
  python3 scripts/143_bootstrap_arch21_nc_finetune_test_auroc.py --n-boot 10000 --with-baseline
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
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

PILOT = PROJECT / "data/pilot_arch21_mane_noncoding_gvs_finetune"
ARCH21 = [
    "ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1", "IDH2",
    "JAK2", "KIT", "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1", "SETBP1", "SF3B1",
    "SRSF2", "TET2", "TP53",
]

MODEL_PATHS = {
    "AlphaGenome": PROJECT / "results/finetune_arch21_nc_gvs_test_predictions.parquet",
    "NTv3": ROOT / "sm_ch_finetune_ntv3/results/ntv3_arch21_nc_gvs_test_predictions.parquet",
    "Borzoi": ROOT / "sm_ch_finetune_borzoi/results/borzoi_arch21_nc_gvs_test_predictions.parquet",
}


def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_cadd() -> pd.DataFrame:
    parts = []
    for kind in ("noncoding", "coding"):
        d = PROJECT / f"data/arch_mane_{kind}_pooled_sm/cadd_by_gene"
        for g in ARCH21:
            p = d / f"{g}_cadd_phred_api.csv"
            if p.is_file():
                parts.append(pd.read_csv(p, usecols=["variant_id", "cadd_phred"]))
    c = pd.concat(parts, ignore_index=True)
    c["variant_id"] = c["variant_id"].astype(str)
    c["cadd_phred"] = pd.to_numeric(c["cadd_phred"], errors="coerce")
    return c.dropna(subset=["cadd_phred"]).drop_duplicates("variant_id")


def load_pilot_split(split: str, cadd: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(PILOT / f"{split}.parquet")
    df["variant_id"] = df["variant_id"].astype(str)
    df = df.merge(cadd, on="variant_id", how="left")
    df["mutation_type"] = [
        mutation_type_feature(r, a, "sbs6") for r, a in zip(df["ref"], df["alt"])
    ]
    df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
    df["gene"] = df["gene"].astype(str)
    df["y"] = df["label"].astype(int)
    return df


def fit_baseline_prob(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    cats = ["mutation_type", "Func.refGene", "gene"]
    pipe = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        ("num", SimpleImputer(strategy="median"), ["cadd_phred"]),
                        ("cat", _ohe(), cats),
                    ]
                ),
            ),
            ("lr", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ]
    )
    pipe.fit(train[["cadd_phred"] + cats], train["y"].values)
    return pipe.predict_proba(test[["cadd_phred"] + cats])[:, 1]


def stratified_bootstrap_auroc(
    y: np.ndarray,
    prob: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    y = np.asarray(y, int)
    prob = np.asarray(prob, float)
    point = float(roc_auc_score(y, prob))
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, float)
    for i in range(n_boot):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        ix = np.concatenate([b0, b1])
        if len(np.unique(y[ix])) < 2:
            boots[i] = np.nan
        else:
            boots[i] = roc_auc_score(y[ix], prob[ix])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), float(np.std(boots, ddof=1))


def paired_bootstrap_delta(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Delta = AUROC(a) - AUROC(b) on same resamples."""
    y = np.asarray(y, int)
    obs = roc_auc_score(y, prob_a) - roc_auc_score(y, prob_b)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, float)
    for i in range(n_boot):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        ix = np.concatenate([b0, b1])
        if len(np.unique(y[ix])) < 2:
            diffs[i] = np.nan
        else:
            diffs[i] = roc_auc_score(y[ix], prob_a[ix]) - roc_auc_score(y[ix], prob_b[ix])
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return float(obs), float(lo), float(hi), p_two


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-baseline", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/arch21_nc_finetune_test_bootstrap",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ag = pd.read_parquet(MODEL_PATHS["AlphaGenome"])
    ag["variant_id"] = ag["variant_id"].astype(str)
    y = ag["label"].astype(int).to_numpy()
    n = len(ag)
    n_pos = int(y.sum())
    print(f"Test n={n} somatic={n_pos} germline={n - n_pos}", flush=True)

    scores: dict[str, np.ndarray] = {}
    for name, path in MODEL_PATHS.items():
        if not path.is_file():
            raise SystemExit(f"Missing predictions: {path}")
        pr = pd.read_parquet(path)
        pr["variant_id"] = pr["variant_id"].astype(str)
        merged = ag[["variant_id"]].merge(
            pr[["variant_id", "prob_ch"]], on="variant_id", how="inner", validate="one_to_one"
        )
        if len(merged) != n:
            raise SystemExit(f"{name}: expected {n} rows, got {len(merged)}")
        scores[name] = merged["prob_ch"].to_numpy(float)

    if args.with_baseline:
        cadd = load_cadd()
        train = load_pilot_split("train", cadd)
        test = load_pilot_split("test", cadd)
        test = test.set_index("variant_id").reindex(ag["variant_id"].tolist()).reset_index()
        if test["y"].isna().any():
            raise SystemExit("Baseline test: variant_id mismatch vs finetune predictions")
        scores["baseline_LR"] = fit_baseline_prob(train, test)

    rows = []
    for name, prob in scores.items():
        point, lo, hi, sd = stratified_bootstrap_auroc(
            y, prob, n_boot=args.n_boot, seed=args.seed
        )
        rows.append(
            {
                "model": name,
                "n": n,
                "n_somatic": n_pos,
                "auroc": point,
                "auroc_ci_lo": lo,
                "auroc_ci_hi": hi,
                "auroc_boot_sd": sd,
                "n_bootstrap": args.n_boot,
            }
        )
        print(
            f"{name:14s} AUROC {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]",
            flush=True,
        )

    ci_path = out_dir / "finetune_test_auroc_bootstrap.csv"
    pd.DataFrame(rows).to_csv(ci_path, index=False)

    pair_rows = []
    for a, b in combinations(scores.keys(), 2):
        obs, lo, hi, p = paired_bootstrap_delta(
            y, scores[a], scores[b], n_boot=args.n_boot, seed=args.seed + 1
        )
        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "delta_auroc_a_minus_b": obs,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "p_bootstrap_two_sided": p,
            }
        )
        print(
            f"ΔAUROC {a} − {b}: {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p≈{p:.4g}",
            flush=True,
        )
    pair_path = out_dir / "finetune_test_auroc_pairwise_bootstrap.csv"
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    print(f"Wrote {ci_path}\nWrote {pair_path}", flush=True)


if __name__ == "__main__":
    main()
