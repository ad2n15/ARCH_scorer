#!/usr/bin/env python3
"""Table 2 — LR bakeoff (CADD vs API vs emb): OOF metrics + 5-fold AUROC mean ± SD.

No retraining. Uses saved OOF from scripts 116 (NC train / NC eval) and 142
(CNC train / coding or noncoding eval). Same CV as bakeoff:
stratify_gene_func + gene onehot, seed=42, n_splits=5.

Outputs:
  results/table2_cadd_api_emb_lr_train_x_eval.csv  (long, for manuscript)
  results/pooled_arch21_cadd_api_emb_lr_wide_train_x_region.csv  (wide legacy)

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/146_table2_cadd_api_emb_lr_fold_sd.py
"""
from __future__ import annotations

import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from eval.classification_metrics import prevalence_matched_table, ranking_metrics  # noqa: E402

_s109 = spec_from_file_location("cv5_lof", PROJECT / "scripts/109_cv5_pooled_lof_emb_baseline.py")
_m109 = module_from_spec(_s109)
assert _s109.loader is not None
_s109.loader.exec_module(_m109)
_make_splits = _m109._make_splits

DEFAULT_NC = (
    "results/pooled_arch21_nc_bakeoff_cadd_api_emb_emb_diff_pca64_"
    "stratify_gene_func_genefeat_onehot/oof_predictions.parquet"
)
DEFAULT_CNC = (
    "results/pooled_arch21_cnc_bakeoff_cadd_api_emb_emb_diff_pca64_"
    "stratify_gene_func_genefeat_onehot/oof_predictions.parquet"
)

ARMS = (
    ("CADD_PHRED", "cadd", "p_cadd_lr"),
    ("AlphaGenome scores (n=10)", "api", "p_api_lr"),
    ("64 PCs compressed embeddings", "emb", "p_emb_lr"),
)

ROWS = (
    ("non-coding", "non-coding", "nc", "all"),
    ("Coding and non-coding", "coding", "cnc", "coding"),
    ("Coding and non-coding", "non-coding", "cnc", "noncoding"),
)


def _fold_auroc_sd(
    df: pd.DataFrame,
    prob_col: str,
    *,
    region: str,
    n_splits: int,
    seed: int,
) -> tuple[float, float, float]:
    y = df["y"].astype(int).to_numpy()
    p = df[prob_col].astype(float).to_numpy()
    if region == "all":
        mask_all = np.ones(len(df), dtype=bool)
    else:
        mask_all = df["region_class"].astype(str).to_numpy() == region
    muts = np.zeros(len(df), dtype=object)
    splits = _make_splits(
        "stratify_gene_func",
        y,
        genes=df["gene"].astype(str).to_numpy(),
        funcs=df["Func.refGene"].astype(str).to_numpy(),
        muts=muts,
        n_splits=n_splits,
        seed=seed,
    )
    fold_au: list[float] = []
    for _tr, te in splits:
        te = te[mask_all[te]]
        if te.size == 0:
            continue
        yt = y[te]
        if yt.min() == yt.max():
            continue
        fold_au.append(float(ranking_metrics(yt, p[te])["auroc"]))
    if len(fold_au) < 2:
        sd = 0.0
    else:
        sd = float(np.std(fold_au, ddof=1))
    mean = float(np.mean(fold_au)) if fold_au else float("nan")
    auroc_oof = float(ranking_metrics(y[mask_all], p[mask_all])["auroc"])
    return mean, sd, auroc_oof


def _pm(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    pm = prevalence_matched_table(y, p)
    return float(pm.get("sensitivity", np.nan)), float(pm.get("specificity", np.nan))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc-oof", default=DEFAULT_NC)
    ap.add_argument("--cnc-oof", default=DEFAULT_CNC)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-long",
        default="results/table2_cadd_api_emb_lr_train_x_eval.csv",
    )
    ap.add_argument(
        "--out-wide",
        default="results/pooled_arch21_cadd_api_emb_lr_wide_train_x_region.csv",
    )
    args = ap.parse_args()

    nc_p = PROJECT / args.nc_oof
    cnc_p = PROJECT / args.cnc_oof
    if not nc_p.is_file():
        raise SystemExit(f"Missing NC OOF (run script 116): {nc_p}")
    if not cnc_p.is_file():
        raise SystemExit(f"Missing CNC OOF (run script 142 --classifiers lr): {cnc_p}")

    nc = pd.read_parquet(nc_p)
    cnc = pd.read_parquet(cnc_p)
    nc["region_class"] = "noncoding"

    data = {"nc": nc, "cnc": cnc}
    long_rows = []

    for train_l, eval_l, cohort, region in ROWS:
        df = data[cohort]
        for feat_label, _arm, pcol in ARMS:
            if pcol not in df.columns:
                raise SystemExit(f"{cohort}: missing column {pcol}")
            sub = df if region == "all" else df.loc[df["region_class"] == region].copy()
            y = sub["y"].astype(int).to_numpy()
            p = sub[pcol].astype(float).to_numpy()
            fold_mean, fold_sd, auroc_oof = _fold_auroc_sd(
                df,
                pcol,
                region=region,
                n_splits=args.n_splits,
                seed=args.seed,
            )
            sens, spec = _pm(y, p)
            fmt = f"{fold_mean:.3f} ± {fold_sd:.3f}"
            long_rows.append(
                {
                    "Feature": feat_label,
                    "Train": train_l,
                    "Eval": eval_l,
                    "AUROC (mean ± SD)": fmt,
                    "auroc_fold_mean": fold_mean,
                    "auroc_fold_sd": fold_sd,
                    "auroc_oof": auroc_oof,
                    "PM_sens": sens,
                    "PM_spec": spec,
                    "n": len(sub),
                    "n_somatic": int(y.sum()),
                }
            )
            print(
                f"{feat_label[:20]:20s} train={train_l[:12]:12s} eval={eval_l:10s} "
                f"AUROC {fmt}  OOF={auroc_oof:.3f}  PM_sens={sens:.3f}",
                flush=True,
            )

    long = pd.DataFrame(long_rows)
    out_long = PROJECT / args.out_long
    out_long.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_long, index=False)

    # Wide table (AUROC only) for legacy merge scripts
    nc_sub = long[(long["Train"] == "non-coding") & (long["Eval"] == "non-coding")]
    cnc_cod = long[(long["Eval"] == "coding")]
    cnc_nc = long[(long["Eval"] == "non-coding") & (long["Train"] == "Coding and non-coding")]
    wide_rows = []
    for feat_label, _a, _p in ARMS:
        r_nc = nc_sub.loc[nc_sub["Feature"] == feat_label].iloc[0]
        r_cc = cnc_cod.loc[cnc_cod["Feature"] == feat_label].iloc[0]
        r_cn = cnc_nc.loc[cnc_nc["Feature"] == feat_label].iloc[0]
        wide_rows.append(
            {
                "feature": feat_label.replace("AlphaGenome scores (n=10)", "API (CMP)").replace(
                    "CADD_PHRED", "CADD"
                ).replace("64 PCs compressed embeddings", "emb PCA64"),
                "NC_train_NC_AUROC": f"{r_nc['auroc_fold_mean']:.3f} ± {r_nc['auroc_fold_sd']:.3f}",
                "NC_train_NC_sens": round(r_nc["PM_sens"], 3),
                "NC_train_NC_spec": round(r_nc["PM_spec"], 3),
                "CNC_train_coding_AUROC": f"{r_cc['auroc_fold_mean']:.3f} ± {r_cc['auroc_fold_sd']:.3f}",
                "CNC_train_coding_sens": round(r_cc["PM_sens"], 3),
                "CNC_train_coding_spec": round(r_cc["PM_spec"], 3),
                "CNC_train_NC_AUROC": f"{r_cn['auroc_fold_mean']:.3f} ± {r_cn['auroc_fold_sd']:.3f}",
                "CNC_train_NC_sens": round(r_cn["PM_sens"], 3),
                "CNC_train_NC_spec": round(r_cn["PM_spec"], 3),
            }
        )
    out_wide = PROJECT / args.out_wide
    pd.DataFrame(wide_rows).to_csv(out_wide, index=False)
    print(f"Wrote {out_long}\nWrote {out_wide}", flush=True)


if __name__ == "__main__":
    main()
