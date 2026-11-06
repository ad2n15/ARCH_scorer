#!/usr/bin/env python
"""Per-gene held-out AUROC for NTv3 heads-only fine-tuning, 21 ARCH genes, germline vs somatic, non-coding.

Models trained on non-coding variants only (the Fig. 5b regime). Replaces the stale v4 Supplementary Table 8,
which came from an earlier NTv3 run (finetune_plus_baseline_per_gene_test.csv, 13 Aug).

Scores are rebuilt from the current predictions with the same code as the overall Fig. 5b numbers:
  - fine-tuned: sm_ch_finetune_ntv3/results/ntv3_arch21_nc_gvs_test_predictions.parquet
  - NTv3 + baseline: meta-LR from sm_ch_finetune/scripts/136_arch21_ft_vs_ft_baseline_lr.py (fit_meta)
  - baseline: the same LR without the fine-tuned probability (CADD-PHRED, SBS6, Func.refGene, gene)
Inputs are only read. Output: per_gene_finetune_ntv3_nc.csv in this folder.

Run: python manuscript_v5_display_items/make_per_gene_finetune_table.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]  # paper/
spec = importlib.util.spec_from_file_location("s136", ROOT / "sm_ch_finetune/scripts/136_arch21_ft_vs_ft_baseline_lr.py")
s136 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s136)

PILOT = ROOT / "sm_ch_finetune/data/pilot_arch21_mane_noncoding_gvs_finetune"
PRED = ROOT / "sm_ch_finetune_ntv3/results/ntv3_arch21_nc_gvs_{}_predictions.parquet"
N_BOOT, MIN_SOM = 2000, 5


def fit_baseline(train, test):
    cats = ["mutation_type", "Func.refGene", "gene"]
    ct = ColumnTransformer([("num", SimpleImputer(strategy="median"), ["cadd_phred"]), ("cat", s136._ohe(), cats)])
    pipe = Pipeline([("ct", ct), ("lr", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs"))])
    pipe.fit(train[["cadd_phred"] + cats], train["y"].values)
    return pipe.predict_proba(test[["cadd_phred"] + cats])[:, 1]


def boot_ci(y, p, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(N_BOOT):  # resample within class so every replicate has both classes
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    cadd = s136.load_cadd()
    train = s136.load_split(PILOT, Path(str(PRED).format("train")), cadd, "train").rename(columns={"prob_ch": "finetune_prob"})
    test = s136.load_split(PILOT, Path(str(PRED).format("test")), cadd, "test").rename(columns={"prob_ch": "finetune_prob"})
    test["p_base"] = fit_baseline(train, test)
    test["p_ens"] = s136.fit_meta(train, test)
    y = test["y"].values
    print(f"overall n={len(test)} somatic={y.sum()}  baseline {roc_auc_score(y, test.p_base):.3f}  "
          f"NTv3 {roc_auc_score(y, test.finetune_prob):.3f}  NTv3+baseline {roc_auc_score(y, test.p_ens):.3f}")

    rng = np.random.default_rng(42)
    rows = []
    for g, s in test.groupby("gene"):
        yy, k = s["y"].values, int(s["y"].sum())
        r = {"gene": g, "n": len(s), "n_somatic": k}
        if 0 < k < len(s):
            r.update(auroc_base=roc_auc_score(yy, s.p_base), auroc_ft=roc_auc_score(yy, s.finetune_prob),
                     auroc_ens=roc_auc_score(yy, s.p_ens), auprc_ens=average_precision_score(yy, s.p_ens),
                     prevalence=k / len(s))
            r["ens_ci_lo"], r["ens_ci_hi"] = boot_ci(yy, s.p_ens.values, rng)
        r["low_n"] = k < MIN_SOM
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("auroc_ens", ascending=False, na_position="last")
    out.to_csv(ROOT / "sm_ch_finetune/results/per_gene_finetune_ntv3_nc.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
