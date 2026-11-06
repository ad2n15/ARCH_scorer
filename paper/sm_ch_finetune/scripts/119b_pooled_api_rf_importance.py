#!/usr/bin/env python3
"""Table 3 / Fig. 3b: pooled feature importance of the ten AlphaGenome CD34+ CMP track scores.

Rebuilt in Nov 2026 (no script for the original output was kept). Reproduces the stored
pooled_api_rf_impurity_vs_shap.csv exactly (max |diff| ~1e-16).

Random forest (`make_track_classifier("rf", random_state=42)`: 300 trees, max depth 3, balanced
class weights) fitted once on all ARCH21 MANE non-coding variants (n = 71,544), using the cohort
loaded exactly as in 106_cv5_pooled_arch_genes.py (API scores, ANNOVAR, CADD; missing CADD dropped)
and median-imputed track scores only (no gene or annotation terms).
Impurity importance and mean |TreeSHAP| (somatic class) are reported as % of the total.

  cd paper/sm_ch_finetune
  python3 scripts/119b_pooled_api_rf_importance.py
"""
from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
spec = spec_from_file_location("cv5_pooled", PROJECT / "scripts/106_cv5_pooled_arch_genes.py")
m = module_from_spec(spec)
spec.loader.exec_module(m)

OUT = PROJECT / "results/plots/arch21_nc_publish/pooled_api_rf_impurity_vs_shap.csv"
SHORT = {"RNA_SEQ__": "RNA_SEQ", "SPLICE_SITE_USAGE__": "SPLICE_USAGE", "SPLICE_JUNCTIONS__": "SPLICE_JUNC",
         "DNASE__": "DNASE"}


def short(c: str) -> str:
    if c.startswith("CHIP_HISTONE__"):
        return c.split("Histone_ChIP_seq_")[1].split("__")[0]
    return next(v for k, v in SHORT.items() if c.startswith(k))


def load() -> pd.DataFrame:
    rows = []
    for gene in m.ARCH21:
        df = pd.read_csv(PROJECT / "data/arch_mane_noncoding_pooled_sm/by_gene" / f"{gene}.csv")
        df["variant_id"] = df["variant_id"].astype(str)
        df["gene"] = gene
        df["y"] = (df["label"].astype(str).str.lower() == "somatic").astype(int)
        sc = pd.read_parquet(PROJECT / "results/track_scores_arch_mane_noncoding_pooled_sm_by_gene" / gene
                             / "test_features_wide.parquet")
        sc["variant_id"] = sc["variant_id"].astype(str)
        df = df.merge(sc[["variant_id"] + m._api_cols(sc)], on="variant_id", how="inner")
        cadd = pd.read_csv(m._resolve_cadd(gene))
        cadd["variant_id"] = cadd["variant_id"].astype(str)
        cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
        df = df.merge(cadd[["variant_id", "cadd_phred"]], on="variant_id", how="left")
        rows.append(df.loc[df["cadd_phred"].notna()].drop_duplicates("variant_id"))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = load()
    api = [c for c in df.columns if c.startswith(tuple(list(SHORT) + ["CHIP_HISTONE__"]))]
    assert len(api) == 10, api
    X = SimpleImputer(strategy="median").fit_transform(df[api])
    clf = m.make_track_classifier("rf", random_state=42).fit(X, df.y.values)
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(X))
    sv = sv[..., 1] if sv.ndim == 3 else sv[1]
    imp = clf.feature_importances_ / clf.feature_importances_.sum()
    mas = np.abs(sv).mean(0)
    t = pd.DataFrame({"feature": [short(c) for c in api], "impurity_importance": imp, "impurity_pct": 100 * imp,
                      "mean_abs_shap": mas, "shap_pct": 100 * mas / mas.sum(), "mean_shap": sv.mean(0)})
    t["impurity_rank"] = t.impurity_importance.rank(ascending=False).astype(int)
    t["shap_rank"] = t.mean_abs_shap.rank(ascending=False).astype(int)
    t = t.sort_values("shap_rank")[["feature", "impurity_rank", "impurity_importance", "impurity_pct", "shap_rank",
                                    "mean_abs_shap", "shap_pct", "mean_shap"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
