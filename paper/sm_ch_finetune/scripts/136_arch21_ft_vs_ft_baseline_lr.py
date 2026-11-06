#!/usr/bin/env python3
"""ARCH21 NC/CNC: finetune-only vs finetune+baseline meta-LR (AG / NTv3 / Borzoi).

Baseline block: CADD PHRED + SBS6 + Func.refGene + gene onehot (same spirit as 08).
CNC also reports coding vs noncoding test slices (PM recomputed on the slice).

  python sm_ch_finetune/scripts/136_arch21_ft_vs_ft_baseline_lr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import mutation_type_feature  # noqa: E402
from eval.classification_metrics import prevalence_matched_table, ranking_metrics  # noqa: E402

ARCH21 = [
    "ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1", "IDH2",
    "JAK2", "KIT", "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1", "SETBP1", "SF3B1",
    "SRSF2", "TET2", "TP53",
]
OUT = PROJECT / "results/arch21_ft_vs_ft_baseline_lr_summary.csv"

JOBS = [
    ("AG", "NC", PROJECT / "data/pilot_arch21_mane_noncoding_gvs_finetune",
     PROJECT / "results/finetune_arch21_nc_gvs_train_predictions.parquet",
     PROJECT / "results/finetune_arch21_nc_gvs_test_predictions.parquet"),
    ("AG", "CNC", PROJECT / "data/pilot_arch21_mane_cnc_gvs_finetune",
     PROJECT / "results/finetune_arch21_cnc_gvs_train_predictions.parquet",
     PROJECT / "results/finetune_arch21_cnc_gvs_test_predictions.parquet"),
    ("NTv3", "NC", PROJECT / "data/pilot_arch21_mane_noncoding_gvs_finetune",
     ROOT / "sm_ch_finetune_ntv3/results/ntv3_arch21_nc_gvs_train_predictions.parquet",
     ROOT / "sm_ch_finetune_ntv3/results/ntv3_arch21_nc_gvs_test_predictions.parquet"),
    ("NTv3", "CNC", PROJECT / "data/pilot_arch21_mane_cnc_gvs_finetune",
     ROOT / "sm_ch_finetune_ntv3/results/ntv3_arch21_cnc_gvs_train_predictions.parquet",
     ROOT / "sm_ch_finetune_ntv3/results/ntv3_arch21_cnc_gvs_test_predictions.parquet"),
    ("Borzoi", "NC", PROJECT / "data/pilot_arch21_mane_noncoding_gvs_finetune",
     ROOT / "sm_ch_finetune_borzoi/results/borzoi_arch21_nc_gvs_train_predictions.parquet",
     ROOT / "sm_ch_finetune_borzoi/results/borzoi_arch21_nc_gvs_test_predictions.parquet"),
    ("Borzoi", "CNC", PROJECT / "data/pilot_arch21_mane_cnc_gvs_finetune",
     ROOT / "sm_ch_finetune_borzoi/results/borzoi_arch21_cnc_gvs_train_predictions.parquet",
     ROOT / "sm_ch_finetune_borzoi/results/borzoi_arch21_cnc_gvs_test_predictions.parquet"),
]


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
            if p.exists():
                parts.append(pd.read_csv(p, usecols=["variant_id", "cadd_phred"]))
    c = pd.concat(parts, ignore_index=True)
    c["variant_id"] = c["variant_id"].astype(str)
    c["cadd_phred"] = pd.to_numeric(c["cadd_phred"], errors="coerce")
    return c.dropna(subset=["cadd_phred"]).drop_duplicates("variant_id")


def load_split(pilot: Path, pred: Path, cadd: pd.DataFrame, split: str) -> pd.DataFrame:
    meta = pd.read_parquet(pilot / f"{split}.parquet")
    meta["variant_id"] = meta["variant_id"].astype(str)
    pr = pd.read_parquet(pred)
    pr["variant_id"] = pr["variant_id"].astype(str)
    df = meta.merge(pr[["variant_id", "prob_ch"]], on="variant_id", how="inner")
    df = df.merge(cadd, on="variant_id", how="left")
    df["mutation_type"] = [
        mutation_type_feature(r, a, "sbs6") for r, a in zip(df["ref"], df["alt"])
    ]
    df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
    df["gene"] = df["gene"].astype(str)
    df["y"] = df["label"].astype(int)
    df["region_class"] = df["region_class"].astype(str)
    return df


def metrics_row(y, p, *, model, cohort, setting, region) -> dict:
    m = ranking_metrics(y, p)
    pm = prevalence_matched_table(y, p)
    return {
        "model": model,
        "cohort": cohort,
        "setting": setting,
        "region": region,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": float(m["auroc"]),
        "auprc": float(m["auprc"]),
        "pm_threshold": float(pm["threshold"]),
        "pm_sensitivity": float(pm["sensitivity"]),
        "pm_specificity": float(pm["specificity"]),
    }


def fit_meta(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    cats = ["mutation_type", "Func.refGene", "gene"]
    ct = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), ["finetune_prob", "cadd_phred"]),
            ("cat", _ohe(), cats),
        ]
    )
    pipe = Pipeline(
        [
            ("ct", ct),
            ("lr", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")),
        ]
    )
    X_tr = train[["finetune_prob", "cadd_phred"] + cats]
    X_te = test[["finetune_prob", "cadd_phred"] + cats]
    pipe.fit(X_tr, train["y"].values)
    return pipe.predict_proba(X_te)[:, 1]


def main() -> None:
    cadd = load_cadd()
    print(f"CADD map n={len(cadd)}")
    rows = []
    payloads = {}
    for model, cohort, pilot, tr_p, te_p in JOBS:
        if not tr_p.exists() or not te_p.exists():
            print(f"SKIP {model} {cohort}: missing {tr_p.name if not tr_p.exists() else te_p.name}")
            continue
        train = load_split(pilot, tr_p, cadd, "train").rename(columns={"prob_ch": "finetune_prob"})
        test = load_split(pilot, te_p, cadd, "test").rename(columns={"prob_ch": "finetune_prob"})
        print(
            f"{model} {cohort}: train={len(train)} test={len(test)} "
            f"cadd_na_te={test.cadd_phred.isna().sum()}"
        )
        y = test["y"].values
        p_ft = test["finetune_prob"].values
        rows.append(metrics_row(y, p_ft, model=model, cohort=cohort, setting="finetune_only", region="overall"))
        p_meta = fit_meta(train, test)
        rows.append(metrics_row(y, p_meta, model=model, cohort=cohort, setting="finetune_plus_baseline_lr", region="overall"))
        if cohort == "CNC":
            for region in ("coding", "noncoding"):
                mask = test["region_class"].values == region
                rows.append(metrics_row(
                    y[mask], p_ft[mask], model=model, cohort=cohort,
                    setting="finetune_only", region=region,
                ))
                rows.append(metrics_row(
                    y[mask], p_meta[mask], model=model, cohort=cohort,
                    setting="finetune_plus_baseline_lr", region=region,
                ))
        payloads[f"{model}_{cohort}"] = {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "cadd_missing_test": int(test.cadd_phred.isna().sum()),
        }
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    (OUT.with_suffix(".meta.json")).write_text(json.dumps(payloads, indent=2) + "\n")
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
