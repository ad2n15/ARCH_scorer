#!/usr/bin/env python3
"""Baseline-only LR on ARCH21 NC and CNC fine-tune splits (no sequence head).

Features: CADD PHRED + SBS6 + Func.refGene + gene onehot (same block as 136).
CNC also reports coding vs noncoding test slices (PM recomputed on the slice).

CPU-only:

  python sm_ch_finetune/scripts/137_arch21_baseline_only_lr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import mutation_type_feature  # noqa: E402
from eval.classification_metrics import prevalence_matched_table, ranking_metrics  # noqa: E402

ARCH21 = [
    "ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1", "IDH2",
    "JAK2", "KIT", "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1", "SETBP1", "SF3B1",
    "SRSF2", "TET2", "TP53",
]
OUT = PROJECT / "results/arch21_baseline_only_lr_summary.csv"

PILOTS = {
    "NC": PROJECT / "data/pilot_arch21_mane_noncoding_gvs_finetune",
    "CNC": PROJECT / "data/pilot_arch21_mane_cnc_gvs_finetune",
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
            if p.exists():
                parts.append(pd.read_csv(p, usecols=["variant_id", "cadd_phred"]))
    c = pd.concat(parts, ignore_index=True)
    c["variant_id"] = c["variant_id"].astype(str)
    c["cadd_phred"] = pd.to_numeric(c["cadd_phred"], errors="coerce")
    return c.dropna(subset=["cadd_phred"]).drop_duplicates("variant_id")


def load_split(pilot: Path, cadd: pd.DataFrame, split: str) -> pd.DataFrame:
    df = pd.read_parquet(pilot / f"{split}.parquet")
    df["variant_id"] = df["variant_id"].astype(str)
    df = df.merge(cadd, on="variant_id", how="left")
    df["mutation_type"] = [
        mutation_type_feature(r, a, "sbs6") for r, a in zip(df["ref"], df["alt"])
    ]
    df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
    df["gene"] = df["gene"].astype(str)
    df["y"] = df["label"].astype(int)
    df["region_class"] = df["region_class"].astype(str)
    return df


def metrics_row(y, p, *, cohort, region) -> dict:
    m = ranking_metrics(y, p)
    pm = prevalence_matched_table(y, p)
    return {
        "model": "baseline_lr",
        "cohort": cohort,
        "setting": "baseline_only",
        "region": region,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": float(m["auroc"]),
        "auprc": float(m["auprc"]),
        "pm_threshold": float(pm["threshold"]),
        "pm_sensitivity": float(pm["sensitivity"]),
        "pm_specificity": float(pm["specificity"]),
    }


def fit_baseline(train: pd.DataFrame, test: pd.DataFrame):
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
    X_tr = train[["cadd_phred"] + cats]
    X_te = test[["cadd_phred"] + cats]
    pipe.fit(X_tr, train["y"].values)
    return pipe.predict_proba(X_te)[:, 1]


def main() -> None:
    cadd = load_cadd()
    print(f"CADD map n={len(cadd)}")
    rows = []
    for cohort, pilot in PILOTS.items():
        train = load_split(pilot, cadd, "train")
        test = load_split(pilot, cadd, "test")
        print(
            f"{cohort}: train={len(train)} test={len(test)} "
            f"cadd_na_te={int(test.cadd_phred.isna().sum())}"
        )
        p = fit_baseline(train, test)
        y = test["y"].values
        rows.append(metrics_row(y, p, cohort=cohort, region="overall"))
        if cohort == "CNC":
            for region in ("coding", "noncoding"):
                mask = test["region_class"].values == region
                rows.append(metrics_row(y[mask], p[mask], cohort=cohort, region=region))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
