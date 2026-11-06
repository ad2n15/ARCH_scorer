#!/usr/bin/env python3
"""Maintainers: build the variant-level tables released on Zenodo (release_data/, git-ignored).

Only data derived from public sources are released:
  genomewide_age_filtered.csv.gz   Colin-Leitzinger et al. 2025 (Human Genomics, CC BY 4.0) age-filtered SNVs:
                                   label (ARCH = 1), split, annotations, held-out scores of the 3 fine-tuned models
  arch21_noncoding_public.csv.gz   21 ARCH genes, MANE non-coding (n = 71,544 analysed): every germline row (gnomAD v4)
  arch21_coding_public.csv.gz      and every somatic row with a gnomAD or IARC source, with CADD-PHRED, ANNOVAR
                                   Func.refGene, the ten AlphaGenome CD34+ CMP scores and the fine-tuning split.
Somatic rows supported only by COSMIC (licensed) are omitted, and COSMIC tokens are removed from `source`.
Licensed users rebuild the full cohort with paper/sm_ch_finetune/scripts (61, 81, 110 read COSMIC v104).

  ALPHAGENOME_ROOT=/path/to/alphagenome python3 scripts/export_public_data.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ALPHAGENOME_ROOT", HERE.parent))  # folder holding sm_ch_finetune*/ projects
SM = ROOT / "sm_ch_finetune"
OUT = HERE / "release_data"
PUBLIC = ("gnomad", "paper_mw", "iarc")
PRED = {"alphagenome": SM / "results/finetune_{}_test_predictions.parquet",
        "ntv3": ROOT / "sm_ch_finetune_ntv3/results/ntv3_{}_test_predictions.parquet",
        "borzoi": ROOT / "sm_ch_finetune_borzoi/results/borzoi_{}_test_predictions.parquet"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def public_source(s: str) -> str:
    return "+".join(t for t in str(s).split("+") if t.startswith(PUBLIC))


def test_scores(key: str) -> pd.DataFrame:
    out = None
    for b, pat in PRED.items():
        p = pd.read_parquet(str(pat).format(key))[["variant_id", "prob_ch"]].rename(columns={"prob_ch": f"test_score_{b}"})
        p["variant_id"] = p["variant_id"].astype(str)
        out = p if out is None else out.merge(p, on="variant_id", how="outer")
    return out


def arch21(kind: str, t1) -> pd.DataFrame:
    df = t1.arm(kind)
    df["source"] = df["source"].map(public_source)
    keep = (df["label"] == "germline") | (df["source"] != "")
    df = df.loc[keep]
    pilot = SM / ("data/pilot_arch21_mane_noncoding_gvs_finetune" if kind == "noncoding"
                  else "data/pilot_arch21_mane_cnc_gvs_finetune")
    split = pd.concat([pd.read_parquet(pilot / f"{s}.parquet")[["variant_id"]].assign(finetune_split=s)
                       for s in ("train", "valid", "test")])
    split["variant_id"] = split["variant_id"].astype(str)
    df = df.merge(split, on="variant_id", how="left")
    if kind == "noncoding":
        df = df.merge(test_scores("arch21_nc_gvs").rename(columns=lambda c: c.replace("test_score", "test_score_nc")),
                      on="variant_id", how="left")
    cols = ["variant_id", "chrom", "pos", "ref", "alt", "gene", "label", "source", "mane_region", "Func.refGene",
            "cadd_phred", *t1.FEATS, "finetune_split"] + [c for c in df.columns if c.startswith("test_score")]
    return df[cols].sort_values(["chrom", "pos", "alt"]).reset_index(drop=True)


def genomewide() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(SM / f"data/processed_age_filtered/{s}.parquet") for s in ("train", "valid", "test")])
    d["variant_id"] = d["variant_id"].astype(str)
    d = d.merge(test_scores("full"), on="variant_id", how="left")
    cols = ["variant_id", "chrom", "pos", "ref", "alt", "label_str", "label", "split", "gene", "func", "exonic_func",
            "cadd", "mutation_type", "mchip", "lchip"] + [c for c in d.columns if c.startswith("test_score")]
    return d[cols].rename(columns={"label_str": "class"}).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t1 = _load(HERE / "paper/sm_ch_finetune/scripts/119_table1_covariate_adjusted_or.py", "t1")
    t1.PROJECT = SM
    t1.U = _load(SM / "scripts/117_univariate_pooled_coding_noncoding.py", "u117")
    t1.P = t1.U._pub
    summary = {}
    for name, df in [("arch21_noncoding_public", arch21("noncoding", t1)), ("arch21_coding_public", arch21("coding", t1)),
                     ("genomewide_age_filtered", genomewide())]:
        df.to_csv(OUT / f"{name}.csv.gz", index=False)
        lab = df["label"].astype(str)
        summary[name] = {"rows": len(df), **lab.value_counts().to_dict()}
        print(name, summary[name], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
