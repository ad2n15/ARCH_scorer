#!/usr/bin/env python3
"""Write a pilot test.parquet (API-scoring input) from a by_gene cohort CSV.

Mirrors the pilot columns produced by ``76_rebuild_pool_wes_wgs_paper_mw.py``.

Example (TP53 coding):
  cd paper/sm_ch_finetune
  python3 scripts/111_make_pilot_parquet_from_cohort.py --gene TP53 \\
      --cohort-dir data/arch_mane_coding_pooled_sm/by_gene \\
      --pilot-dir data/pilot_arch_mane_coding_pooled_sm_by_gene
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from paths import project_subpath

PILOT_COLS = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "variant_id",
    "gene",
    "label",
    "label_str",
    "split",
    "mane_region",
    "source",
    "n_haem",
    "haem_ge3",
]


def _write_pilot(gene: str, cohort_dir: Path, pilot_dir: Path) -> None:
    src = cohort_dir / f"{gene}.csv"
    if not src.is_file():
        raise SystemExit(f"Missing cohort: {src}")
    out_dir = pilot_dir / gene
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df["chrom"] = ["chr" + str(c).replace("chr", "") for c in df["chrom"]]
    df["label_str"] = df["label"].astype(str)
    df["label"] = (df["label_str"] == "somatic").astype("int64")
    df["split"] = "test"
    for c in ("n_haem", "haem_ge3", "mane_region", "source"):
        if c not in df.columns:
            df[c] = None

    out = out_dir / "test.parquet"
    df[PILOT_COLS].to_parquet(out, index=False)
    print(
        f"{gene}: wrote {out} n={len(df)} "
        f"somatic={int(df['label'].sum())} germline={int((df['label'] == 0).sum())}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--gene", default="TP53")
    group.add_argument("--genes", help="Comma-separated genes")
    group.add_argument("--all", action="store_true", help="Use all CSVs in cohort-dir")
    ap.add_argument(
        "--cohort-dir",
        default="data/arch_mane_coding_pooled_sm/by_gene",
    )
    ap.add_argument(
        "--pilot-dir",
        default="data/pilot_arch_mane_coding_pooled_sm_by_gene",
    )
    args = ap.parse_args()

    cohort_dir = project_subpath(PROJECT, args.cohort_dir)
    pilot_dir = project_subpath(PROJECT, args.pilot_dir)
    if args.all:
        genes = sorted(path.stem.upper() for path in cohort_dir.glob("*.csv"))
    elif args.genes:
        genes = [gene.strip().upper() for gene in args.genes.split(",") if gene.strip()]
    else:
        genes = [args.gene.strip().upper()]
    if not genes:
        raise SystemExit(f"No cohort CSVs found in {cohort_dir}")
    for gene in genes:
        _write_pilot(gene, cohort_dir, pilot_dir)


if __name__ == "__main__":
    main()
