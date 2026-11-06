#!/usr/bin/env python3
"""CADD PHRED for pooled-SM cohort genes from the local CADD v1.7 whole-genome TSV.

Drop-in replacement for the CADD web API fetch (scripts/50_fetch_tp53_cadd_api.py):
writes the same schema to data/arch_mane_noncoding_pooled_sm/cadd_by_gene/.

One tabix query per gene span, then match wanted (pos, ref, alt) triples.

Example:
  cd paper/sm_ch_finetune
  python3 scripts/86_cadd_from_local_tsv_by_gene.py --gene TP53
  python3 scripts/86_cadd_from_local_tsv_by_gene.py --all
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent

COHORT_DIR = PROJECT / "data/arch_mane_noncoding_pooled_sm/by_gene"
OUT_DIR = PROJECT / "data/arch_mane_noncoding_pooled_sm/cadd_by_gene"
CADD_TSV = ROOT / "cadd_v1.7_GRCh38/whole_genome_SNVs.tsv.gz"
TABIX = __import__("os").environ.get("TABIX", "tabix")
CADD_VERSION = "GRCh38-v1.7"

OUT_COLS = [
    "variant_id",
    "source",
    "chrom",
    "pos",
    "ref",
    "alt",
    "cadd_phred",
    "cadd_raw",
    "cadd_status",
    "cadd_version",
]


def _lookup(chrom: str, lo: int, hi: int, want: set[tuple[int, str, str]]) -> dict:
    """Stream one tabix span; keep only wanted (pos, ref, alt)."""
    found: dict[tuple[int, str, str], tuple[float, float]] = {}
    for contig in (chrom, f"chr{chrom}"):
        proc = subprocess.Popen(
            [TABIX, str(CADD_TSV), f"{contig}:{lo}-{hi}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1 << 20,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            key = (int(parts[1]), parts[2], parts[3])
            if key in want:
                try:
                    found[key] = (float(parts[5]), float(parts[4]))
                except ValueError:
                    continue
        proc.wait()
        if found:
            break
    return found


def cadd_one(gene: str) -> Path:
    cohort_path = COHORT_DIR / f"{gene}.csv"
    if not cohort_path.is_file():
        raise SystemExit(f"Missing cohort: {cohort_path}")
    df = pd.read_csv(cohort_path)
    df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["pos"] = df["pos"].astype(int)

    rows: list[dict] = []
    for chrom, grp in df.groupby("chrom", sort=True):
        want = {
            (int(r.pos), str(r.ref).upper(), str(r.alt).upper())
            for r in grp.itertuples(index=False)
        }
        lo, hi = int(grp["pos"].min()), int(grp["pos"].max())
        print(
            f"  {gene} chr{chrom}:{lo}-{hi} alleles={len(want)} …",
            flush=True,
        )
        found = _lookup(str(chrom), lo, hi, want)
        for r in grp.itertuples(index=False):
            key = (int(r.pos), str(r.ref).upper(), str(r.alt).upper())
            hit = found.get(key)
            rows.append(
                {
                    "variant_id": str(r.variant_id),
                    "source": getattr(r, "source", None),
                    "chrom": str(chrom),
                    "pos": int(r.pos),
                    "ref": r.ref,
                    "alt": r.alt,
                    "cadd_phred": hit[0] if hit else None,
                    "cadd_raw": hit[1] if hit else None,
                    "cadd_status": "ok" if hit else "not_found",
                    "cadd_version": CADD_VERSION,
                }
            )

    out = pd.DataFrame(rows, columns=OUT_COLS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{gene}_cadd_phred_local.csv"
    out.to_csv(out_path, index=False)
    n_ok = int((out["cadd_status"] == "ok").sum())
    print(f"{gene}: {n_ok}/{len(out)} scored -> {out_path}", flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene")
    ap.add_argument("--genes", help="comma-separated")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not CADD_TSV.is_file():
        raise SystemExit(f"Missing CADD TSV: {CADD_TSV}")
    if not Path(f"{CADD_TSV}.tbi").is_file():
        raise SystemExit(f"Missing CADD index: {CADD_TSV}.tbi")

    if args.all:
        genes = sorted(p.stem for p in COHORT_DIR.glob("*.csv"))
    elif args.genes:
        genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    elif args.gene:
        genes = [args.gene.strip().upper()]
    else:
        raise SystemExit("Set --gene, --genes or --all")

    for g in genes:
        cadd_one(g)


if __name__ == "__main__":
    main()
