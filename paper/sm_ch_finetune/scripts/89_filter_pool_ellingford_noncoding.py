#!/usr/bin/env python3
"""Filter pooled MANE-noncoding cohort to Ellingford noncoding definition.

Drops alleles in MANE Select CDS, including the ±2 coding-exon splice flanks.
Keeps intronic canonical splice (±2 into the intron). Rewrites pool CSVs,
by-gene CSVs, pilots, and avinput.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.mane_ellingford import load_mane_cds_exons, violates_ellingford_noncoding

POOL = PROJECT / "data/arch_mane_noncoding_pooled_sm"
GTF = PROJECT / "data/reference/gencode.v46.annotation.gtf"
COMB = POOL / "arch_mane_noncoding_germline_plus_somatic_pooled.csv"
PILOT_ROOT = PROJECT / "data/pilot_arch_mane_noncoding_pooled_sm"
BY_GENE_PILOT = PROJECT / "data/pilot_arch_mane_noncoding_pooled_sm_by_gene"
BY_GENE_CSV = POOL / "by_gene"


def main() -> None:
    comb = pd.read_csv(COMB, low_memory=False)
    n0 = len(comb)
    genes = set(comb["gene"].astype(str))
    cds, _exons = load_mane_cds_exons(GTF, genes)

    drop_mask = [
        violates_ellingford_noncoding(c, int(p), g, cds)
        for c, p, g in zip(comb["chrom"], comb["pos"], comb["gene"])
    ]
    drop_mask = np.asarray(drop_mask, dtype=bool)
    dropped = comb.loc[drop_mask].copy()
    keep = comb.loc[~drop_mask].copy()

    print(
        f"Pool {n0:,} → {len(keep):,}  dropped_CDS_or_exonic_splice_flank={int(drop_mask.sum()):,}",
        flush=True,
    )
    if len(dropped):
        print("Dropped by Func/region unavailable here; variant_ids sample:")
        print(dropped[["gene", "label", "mane_region", "variant_id"]].head(20).to_string(index=False))

    germ = keep.loc[keep["label"].astype(str) == "germline"].copy()
    som = keep.loc[keep["label"].astype(str) == "somatic"].copy()

    germ.to_csv(POOL / "arch_mane_noncoding_germline_abn_ge5.csv", index=False)
    som.to_csv(POOL / "arch_mane_noncoding_somatic_pooled.csv", index=False)
    keep.to_csv(COMB, index=False)

    with open(POOL / "arch_mane_noncoding_pooled.avinput", "w") as f:
        for r in keep.itertuples(index=False):
            c = str(r.chrom).replace("chr", "")
            f.write(f"{c}\t{int(r.pos)}\t{int(r.pos)}\t{r.ref}\t{r.alt}\n")

    pdf = keep.copy()
    pdf["chrom"] = ["chr" + str(c).replace("chr", "") for c in pdf["chrom"]]
    pdf["label_str"] = pdf["label"].astype(str)
    pdf["label"] = (pdf["label_str"] == "somatic").astype(np.int64)
    pdf["split"] = "test"
    pilot_cols = [
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
    for c in ("n_haem", "haem_ge3"):
        if c not in pdf.columns:
            pdf[c] = 0 if c == "haem_ge3" else np.nan
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    pdf[pilot_cols].to_parquet(PILOT_ROOT / "test.parquet", index=False)

    BY_GENE_PILOT.mkdir(parents=True, exist_ok=True)
    BY_GENE_CSV.mkdir(parents=True, exist_ok=True)
    for gene, g in keep.groupby("gene"):
        d = BY_GENE_PILOT / gene
        d.mkdir(exist_ok=True)
        gp = pdf.loc[pdf.gene == gene]
        gp[pilot_cols].to_parquet(d / "test.parquet", index=False)
        g.to_csv(BY_GENE_CSV / f"{gene}.csv", index=False)

    by_gene = (
        keep.groupby("gene")
        .agg(
            germline=("label", lambda s: int((s.astype(str) == "germline").sum())),
            somatic_total=("label", lambda s: int((s.astype(str) == "somatic").sum())),
            all_total=("label", "size"),
        )
        .reset_index()
    )
    by_gene.to_csv(POOL / "variant_counts_by_gene_wes_wgs.csv", index=False)

    summary = {
        "definition": (
            "Ellingford et al. Genome Med 2022: noncoding = outside MANE CDS; "
            "canonical splice kept only as intronic ±2; coding-exon ±2 excluded"
        ),
        "n_before": int(n0),
        "n_after": int(len(keep)),
        "n_dropped": int(drop_mask.sum()),
        "n_germline": int(len(germ)),
        "n_somatic": int(len(som)),
        "dropped_variant_ids": dropped["variant_id"].astype(str).tolist(),
    }
    (POOL / "ellingford_noncoding_filter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Wrote {COMB}")
    print(f"Wrote {POOL / 'ellingford_noncoding_filter_summary.json'}")


if __name__ == "__main__":
    main()
