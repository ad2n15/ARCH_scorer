#!/usr/bin/env python3
"""Extract Cosmic MutantCensus SNVs that fall in ARCH MANE *noncoding* windows.

Same strategy as gnomAD WGS∪WES: another Cosmic catalogue, same genomic
noncoding definition + haem filters.

Default somatic Cosmic rule (match current pooled NCV confirmed set):
  n_haem >= 1 AND n_haem_confirmed >= 1
  (haem = haematopoietic_and_lymphoid_tissue;
   confirmed = "Confirmed somatic variant")

Example:
  cd paper/sm_ch_finetune
  python3 scripts/81_extract_cosmic_mutantcensus_mane_noncoding.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
HAEM_SITE = "haematopoietic_and_lymphoid_tissue"
CONFIRMED = "Confirmed somatic variant"


def _load_m61():
    path = PROJECT / "scripts/61_build_arch_mane_noncoding_lists.py"
    spec = importlib.util.spec_from_file_location("m61", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mutations",
        default=str(ROOT / "Cosmic_MutantCensus_v104_GRCh38.tsv"),
    )
    p.add_argument(
        "--classification",
        default=str(ROOT / "Cosmic_Classification_v104_GRCh38.tsv"),
    )
    p.add_argument("--gtf", default=str(PROJECT / "data/reference/gencode.v46.annotation.gtf"))
    p.add_argument("--min-haem", type=int, default=1)
    p.add_argument("--min-haem-confirmed", type=int, default=1)
    p.add_argument(
        "--out-dir",
        default=str(PROJECT / "data/arch_mane_noncoding"),
    )
    args = p.parse_args()

    m61 = _load_m61()
    gtf = Path(args.gtf)
    mane = m61.load_mane_select_map(gtf, m61.ARCH21)
    gene_ivs = {g: m61.load_mane_noncoding_intervals(gtf, mane[g]) for g in m61.ARCH21}
    by_chrom: dict[str, list] = {}
    for g, iv in gene_ivs.items():
        by_chrom.setdefault(str(iv["chrom"]), []).append(
            (iv["window"][0], iv["window"][1], g, iv)
        )

    clf = pd.read_csv(args.classification, sep="\t", low_memory=False)
    haem_pheno = set(
        clf.loc[clf["PRIMARY_SITE"].astype(str) == HAEM_SITE, "COSMIC_PHENOTYPE_ID"].astype(str)
    )
    print(f"haem phenotype IDs: {len(haem_pheno)}", flush=True)

    usecols = [
        "GENE_SYMBOL",
        "COSMIC_PHENOTYPE_ID",
        "GENOMIC_MUTATION_ID",
        "CHROMOSOME",
        "GENOME_START",
        "GENOMIC_WT_ALLELE",
        "GENOMIC_MUT_ALLELE",
        "HGVSG",
        "MUTATION_SOMATIC_STATUS",
        "MUTATION_DESCRIPTION",
    ]
    kept: list[pd.DataFrame] = []
    n_raw = 0
    for i, chunk in enumerate(
        pd.read_csv(
            args.mutations,
            sep="\t",
            usecols=lambda c: c in set(usecols),
            chunksize=500_000,
            dtype=str,
            low_memory=True,
        ),
        1,
    ):
        n_raw += len(chunk)
        chunk["chrom"] = chunk["CHROMOSOME"].map(m61._chrom_str)
        chunk = chunk.loc[chunk["chrom"].isin(by_chrom)].copy()
        if chunk.empty:
            continue
        chunk["pos"] = pd.to_numeric(chunk["GENOME_START"], errors="coerce")
        chunk = chunk.loc[chunk["pos"].notna()].copy()
        chunk["pos"] = chunk["pos"].astype(int)
        chunk["ref"] = chunk["GENOMIC_WT_ALLELE"].astype(str).str.upper()
        chunk["alt"] = chunk["GENOMIC_MUT_ALLELE"].astype(str).str.upper()
        chunk = chunk.loc[
            (chunk["ref"].str.len() == 1)
            & (chunk["alt"].str.len() == 1)
            & chunk["ref"].isin(list("ACGT"))
            & chunk["alt"].isin(list("ACGT"))
        ].copy()
        if chunk.empty:
            continue

        genes, regions, keep_idx = [], [], []
        for idx, chrom, pos in zip(chunk.index, chunk["chrom"], chunk["pos"]):
            hit = None
            for lo, hi, g, iv in by_chrom.get(str(chrom), []):
                if lo <= pos <= hi:
                    reg = m61.classify_pos(int(pos), iv)
                    if reg is not None:
                        hit = (g, reg)
                        break
            if hit is None:
                continue
            keep_idx.append(idx)
            genes.append(hit[0])
            regions.append(hit[1])
        if not keep_idx:
            continue
        sub = chunk.loc[keep_idx].copy()
        sub["gene"] = genes
        sub["mane_region"] = regions
        sub["_haem"] = sub["COSMIC_PHENOTYPE_ID"].astype(str).isin(haem_pheno)
        sub["_confirmed"] = sub["MUTATION_SOMATIC_STATUS"].astype(str) == CONFIRMED
        kept.append(sub)
        if i % 5 == 0:
            print(f"  chunk {i}: scanned {n_raw:,} kept_window_rows={sum(len(x) for x in kept):,}", flush=True)

    if not kept:
        raise SystemExit("No MutantCensus rows in MANE noncoding windows")
    win = pd.concat(kept, ignore_index=True)
    print(f"scanned={n_raw:,} MANE-noncoding SNV rows={len(win):,}", flush=True)

    haem = win.loc[win["_haem"]].copy()
    n_haem = haem.groupby("GENOMIC_MUTATION_ID", dropna=False).size().rename("n_haem")
    n_haem_conf = (
        haem.loc[haem["_confirmed"]]
        .groupby("GENOMIC_MUTATION_ID", dropna=False)
        .size()
        .rename("n_haem_confirmed")
    )
    stats = pd.concat([n_haem, n_haem_conf], axis=1).fillna(0).astype(int)
    keep_ids = stats.index[
        (stats["n_haem"] >= args.min_haem)
        & (stats["n_haem_confirmed"] >= args.min_haem_confirmed)
    ]
    print(
        f"IDs with n_haem>={args.min_haem} & n_haem_confirmed>={args.min_haem_confirmed}: "
        f"{len(keep_ids)}",
        flush=True,
    )

    sub = haem.loc[haem["GENOMIC_MUTATION_ID"].isin(keep_ids)].copy()
    sub = sub.merge(stats, left_on="GENOMIC_MUTATION_ID", right_index=True, how="left")
    sub["variant_id"] = (
        sub["chrom"].astype(str)
        + "-"
        + sub["pos"].astype(str)
        + "-"
        + sub["ref"]
        + "-"
        + sub["alt"]
    )
    sub["_n"] = 1
    gcols = {
        "chrom": ("chrom", "first"),
        "pos": ("pos", "first"),
        "ref": ("ref", "first"),
        "alt": ("alt", "first"),
        "gene": ("gene", "first"),
        "mane_region": ("mane_region", "first"),
        "n_cosmic_rows": ("_n", "sum"),
        "n_haem": ("n_haem", "max"),
        "n_haem_confirmed": ("n_haem_confirmed", "max"),
        "GENOMIC_MUTATION_ID": ("GENOMIC_MUTATION_ID", "first"),
        "HGVSG": ("HGVSG", "first"),
        "GENE_SYMBOL": ("GENE_SYMBOL", "first"),
        "MUTATION_DESCRIPTION": ("MUTATION_DESCRIPTION", "first"),
    }
    out = sub.groupby("variant_id", as_index=False).agg(**{k: v for k, v in gcols.items()})
    out["label"] = "somatic"
    out["source"] = (
        f"cosmic_mutantcensus_v104_mane_nc_haem_min{args.min_haem}"
        f"_ge{args.min_haem_confirmed}conf"
    )
    out["mane_transcript"] = out["gene"].map(lambda g: gene_ivs[g]["transcript_id"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"haem_min{args.min_haem}_ge{args.min_haem_confirmed}conf"
    path = out_dir / f"arch_mane_noncoding_somatic_cosmic_mutantcensus_{tag}.csv"
    out.to_csv(path, index=False)

    # overlap with current NCV confirmed set
    ncv = pd.read_csv(out_dir / "arch_mane_noncoding_somatic_cosmic_ncv_haem_min1.csv")
    ncv = ncv.loc[ncv["n_haem_confirmed"].astype(float) >= 1]
    ncv_ids = set(ncv["variant_id"].astype(str))
    mc_ids = set(out["variant_id"].astype(str))
    summary = {
        "n_mutantcensus_mane_nc": int(len(out)),
        "n_ncv_confirmed": int(len(ncv_ids)),
        "overlap": int(len(mc_ids & ncv_ids)),
        "mutantcensus_only": int(len(mc_ids - ncv_ids)),
        "ncv_only": int(len(ncv_ids - mc_ids)),
        "union": int(len(mc_ids | ncv_ids)),
        "by_gene": out["gene"].value_counts().sort_index().to_dict(),
        "by_region": out["mane_region"].value_counts().to_dict(),
        "out": str(path),
    }
    (out_dir / f"cosmic_mutantcensus_{tag}_vs_ncv_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
