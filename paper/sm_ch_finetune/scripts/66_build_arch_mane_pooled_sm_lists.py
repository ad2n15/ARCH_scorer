#!/usr/bin/env python3
"""Build ARCH MANE noncoding germline/somatic lists (pooled SM definition).

Germline:
  previous gnomAD filters (PASS, SNV, 0<AF<0.01, DP>=10, median AB>0.05)
  AND ab_n_carriers_hist >= 5
  AND not in somatic pool

Somatic (union):
  1) gnomAD AS_pab SM (AS_pab_max<0.05 & 0.05<ab_median<=0.30)
  2) paper MW SM with AB n>2 (FDR q<0.05 & Hedges g>=0.8 vs rs2276599)
  3) all COSMIC NCV SNVs in ARCH MANE noncoding windows (any tissue)

Overlap germline ∩ somatic dropped from BOTH.

Also writes pilot parquet + avinput for API / ANNOVAR.

Example:
  cd paper/sm_ch_finetune
  python3 scripts/66_build_arch_mane_pooled_sm_lists.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent

HAEM = "haematopoietic_and_lymphoid_tissue"
ARCH21 = [
    "ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1", "IDH2",
    "JAK2", "KIT", "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1", "SETBP1", "SF3B1",
    "SRSF2", "TET2", "TP53",
]


def _chrom_str(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s[3:] if s.lower().startswith("chr") else s


def _load_mod61():
    import importlib.util

    path = PROJECT / "scripts/61_build_arch_mane_noncoding_lists.py"
    spec = importlib.util.spec_from_file_location("arch61", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cosmic_all_mane(intervals: dict, ncv: Path, chunksize: int) -> pd.DataFrame:
    by_chrom: dict[str, list[tuple[int, int, str, dict]]] = defaultdict(list)
    for g, iv in intervals.items():
        lo, hi = iv["window"]
        by_chrom[str(iv["chrom"])].append((lo, hi, g, iv))

    allele: dict[str, dict] = {}
    usecols = {
        "COSMIC_PHENOTYPE_ID",
        "GENOMIC_MUTATION_ID",
        "CHROMOSOME",
        "GENOME_START",
        "GENOMIC_WT_ALLELE",
        "GENOMIC_MUT_ALLELE",
        "MUTATION_SOMATIC_STATUS",
        "GENE_SYMBOL",
    }
    n_raw = 0
    print(f"Streaming COSMIC NCV for all-MANE alleles: {ncv}", flush=True)
    for i, chunk in enumerate(
        pd.read_csv(
            ncv,
            sep="\t",
            usecols=lambda c: c in usecols,
            chunksize=chunksize,
            dtype=str,
            low_memory=True,
        ),
        1,
    ):
        n_raw += len(chunk)
        chunk["chrom"] = chunk["CHROMOSOME"].map(_chrom_str)
        chunk = chunk.loc[chunk["chrom"].isin(by_chrom)]
        if chunk.empty:
            if i % 20 == 0:
                print(f"  scanned {n_raw:,} alleles_so_far={len(allele)}", flush=True)
            continue
        chunk = chunk.copy()
        chunk["pos"] = pd.to_numeric(chunk["GENOME_START"], errors="coerce")
        chunk = chunk.loc[chunk["pos"].notna()]
        chunk["pos"] = chunk["pos"].astype(int)
        chunk["ref"] = chunk["GENOMIC_WT_ALLELE"].astype(str).str.upper()
        chunk["alt"] = chunk["GENOMIC_MUT_ALLELE"].astype(str).str.upper()
        chunk = chunk.loc[
            (chunk["ref"].str.len() == 1)
            & (chunk["alt"].str.len() == 1)
            & chunk["ref"].isin(list("ACGT"))
            & chunk["alt"].isin(list("ACGT"))
        ]
        if chunk.empty:
            continue
        for r in chunk.itertuples(index=False):
            chrom, pos = r.chrom, int(r.pos)
            hit_g = None
            hit_iv = None
            for lo, hi, g, iv in by_chrom[chrom]:
                if lo <= pos <= hi:
                    hit_g, hit_iv = g, iv
                    break
            if hit_g is None:
                continue
            # refine to MANE noncoding interval class
            region = None
            # lazy import classify via intervals already loaded with mod outside
            vid = f"{chrom}-{pos}-{r.ref}-{r.alt}"
            if vid not in allele:
                allele[vid] = {
                    "variant_id": vid,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": r.ref,
                    "alt": r.alt,
                    "gene": hit_g,
                    "mane_transcript": hit_iv["transcript_id"],
                    "n_cosmic_rows": 0,
                    "GENOMIC_MUTATION_ID": getattr(r, "GENOMIC_MUTATION_ID", None),
                }
            allele[vid]["n_cosmic_rows"] += 1
        if i % 20 == 0:
            print(f"  scanned {n_raw:,} alleles_so_far={len(allele)}", flush=True)

    print(f"COSMIC all-MANE unique SNV alleles: {len(allele)}", flush=True)
    return pd.DataFrame(list(allele.values()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--candidates",
        default=str(
            PROJECT
            / "data/arch_mane_noncoding_gnomad_ab/mane_noncoding_gnomad_all_candidates_with_metrics.csv"
        ),
    )
    p.add_argument(
        "--aspab-sm",
        default=str(
            PROJECT
            / "data/arch_mane_noncoding_gnomad_ab/mane_noncoding_gnomad_somatic_ab_skew_sm.csv"
        ),
    )
    p.add_argument(
        "--paper-mw-sm",
        default=str(
            PROJECT
            / "data/arch_mane_noncoding_gnomad_ab_minABn2/mane_noncoding_gnomad_somatic_paper_mw_sm.csv"
        ),
    )
    p.add_argument("--gtf", default=str(PROJECT / "data/reference/gencode.v46.annotation.gtf"))
    p.add_argument(
        "--intervals",
        default=str(PROJECT / "data/arch_mane_noncoding/arch_mane_noncoding_intervals.json"),
    )
    p.add_argument("--cosmic-ncv", default=str(ROOT / "Cosmic_NonCodingVariants_v104_GRCh38.tsv"))
    p.add_argument("--min-ab-n-germline", type=int, default=5)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument(
        "--out-dir",
        default=str(PROJECT / "data/arch_mane_noncoding_pooled_sm"),
    )
    p.add_argument(
        "--pilot-dir",
        default=str(PROJECT / "data/pilot_arch_mane_noncoding_pooled_sm"),
    )
    p.add_argument("--skip-cosmic", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir = Path(args.pilot_dir)
    pilot_dir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(args.candidates)
    cand["variant_id"] = cand["variant_id"].astype(str)
    aspab = pd.read_csv(args.aspab_sm)
    aspab["variant_id"] = aspab["variant_id"].astype(str)
    paper = pd.read_csv(args.paper_mw_sm)
    paper["variant_id"] = paper["variant_id"].astype(str)

    aspab_ids = set(aspab["variant_id"])
    paper_ids = set(paper["variant_id"])

    # COSMIC all in MANE noncoding (need full intervals for classify_pos)
    mod = _load_mod61()
    gtf = Path(args.gtf)
    mane: dict[str, str] = {}
    want = set(ARCH21)
    with open(gtf) as f:
        for line in f:
            if line.startswith("#") or "\ttranscript\t" not in line:
                continue
            if 'tag "MANE_Select"' not in line:
                continue
            mg = re.search(r'gene_name "([^"]+)"', line)
            mt = re.search(r'transcript_id "([^"]+)"', line)
            if mg and mt and mg.group(1) in want and mg.group(1) not in mane:
                mane[mg.group(1)] = mt.group(1)
    missing = [g for g in ARCH21 if g not in mane]
    if missing:
        raise SystemExit(f"MANE missing: {missing}")
    print("Loading full MANE noncoding intervals from GTF …", flush=True)
    intervals = {g: mod.load_mane_noncoding_intervals(gtf, mane[g]) for g in ARCH21}

    # attach mane_region for cosmic via classify
    cosmic_path = out_dir / "somatic_cosmic_ncv_all_mane.csv"
    if args.skip_cosmic and cosmic_path.exists():
        cosmic = pd.read_csv(cosmic_path)
    else:
        cosmic = _cosmic_all_mane(intervals, Path(args.cosmic_ncv), args.chunksize)
        # classify region + drop CDS hits if any
        rows = []
        for r in cosmic.itertuples(index=False):
            iv = intervals.get(r.gene)
            if iv is None:
                continue
            region = mod.classify_pos(int(r.pos), iv)
            if region is None:
                continue
            d = r._asdict() if hasattr(r, "_asdict") else dict(zip(cosmic.columns, r))
            d["mane_region"] = region
            rows.append(d)
        cosmic = pd.DataFrame(rows)
        cosmic.to_csv(cosmic_path, index=False)
    cosmic["variant_id"] = cosmic["variant_id"].astype(str)
    cosmic_ids = set(cosmic["variant_id"])

    somatic_ids = aspab_ids | paper_ids | cosmic_ids

    # germline: previous filters already in candidates; ab_n >= 5; not somatic
    abn = pd.to_numeric(cand["ab_n_carriers_hist"], errors="coerce")
    germ_mask = (abn >= int(args.min_ab_n_germline)) & (~cand["variant_id"].isin(somatic_ids))
    germ = cand.loc[germ_mask].copy()
    germ["label"] = "germline"
    germ["source"] = "gnomad_v4_genomes"
    germ["sm_source"] = ""

    # build somatic rows with source tags
    meta = cand.set_index("variant_id", drop=False)
    som_rows = []
    for vid in sorted(somatic_ids):
        tags = []
        if vid in aspab_ids:
            tags.append("gnomad_aspab")
        if vid in paper_ids:
            tags.append("paper_mw_abn_gt2")
        if vid in cosmic_ids:
            tags.append("cosmic_ncv_all")
        src = "+".join(tags)
        if vid in meta.index:
            r = meta.loc[vid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            row = {
                "variant_id": vid,
                "chrom": r["chrom"],
                "pos": int(r["pos"]),
                "ref": r["ref"],
                "alt": r["alt"],
                "gene": r["gene"],
                "mane_region": r.get("mane_region"),
                "mane_transcript": r.get("mane_transcript"),
                "af": r.get("af"),
                "ab_median": r.get("ab_median"),
                "ab_n_carriers_hist": r.get("ab_n_carriers_hist"),
                "AS_pab_max": r.get("AS_pab_max"),
                "label": "somatic",
                "source": src,
                "sm_source": src,
            }
        elif vid in cosmic_ids:
            cr = cosmic.loc[cosmic["variant_id"] == vid].iloc[0]
            row = {
                "variant_id": vid,
                "chrom": cr["chrom"],
                "pos": int(cr["pos"]),
                "ref": cr["ref"],
                "alt": cr["alt"],
                "gene": cr["gene"],
                "mane_region": cr.get("mane_region"),
                "mane_transcript": cr.get("mane_transcript"),
                "af": float("nan"),
                "ab_median": float("nan"),
                "ab_n_carriers_hist": float("nan"),
                "AS_pab_max": float("nan"),
                "label": "somatic",
                "source": src,
                "sm_source": src,
                "n_cosmic_rows": cr.get("n_cosmic_rows"),
                "GENOMIC_MUTATION_ID": cr.get("GENOMIC_MUTATION_ID"),
            }
        else:
            # paper/aspab only should be in meta; skip orphans
            continue
        som_rows.append(row)
    som = pd.DataFrame(som_rows)

    # drop residual overlaps (safety)
    both = set(germ["variant_id"]) & set(som["variant_id"])
    germ = germ.loc[~germ["variant_id"].isin(both)].copy()
    som = som.loc[~som["variant_id"].isin(both)].copy()

    keep_cols = [
        "variant_id", "chrom", "pos", "ref", "alt", "gene", "label", "source",
        "sm_source", "mane_region", "mane_transcript", "af", "ab_median",
        "ab_n_carriers_hist", "AS_pab_max", "rsid", "dp_as_vardp",
        "n_cosmic_rows", "GENOMIC_MUTATION_ID",
    ]
    for df in (germ, som):
        for c in keep_cols:
            if c not in df.columns:
                df[c] = pd.NA

    germ_out = germ[keep_cols]
    som_out = som[keep_cols]
    combined = pd.concat([germ_out, som_out], ignore_index=True, sort=False)

    germ_path = out_dir / "arch_mane_noncoding_germline_abn_ge5.csv"
    som_path = out_dir / "arch_mane_noncoding_somatic_pooled.csv"
    all_path = out_dir / "arch_mane_noncoding_germline_plus_somatic_pooled.csv"
    av_path = out_dir / "arch_mane_noncoding_pooled.avinput"
    summary_path = out_dir / "build_summary.json"

    germ_out.to_csv(germ_path, index=False)
    som_out.to_csv(som_path, index=False)
    combined.to_csv(all_path, index=False)

    # avinput: chrom pos pos ref alt (no chr)
    with open(av_path, "w") as f:
        for r in combined.itertuples(index=False):
            chrom = _chrom_str(r.chrom)
            f.write(f"{chrom}\t{int(r.pos)}\t{int(r.pos)}\t{r.ref}\t{r.alt}\n")

    # pilot parquet for AlphaGenome API
    pilot = combined.copy()
    pilot["chrom"] = pilot["chrom"].astype(str).map(
        lambda c: c if str(c).startswith("chr") else f"chr{c}"
    )
    pilot["label_str"] = pilot["label"].astype(str)
    pilot["split"] = "test"
    pilot_cols = [
        "chrom", "pos", "ref", "alt", "variant_id", "gene", "label", "label_str",
        "split", "mane_region", "source",
    ]
    pilot[pilot_cols].to_parquet(pilot_dir / "test.parquet", index=False)

    # source breakdown
    src_counts = som_out["source"].value_counts().to_dict()
    summary = {
        "germline_rule": (
            f"PASS/SNV/0<AF<0.01/DP>=10/median_AB>0.05 AND ab_n>={args.min_ab_n_germline} "
            "AND not in somatic pool"
        ),
        "somatic_rule": (
            "union of gnomAD AS_pab SM + paper MW (AB n>2, q<0.05, g>=0.8) "
            "+ all COSMIC NCV in ARCH MANE noncoding windows"
        ),
        "n_germline": int(len(germ_out)),
        "n_somatic": int(len(som_out)),
        "n_combined": int(len(combined)),
        "n_overlap_dropped": int(len(both)),
        "somatic_input_counts": {
            "aspab": len(aspab_ids),
            "paper_mw_abn_gt2": len(paper_ids),
            "cosmic_all_mane": len(cosmic_ids),
            "union_before_overlap_drop": len(somatic_ids),
        },
        "somatic_source_tag_counts": src_counts,
        "germline_by_gene": germ_out.groupby("gene").size().to_dict(),
        "somatic_by_gene": som_out.groupby("gene").size().to_dict(),
        "outputs": {
            "germline": str(germ_path),
            "somatic": str(som_path),
            "combined": str(all_path),
            "avinput": str(av_path),
            "pilot_parquet": str(pilot_dir / "test.parquet"),
            "cosmic_all": str(cosmic_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "n_germline", "n_somatic", "n_combined", "n_overlap_dropped", "somatic_input_counts"
    )}, indent=2), flush=True)
    print(f"Wrote {all_path}", flush=True)
    print(f"Wrote {pilot_dir / 'test.parquet'}", flush=True)
    print(f"Wrote {av_path}", flush=True)


if __name__ == "__main__":
    main()
