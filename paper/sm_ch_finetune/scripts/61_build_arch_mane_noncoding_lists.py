#!/usr/bin/env python3
"""Build MANE noncoding germline/somatic lists for 21 ARCH genes.

Germline (gnomAD v4.1 genomes):
  - MANE Select noncoding: intron + 5'/3' UTR + 200 bp core promoter
  - PASS SNVs, AF < 0.01 (and AF > 0), exclude putative low AB (median AB <= 0.30)

Somatic (COSMIC NonCodingVariants v104 GRCh38), default --count-scope haem:
  - same MANE noncoding intervals
  - haematopoietic_and_lymphoid_tissue
  - GENOMIC_MUTATION_ID seen >= --min-count times in haemato (any somatic status)
  - AND at least --min-confirmed of those haemato rows are Confirmed somatic
    (Option A default: min-count=3, min-confirmed=0 — no Confirmed required;
     germline∩somatic alleles are dropped using the gnomAD germline set)

Overlap alleles (germline ∩ somatic) are dropped from BOTH sides.

Example (host; needs tabix from samtools — not available inside default apptainer bind):

  module load samtools
  cd paper
  python3 sm_ch_finetune/scripts/61_build_arch_mane_noncoding_lists.py \\
    --count-scope haem --min-count 3 --min-confirmed 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

HAEM_SITE = "haematopoietic_and_lymphoid_tissue"
CONFIRMED = "Confirmed somatic variant"

# Publication GVS 21-gene ARCH set (excludes BRCC3 / U2AF1 from CH_genes_list.txt)
ARCH21 = [
    "ASXL1",
    "ASXL2",
    "CBL",
    "DNMT3A",
    "ETNK1",
    "GNAS",
    "GNB1",
    "IDH1",
    "IDH2",
    "JAK2",
    "KIT",
    "KRAS",
    "MPL",
    "NRAS",
    "PPM1D",
    "RUNX1",
    "SETBP1",
    "SF3B1",
    "SRSF2",
    "TET2",
    "TP53",
]


def _info_map(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in info.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        else:
            out[part] = "1"
    return out


def _ab_median(freq: str | None) -> float | None:
    if not freq or freq in (".", "NA"):
        return None
    try:
        counts = [int(x) for x in freq.split("|")]
    except ValueError:
        return None
    if len(counts) != 20 or sum(counts) == 0:
        return None
    total = sum(counts)
    cum = 0
    for i, c in enumerate(counts):
        cum += c
        if cum >= total / 2:
            return (i + 0.5) / 20.0
    return None


def _chrom_str(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.lower().startswith("chr"):
        return s[3:]
    return s


def load_mane_select_map(gtf: Path, genes: list[str]) -> dict[str, str]:
    """gene -> MANE Select transcript_id (with version)."""
    want = set(genes)
    found: dict[str, str] = {}
    with open(gtf) as f:
        for line in f:
            if line.startswith("#") or "\ttranscript\t" not in line:
                continue
            if 'tag "MANE_Select"' not in line:
                continue
            m_g = re.search(r'gene_name "([^"]+)"', line)
            m_t = re.search(r'transcript_id "([^"]+)"', line)
            if not m_g or not m_t:
                continue
            g = m_g.group(1)
            if g in want and g not in found:
                found[g] = m_t.group(1)
    missing = [g for g in genes if g not in found]
    if missing:
        raise SystemExit(f"MANE Select missing for: {missing}")
    return found


def load_mane_noncoding_intervals(gtf: Path, transcript_id: str) -> dict:
    exons: list[tuple[int, int]] = []
    cds: list[tuple[int, int]] = []
    tx: tuple[int, int, str] | None = None
    chrom = None
    with open(gtf) as f:
        for line in f:
            if transcript_id not in line or line.startswith("#"):
                continue
            p = line.rstrip().split("\t")
            if len(p) < 9:
                continue
            feat, start, end, strand = p[2], int(p[3]), int(p[4]), p[6]
            chrom = _chrom_str(p[0])
            if feat == "transcript":
                tx = (start, end, strand)
            elif feat == "exon":
                exons.append((start, end))
            elif feat == "CDS":
                cds.append((start, end))
    if tx is None or not exons or chrom is None:
        raise SystemExit(f"MANE transcript {transcript_id} not found in {gtf}")

    exons_s = sorted(exons)
    cds_s = sorted(cds)
    introns: list[tuple[int, int]] = []
    for (a, b), (c, _d) in zip(exons_s, exons_s[1:]):
        if c > b + 1:
            introns.append((b + 1, c - 1))

    utr_ivs: list[tuple[int, int]] = []
    for es, ee in exons_s:
        cur = es
        overlapping = [c for c in cds_s if not (c[1] < es or c[0] > ee)]
        if not overlapping:
            utr_ivs.append((es, ee))
            continue
        for cs, ce in overlapping:
            if cur < cs:
                utr_ivs.append((cur, cs - 1))
            cur = max(cur, ce + 1)
        if cur <= ee:
            utr_ivs.append((cur, ee))

    tss = tx[1] if tx[2] == "-" else tx[0]
    if tx[2] == "-":
        promoter = (tss + 1, tss + 200)
    else:
        promoter = (tss - 200, tss - 1)

    return {
        "transcript_id": transcript_id,
        "chrom": chrom,
        "tx_start": tx[0],
        "tx_end": tx[1],
        "strand": tx[2],
        "tss": tss,
        "promoter": promoter,
        "introns": introns,
        "utr": utr_ivs,
        "cds": cds_s,
        "exons": exons_s,
        "window": (
            min(promoter[0], tx[0]) - 5,
            max(promoter[1], tx[1]) + 5,
        ),
    }


def classify_pos(pos: int, iv: dict) -> str | None:
    ps, pe = iv["promoter"]
    if ps <= pos <= pe:
        return "core_promoter_200"
    if iv["cds"]:
        cds_min = min(c[0] for c in iv["cds"])
        cds_max = max(c[1] for c in iv["cds"])
    else:
        cds_min = cds_max = None
    for s, e in iv["utr"]:
        if s <= pos <= e:
            if cds_min is None:
                return "UTR"
            if iv["strand"] == "-":
                return "UTR5" if pos > cds_max else "UTR3"
            return "UTR5" if pos < cds_min else "UTR3"
    for s, e in iv["introns"]:
        if s <= pos <= e:
            return "intron"
    return None


def _parse_vcf_record(
    chrom: str,
    pos: int,
    rsid: str,
    ref: str,
    alt: str,
    filt: str,
    info: str,
    *,
    gene: str,
    iv: dict,
    max_af: float,
) -> list[dict]:
    if filt != "PASS":
        return []
    im0 = _info_map(info)
    alts = alt.split(",")
    afs = im0.get("AF", "").split(",") if im0.get("AF") else []
    rows = []
    for i, a in enumerate(alts):
        ref_u, a_u = ref.upper(), a.upper()
        if len(ref_u) != 1 or len(a_u) != 1:
            continue
        if ref_u not in "ACGT" or a_u not in "ACGT":
            continue
        region = classify_pos(pos, iv)
        if region is None:
            continue
        try:
            af = float(afs[i]) if i < len(afs) and afs[i] not in ("", ".") else float("nan")
        except ValueError:
            af = float("nan")
        if not (pd.isna(af) or af < max_af):
            continue
        if not (pd.isna(af) or af > 0):
            continue
        ab = _ab_median(im0.get("ab_hist_alt_bin_freq"))
        if ab is not None and ab <= 0.30:
            continue
        chrom_n = _chrom_str(chrom)
        rows.append(
            {
                "variant_id": f"{chrom_n}-{pos}-{ref_u}-{a_u}",
                "chrom": chrom_n,
                "pos": pos,
                "ref": ref_u,
                "alt": a_u,
                "gene": gene,
                "label": "germline",
                "source": "gnomad_v4_genomes",
                "mane_region": region,
                "mane_transcript": iv["transcript_id"],
                "af": af,
                "rsid": None if rsid == "." else rsid,
                "ab_median_approx": ab,
                "cadd_phred": float(im0["cadd_phred"])
                if im0.get("cadd_phred") not in (None, "", ".")
                else float("nan"),
            }
        )
    return rows


def _find_tabix() -> str | None:
    import os
    import shutil

    for cand in (
        os.environ.get("TABIX_BIN"),
        shutil.which("tabix"),
        __import__("os").environ.get("TABIX", "tabix"),
    ):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def parse_gnomad_gene(
    vcf_path: Path,
    gene: str,
    iv: dict,
    *,
    max_af: float,
) -> pd.DataFrame:
    """Read gnomAD WGS variants in the gene MANE noncoding window.

    Prefers ``tabix`` (module load samtools). Falls back to gzip stream for .bgz.
    """
    import gzip
    import subprocess

    rows: list[dict] = []
    lo, hi = iv["window"]
    chrom = iv["chrom"]
    vcf_s = str(vcf_path)

    def _consume_vcf_lines(lines) -> None:
        for line in lines:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            c, pos_s, rsid, ref, alt, _q, filt, info = cols[:8]
            rows.extend(
                _parse_vcf_record(
                    c,
                    int(pos_s),
                    rsid,
                    ref,
                    alt,
                    filt,
                    info,
                    gene=gene,
                    iv=iv,
                    max_af=max_af,
                )
            )

    tabix = _find_tabix()
    tbi = Path(vcf_s + ".tbi")
    if tabix and tbi.exists() and vcf_s.endswith((".vcf.bgz", ".vcf.gz")):
        # Contig naming: gnomAD v4 genomes use chrN
        last_err = None
        for contig in (f"chr{chrom}", chrom):
            region = f"{contig}:{lo}-{hi}"
            proc = subprocess.run(
                [tabix, vcf_s, region],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                _consume_vcf_lines(proc.stdout.splitlines())
                break
            last_err = proc.stderr.strip() or f"tabix exit {proc.returncode}"
        else:
            raise SystemExit(
                f"tabix failed for {gene} {vcf_s} window chr{chrom}:{lo}-{hi}: {last_err}\n"
                "Run: module load samtools"
            )
        return pd.DataFrame(rows).drop_duplicates("variant_id") if rows else pd.DataFrame()

    if tabix is None and vcf_s.endswith((".vcf.bgz", ".vcf.gz")) and tbi.exists():
        raise SystemExit(
            "Indexed bgzip VCF found but tabix is not on PATH.\n"
            "Run:  module load samtools\n"
            "Then re-run this script on the host (or bind samtools into apptainer)."
        )

    # sequential fallback (plain VCF or bgzip without index)
    if vcf_s.endswith((".gz", ".bgz")):
        opener = gzip.open
    else:
        opener = open
    with opener(vcf_path, "rt") as f:  # type: ignore[arg-type]
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            c, pos_s, rsid, ref, alt, _q, filt, info = cols[:8]
            if _chrom_str(c) != chrom:
                continue
            pos = int(pos_s)
            if pos < lo or pos > hi:
                continue
            rows.extend(
                _parse_vcf_record(
                    c,
                    pos,
                    rsid,
                    ref,
                    alt,
                    filt,
                    info,
                    gene=gene,
                    iv=iv,
                    max_af=max_af,
                )
            )
    return pd.DataFrame(rows).drop_duplicates("variant_id") if rows else pd.DataFrame()


def parse_cosmic_ncv_arch(
    ncv_path: Path,
    clf_path: Path,
    gene_ivs: dict[str, dict],
    *,
    min_count: int,
    min_confirmed: int,
    count_scope: str,
    require_haem: bool,
    snv_only: bool = True,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Filter COSMIC NCV to MANE noncoding with haem recurrence + ≥1 confirmed.

    Default (count_scope=haem):
      n_haem >= min_count (haemato rows, any status)
      n_haem_confirmed >= min_confirmed (Confirmed somatic among those haem rows)

    Streams the large TSV in chunks.
    """
    print(f"Streaming COSMIC NCV: {ncv_path} (chunksize={chunksize:,})")
    clf = pd.read_csv(clf_path, sep="\t", low_memory=False)
    haem_pheno = set(
        clf.loc[clf["PRIMARY_SITE"].astype(str) == HAEM_SITE, "COSMIC_PHENOTYPE_ID"].astype(str)
    )
    print(f"  haem phenotype IDs: {len(haem_pheno):,}")
    print(
        f"  rule: n_haem>={min_count} AND n_haem_confirmed>={min_confirmed} "
        f"(count_scope={count_scope})"
    )

    by_chrom: dict[str, list[tuple[int, int, str, dict]]] = {}
    for gene, iv in gene_ivs.items():
        by_chrom.setdefault(iv["chrom"], []).append((iv["window"][0], iv["window"][1], gene, iv))

    usecols = [
        "GENE_SYMBOL",
        "MUTATION_NC_ID",
        "COSMIC_PHENOTYPE_ID",
        "GENOMIC_MUTATION_ID",
        "CHROMOSOME",
        "GENOME_START",
        "GENOMIC_WT_ALLELE",
        "GENOMIC_MUT_ALLELE",
        "HGVSG",
        "MUTATION_SOMATIC_STATUS",
    ]
    kept_parts: list[pd.DataFrame] = []
    n_raw = 0
    n_win = 0
    for i, chunk in enumerate(
        pd.read_csv(
            ncv_path,
            sep="\t",
            usecols=lambda c: c in set(usecols),
            chunksize=chunksize,
            low_memory=True,
            dtype=str,
        ),
        1,
    ):
        n_raw += len(chunk)
        chunk["chrom"] = chunk["CHROMOSOME"].map(_chrom_str)
        chunk = chunk.loc[chunk["chrom"].isin(by_chrom)].copy()
        if chunk.empty:
            if i % 10 == 0:
                print(f"  … chunk {i}: scanned {n_raw:,} rows, kept window {n_win:,}")
            continue
        chunk["pos"] = pd.to_numeric(chunk["GENOME_START"], errors="coerce")
        chunk = chunk.loc[chunk["pos"].notna()].copy()
        chunk["pos"] = chunk["pos"].astype(int)
        chunk["ref"] = chunk["GENOMIC_WT_ALLELE"].where(
            chunk["GENOMIC_WT_ALLELE"].notna() & (chunk["GENOMIC_WT_ALLELE"] != "nan"), None
        )
        chunk["alt"] = chunk["GENOMIC_MUT_ALLELE"].where(
            chunk["GENOMIC_MUT_ALLELE"].notna() & (chunk["GENOMIC_MUT_ALLELE"] != "nan"), None
        )
        chunk = chunk.loc[chunk["ref"].notna() & chunk["alt"].notna()].copy()
        chunk["ref"] = chunk["ref"].astype(str).str.upper()
        chunk["alt"] = chunk["alt"].astype(str).str.upper()
        if snv_only:
            chunk = chunk.loc[
                (chunk["ref"].str.len() == 1)
                & (chunk["alt"].str.len() == 1)
                & chunk["ref"].isin(list("ACGT"))
                & chunk["alt"].isin(list("ACGT"))
            ].copy()
        if chunk.empty:
            continue

        gene_col = []
        region_col = []
        keep_idx = []
        for idx, chrom, pos in zip(chunk.index, chunk["chrom"], chunk["pos"]):
            hit_gene = None
            hit_reg = None
            for lo, hi, gene, iv in by_chrom.get(chrom, []):
                if lo <= pos <= hi:
                    reg = classify_pos(int(pos), iv)
                    if reg is not None:
                        hit_gene, hit_reg = gene, reg
                        break
            if hit_gene is not None:
                keep_idx.append(idx)
                gene_col.append(hit_gene)
                region_col.append(hit_reg)
        if not keep_idx:
            if i % 10 == 0:
                print(f"  … chunk {i}: scanned {n_raw:,} rows, kept window {n_win:,}")
            continue
        sub = chunk.loc[keep_idx].copy()
        sub["gene"] = gene_col
        sub["mane_region"] = region_col
        sub["_confirmed"] = sub["MUTATION_SOMATIC_STATUS"].astype(str) == CONFIRMED
        sub["_haem"] = sub["COSMIC_PHENOTYPE_ID"].astype(str).isin(haem_pheno)

        if count_scope == "haem":
            # keep all haemato rows in window (confirmed status tracked separately)
            sub = sub.loc[sub["_haem"]].copy()
        elif count_scope == "all":
            # keep all window rows; haem/confirmed flags used later
            pass
        else:
            raise SystemExit("--count-scope must be haem or all")

        n_win += len(sub)
        if len(sub):
            kept_parts.append(sub)
        if i % 10 == 0:
            print(f"  … chunk {i}: scanned {n_raw:,} rows, kept-in-scope {n_win:,}")

    print(f"  scanned rows: {n_raw:,}; rows kept in MANE window + scope: {n_win:,}")
    if not kept_parts:
        return pd.DataFrame()

    win = pd.concat(kept_parts, ignore_index=True)
    print(
        f"  haem rows: {int(win['_haem'].sum()):,}; "
        f"haem+confirmed: {int((win['_haem'] & win['_confirmed']).sum()):,}"
    )

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
        (stats["n_haem"] >= min_count) & (stats["n_haem_confirmed"] >= min_confirmed)
    ]
    print(
        f"  unique GENOMIC_MUTATION_ID with n_haem>={min_count} "
        f"AND n_haem_confirmed>={min_confirmed}: {len(keep_ids)}"
    )

    sub = haem.loc[haem["GENOMIC_MUTATION_ID"].isin(keep_ids)].copy()
    if require_haem is False and count_scope == "all":
        # rare path: expand to all-site rows for kept IDs
        sub = win.loc[win["GENOMIC_MUTATION_ID"].isin(keep_ids)].copy()
    print(f"  rows for retained alleles: {len(sub):,}")

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
    }
    for c in ("MUTATION_NC_ID", "HGVSG", "GENE_SYMBOL"):
        if c in sub.columns:
            gcols[c] = (c, "first")
    agg = sub.groupby("variant_id", as_index=False).agg(**gcols)
    agg["mane_transcript"] = agg["gene"].map(
        lambda g: gene_ivs[g]["transcript_id"] if g in gene_ivs else None
    )
    agg["label"] = "somatic"
    agg["source"] = (
        f"cosmic_ncv_v104_haem_min{min_count}_ge{min_confirmed}conf"
    )
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtf", default=str(PROJECT / "data/reference/gencode.v46.annotation.gtf"))
    p.add_argument(
        "--gnomad-dir",
        default=str(ROOT),
        help="Directory with gnomad.genomes.v4.1.1.sites.chrN.vcf.bgz",
    )
    p.add_argument(
        "--gnomad-pattern",
        default="gnomad.genomes.v4.1.1.sites.chr{chrom}.vcf.bgz",
    )
    p.add_argument(
        "--cosmic-ncv",
        default=str(ROOT / "Cosmic_NonCodingVariants_v104_GRCh38.tsv"),
    )
    p.add_argument(
        "--classification",
        default=str(ROOT / "Cosmic_Classification_v104_GRCh38.tsv"),
    )
    p.add_argument("--max-af", type=float, default=0.01)
    p.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum haemato observations per GENOMIC_MUTATION_ID (any status)",
    )
    p.add_argument(
        "--min-confirmed",
        type=int,
        default=0,
        help="Minimum Confirmed somatic haemato observations (0 = Option A, no Confirmed filter)",
    )
    p.add_argument(
        "--count-scope",
        choices=["haem", "all"],
        default="haem",
        help="haem: count haemato rows (default). all: still applies haem n + confirmed rule.",
    )
    p.add_argument(
        "--no-require-haem",
        action="store_true",
        help="Legacy flag; haem recurrence rule still uses haemato counts.",
    )
    p.add_argument(
        "--genes",
        default=",".join(ARCH21),
        help="Comma-separated gene symbols (default: publication 21 ARCH genes)",
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT / "data/arch_mane_noncoding"),
    )
    p.add_argument("--genes-only", default="", help="Optional subset, e.g. TP53,DNMT3A")
    args = p.parse_args()

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    if args.genes_only:
        want = {g.strip() for g in args.genes_only.split(",") if g.strip()}
        genes = [g for g in genes if g in want]
    require_haem = not args.no_require_haem

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mane_map = load_mane_select_map(Path(args.gtf), genes)
    gene_ivs: dict[str, dict] = {}
    print("MANE noncoding intervals:")
    for g in genes:
        iv = load_mane_noncoding_intervals(Path(args.gtf), mane_map[g])
        gene_ivs[g] = iv
        print(
            f"  {g:8s} {iv['transcript_id']:22s} chr{iv['chrom']}:{iv['window'][0]}-{iv['window'][1]} "
            f"introns={len(iv['introns'])} utr={len(iv['utr'])}"
        )

    # --- germline ---
    germ_parts = []
    gnomad_dir = Path(args.gnomad_dir)
    for g in genes:
        chrom = gene_ivs[g]["chrom"]
        vcf = gnomad_dir / args.gnomad_pattern.format(chrom=chrom)
        if not vcf.exists():
            raise SystemExit(f"Missing gnomAD WGS VCF for {g}: {vcf}")
        print(f"\n[{g}] reading {vcf.name} …")
        part = parse_gnomad_gene(vcf, g, gene_ivs[g], max_af=args.max_af)
        print(f"  germline candidates: {len(part)}")
        if len(part):
            print(" ", part["mane_region"].value_counts().to_dict())
            germ_parts.append(part)
    germ = (
        pd.concat(germ_parts, ignore_index=True).drop_duplicates("variant_id")
        if germ_parts
        else pd.DataFrame()
    )
    print(f"\nTOTAL germline before overlap drop: {len(germ)}")

    # --- somatic ---
    som = parse_cosmic_ncv_arch(
        Path(args.cosmic_ncv),
        Path(args.classification),
        gene_ivs,
        min_count=args.min_count,
        min_confirmed=args.min_confirmed,
        count_scope=args.count_scope,
        require_haem=require_haem,
    )
    print(f"TOTAL somatic unique alleles: {len(som)}")
    if len(som):
        print(som.groupby("gene").size().sort_values(ascending=False).to_string())

    both = set(germ["variant_id"]) & set(som["variant_id"]) if len(germ) and len(som) else set()
    germ2 = germ.loc[~germ["variant_id"].isin(both)].copy() if len(germ) else germ
    som2 = som.loc[~som["variant_id"].isin(both)].copy() if len(som) else som
    print(f"Dropped germline∩somatic overlap alleles: {len(both)}")

    tag = f"haem_min{args.min_count}_ge{args.min_confirmed}conf"
    germ_path = out_dir / "arch_mane_noncoding_germline_gnomad_wgs_af001.csv"
    som_path = out_dir / f"arch_mane_noncoding_somatic_cosmic_ncv_{tag}.csv"
    all_path = out_dir / f"arch_mane_noncoding_germline_plus_somatic_{tag}.csv"
    meta_path = out_dir / "arch_mane_noncoding_build_summary.json"
    iv_path = out_dir / "arch_mane_noncoding_intervals.json"

    germ2.to_csv(germ_path, index=False)
    som2.to_csv(som_path, index=False)
    combined = pd.concat([germ2, som2], ignore_index=True, sort=False)
    combined.to_csv(all_path, index=False)

    intervals_out = {
        g: {
            "transcript_id": iv["transcript_id"],
            "chrom": iv["chrom"],
            "strand": iv["strand"],
            "tss": iv["tss"],
            "promoter": list(iv["promoter"]),
            "window": list(iv["window"]),
            "n_introns": len(iv["introns"]),
            "n_utr_intervals": len(iv["utr"]),
        }
        for g, iv in gene_ivs.items()
    }
    iv_path.write_text(json.dumps(intervals_out, indent=2) + "\n")

    summary = {
        "definition": {
            "regions": ["intron", "UTR5", "UTR3", "core_promoter_200"],
            "genes": genes,
            "gtf": str(args.gtf),
            "max_af_germline": args.max_af,
            "ab_exclude_median_le": 0.30,
            "primary_site": HAEM_SITE,
            "min_haem_count": args.min_count,
            "min_haem_confirmed": args.min_confirmed,
            "somatic_rule": (
                f"n_haem>={args.min_count} AND n_haem_confirmed>={args.min_confirmed} "
                f"(haemato rows any status; confirmed among haem)"
            ),
            "count_scope": args.count_scope,
        },
        "germline": {
            "n": int(len(germ2)),
            "by_gene": germ2.groupby("gene").size().to_dict() if len(germ2) else {},
            "by_region": germ2["mane_region"].value_counts().to_dict() if len(germ2) else {},
            "n_overlap_dropped": int(len(both)),
        },
        "somatic": {
            "n": int(len(som2)),
            "by_gene": som2.groupby("gene").size().to_dict() if len(som2) else {},
            "by_region": som2["mane_region"].value_counts().to_dict() if len(som2) else {},
        },
        "outputs": {
            "germline": str(germ_path),
            "somatic": str(som_path),
            "combined": str(all_path),
            "intervals": str(iv_path),
        },
    }
    meta_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {germ_path}")
    print(f"Wrote {som_path}")
    print(f"Wrote {all_path}")
    print(f"Wrote {meta_path}")
    print(
        f"TOTAL combined n={len(combined)} "
        f"(germline={len(germ2)}, somatic={len(som2)})"
    )


if __name__ == "__main__":
    main()
