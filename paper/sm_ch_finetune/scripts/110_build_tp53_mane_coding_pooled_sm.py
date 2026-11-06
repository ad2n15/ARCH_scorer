#!/usr/bin/env python3
"""Build ARCH21 MANE *coding* (CDS) pooled-SM cohorts with the same filters as noncoding.

Region: MANE Select CDS only (Ellingford coding exons; NOT UTR/intron/promoter).

gnomAD v4.1.1 WGS ∪ WES (same as noncoding):
  PASS, SNV, 0 < AF < 0.01, DP≥10, median AB>0.05 (exclude bin-0-only),
  ab_n_carriers_hist > 5.

Somatic gnomAD = paper MW (vs rs2276599):
  one-sided MW, BH q < 0.05, Hedges g ≥ 0.8, ab_n > 5.

Germline gnomAD = ab_n > 5 AND mw_fdr_q ≥ 0.05 (not AB-skewed).

Somatic catalogues:
  COSMIC MutantCensus haem≥1 & confirmed≥1 in each gene's CDS
  (+ optional IARC TP53 haemato SNVs in CDS; default on)

Overlap: germline ∩ (COSMIC ∪ IARC) dropped from BOTH sides.
Paper MW alleles removed from germline only.

Requires: module load samtools (tabix)

Example:
  cd paper/sm_ch_finetune
  module load samtools
  python3 scripts/110_build_tp53_mane_coding_pooled_sm.py --genes ASXL1,ASXL2,...
  python3 scripts/110_build_tp53_mane_coding_pooled_sm.py --skip-existing
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from data.mane_ellingford import in_intervals, load_mane_cds_exons

HAEM_SITE = "haematopoietic_and_lymphoid_tissue"
CONFIRMED = "Confirmed somatic variant"
REF_RSID = "rs2276599"
REF_CHROM = "2"
REF_POS = 25247044
REF_ALT = "T"
N_BINS = 20
ARCH21 = [
    "ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1",
    "IDH2", "JAK2", "KIT", "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1",
    "SETBP1", "SF3B1", "SRSF2", "TET2", "TP53",
]

COLS = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "label",
    "source",
    "sm_source",
    "mane_region",
    "af",
    "ab_median",
    "ab_n_carriers_hist",
    "AS_pab_max",
    "n_haem",
    "haem_ge3",
]


def _load_m65():
    path = PROJECT / "scripts/65_compare_paper_mw_sm_vs_aspab.py"
    spec = importlib.util.spec_from_file_location("m65", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_m61():
    path = PROJECT / "scripts/61_build_arch_mane_noncoding_lists.py"
    spec = importlib.util.spec_from_file_location("m61", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cds_window(cds_ivs: list[tuple[str, int, int]]) -> tuple[str, int, int]:
    chrom = cds_ivs[0][0]
    lo = min(s for _, s, _ in cds_ivs)
    hi = max(e for _, _, e in cds_ivs)
    return chrom, lo, hi


def _fetch_cds_candidates(
    m65,
    tabix: str,
    vcf: Path,
    gene: str,
    chrom: str,
    lo: int,
    hi: int,
    cds_ivs: list[tuple[str, int, int]],
    *,
    max_af: float,
    min_dp: int,
    min_ab_n: int,
    source: str,
) -> list[dict]:
    """Same gnomAD filters as script 65, but keep only positions in MANE CDS."""
    lines: list[str] = []
    for contig in (f"chr{chrom}", chrom):
        proc = subprocess.run(
            [tabix, str(vcf), f"{contig}:{lo}-{hi}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            lines = proc.stdout.splitlines()
            break
    centers = m65._centers(N_BINS)
    out: list[dict] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        c, pos_s, rsid, ref, alt, _q, filt, info = cols[:8]
        if filt != "PASS":
            continue
        pos = int(pos_s)
        if not in_intervals(chrom, pos, cds_ivs):
            continue
        im = m65._info_map(info)
        alts = alt.split(",")
        afs = im.get("AF", "").split(",") if im.get("AF") else []
        pabs = im.get("AS_pab_max", "").split(",") if im.get("AS_pab_max") else []
        hist = m65._parse_hist(im.get("ab_hist_alt_bin_freq"))
        if hist is None:
            continue
        if int(hist[1:].sum()) == 0:  # exclusive first bin = artifact
            continue
        mean, sd, med, n = m65._hist_mean_sd_med(hist, centers)
        if n <= min_ab_n:  # paper: more than five → keep n > 5
            continue
        if med is None or med <= 0.05:
            continue
        for i, a in enumerate(alts):
            ru, au = ref.upper(), a.upper()
            if len(ru) != 1 or len(au) != 1 or ru not in "ACGT" or au not in "ACGT":
                continue
            try:
                af = float(afs[i]) if i < len(afs) and afs[i] not in ("", ".") else float("nan")
            except ValueError:
                af = float("nan")
            if not (af == af and 0 < af < max_af):
                continue
            dp = m65._parse_dp(im, i)
            if dp is None or dp < min_dp:
                continue
            try:
                pab = (
                    float(pabs[i])
                    if i < len(pabs) and pabs[i] not in ("", ".")
                    else float("nan")
                )
            except ValueError:
                pab = float("nan")
            chrom_n = str(c).replace("chr", "")
            out.append(
                {
                    "variant_id": f"{chrom_n}-{pos}-{ru}-{au}",
                    "chrom": chrom_n,
                    "pos": pos,
                    "ref": ru,
                    "alt": au,
                    "gene": gene,
                    "mane_region": "CDS",
                    "rsid": None if rsid == "." else rsid,
                    "af": af,
                    "dp": dp,
                    "ab_median": med,
                    "ab_mean": mean,
                    "ab_sd": sd,
                    "ab_n_carriers_hist": n,
                    "ab_hist_alt_bin_freq": im.get("ab_hist_alt_bin_freq"),
                    "AS_pab_max": pab,
                    "source": source,
                    "_hist_counts": hist,
                }
            )
    return out


def _run_paper_mw(
    m65,
    rows: list[dict],
    ref_samples: np.ndarray,
    ref_mean: float,
    ref_sd: float,
    ref_n: int,
    *,
    fdr_q: float,
    hedges_g_min: float,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    centers = m65._centers(N_BINS)
    pvals, gvals = [], []
    for r in rows:
        hist = r["_hist_counts"]
        mean_c, sd_c, _med, n_c = m65._hist_mean_sd_med(hist, centers)
        samples = m65._expand(hist, centers)
        try:
            # candidate AB lower than reference
            stat = mannwhitneyu(samples, ref_samples, alternative="less")
            p = float(stat.pvalue)
        except ValueError:
            p = float("nan")
        g = m65._hedges_g_lower(mean_c, sd_c, n_c, ref_mean, ref_sd, ref_n)
        pvals.append(p)
        gvals.append(g)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_hist_counts"} for r in rows])
    df["mw_pval"] = pvals
    df["mw_fdr_q"] = m65._bh_qvalues(np.asarray(pvals, dtype=float))
    df["hedges_g_vs_rs2276599"] = gvals
    return df


def _extract_cosmic_cds(
    *,
    gene: str,
    mutations: Path,
    classification: Path,
    cds_ivs: list[tuple[str, int, int]],
    chrom: str,
    lo: int,
    hi: int,
    min_haem: int,
    min_haem_conf: int,
) -> pd.DataFrame:
    clf = pd.read_csv(classification, sep="\t", low_memory=False)
    haem_pheno = set(
        clf.loc[clf["PRIMARY_SITE"].astype(str) == HAEM_SITE, "COSMIC_PHENOTYPE_ID"].astype(
            str
        )
    )
    usecols = {
        "GENE_SYMBOL",
        "COSMIC_PHENOTYPE_ID",
        "GENOMIC_MUTATION_ID",
        "CHROMOSOME",
        "GENOME_START",
        "GENOMIC_WT_ALLELE",
        "GENOMIC_MUT_ALLELE",
        "MUTATION_SOMATIC_STATUS",
    }
    kept = []
    n_raw = 0
    for chunk in pd.read_csv(
        mutations,
        sep="\t",
        usecols=lambda c: c in usecols,
        chunksize=500_000,
        dtype=str,
        low_memory=True,
    ):
        n_raw += len(chunk)
        chunk = chunk.loc[chunk["GENE_SYMBOL"].astype(str) == gene]
        if chunk.empty:
            continue
        chunk["CHROMOSOME"] = chunk["CHROMOSOME"].astype(str).str.replace("^chr", "", regex=True)
        chunk["GENOME_START"] = pd.to_numeric(chunk["GENOME_START"], errors="coerce")
        chunk = chunk.loc[
            (chunk["CHROMOSOME"] == str(chrom))
            & chunk["GENOME_START"].notna()
            & (chunk["GENOME_START"] >= lo)
            & (chunk["GENOME_START"] <= hi)
        ]
        if chunk.empty:
            continue
        for _, r in chunk.iterrows():
            pos = int(r["GENOME_START"])
            if not in_intervals(chrom, pos, cds_ivs):
                continue
            ref = str(r["GENOMIC_WT_ALLELE"]).upper()
            alt = str(r["GENOMIC_MUT_ALLELE"]).upper()
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue
            kept.append(
                {
                    "GENOMIC_MUTATION_ID": r["GENOMIC_MUTATION_ID"],
                    "COSMIC_PHENOTYPE_ID": str(r["COSMIC_PHENOTYPE_ID"]),
                    "MUTATION_SOMATIC_STATUS": str(r["MUTATION_SOMATIC_STATUS"]),
                    "variant_id": f"{chrom}-{pos}-{ref}-{alt}",
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene": gene,
                    "mane_region": "CDS",
                }
            )
    print(f"COSMIC MutantCensus scanned={n_raw:,} {gene}-CDS SNV rows={len(kept):,}", flush=True)
    if not kept:
        return pd.DataFrame()
    win = pd.DataFrame(kept)
    haem = win.loc[win["COSMIC_PHENOTYPE_ID"].isin(haem_pheno)].copy()
    n_haem = haem.groupby("GENOMIC_MUTATION_ID").size().rename("n_haem")
    n_conf = (
        haem.loc[haem["MUTATION_SOMATIC_STATUS"] == CONFIRMED]
        .groupby("GENOMIC_MUTATION_ID")
        .size()
        .rename("n_haem_confirmed")
    )
    stats = pd.concat([n_haem, n_conf], axis=1).fillna(0).astype(int)
    keep_ids = stats.index[
        (stats["n_haem"] >= min_haem) & (stats["n_haem_confirmed"] >= min_haem_conf)
    ]
    out = (
        win.loc[win["GENOMIC_MUTATION_ID"].isin(keep_ids)]
        .drop_duplicates("variant_id")
        .merge(stats.reset_index(), on="GENOMIC_MUTATION_ID", how="left")
    )
    out = out.groupby("variant_id", as_index=False).agg(
        {
            "chrom": "first",
            "pos": "first",
            "ref": "first",
            "alt": "first",
            "gene": "first",
            "mane_region": "first",
            "n_haem": "max",
            "n_haem_confirmed": "max",
        }
    )
    print(
        f"COSMIC {gene} CDS haem≥{min_haem} & conf≥{min_haem_conf}: n={len(out)}",
        flush=True,
    )
    return out


def _iarc_cds(cds_ivs, chrom: str, *, tumor_only: bool = True) -> pd.DataFrame:
    path = PROJECT / "data/pilot_iarc_tp53_hemato/iarc_tp53_hemato_unique_snvs.csv"
    if not path.is_file():
        print(f"IARC unique SNVs missing: {path} — skipping", flush=True)
        return pd.DataFrame()
    df = pd.read_csv(path)
    if tumor_only and "in_tumor" in df.columns:
        n_pre = len(df)
        df = df.loc[df["in_tumor"].astype(bool)].copy()
        print(f"IARC tumor_only filter: {n_pre} -> {len(df)}", flush=True)
    # expect chrom/pos/ref/alt or variant_id
    if "variant_id" not in df.columns:
        if {"chrom", "pos", "ref", "alt"}.issubset(df.columns):
            df["variant_id"] = (
                df["chrom"].astype(str).str.replace("^chr", "", regex=True)
                + "-"
                + df["pos"].astype(str)
                + "-"
                + df["ref"].astype(str)
                + "-"
                + df["alt"].astype(str)
            )
        else:
            print("IARC file lacks variant columns — skipping", flush=True)
            return pd.DataFrame()
    parts = df["variant_id"].astype(str).str.split("-", n=3, expand=True)
    df["chrom"] = parts[0].str.replace("^chr", "", regex=True)
    df["pos"] = pd.to_numeric(parts[1], errors="coerce")
    df["ref"] = parts[2].astype(str).str.upper()
    df["alt"] = parts[3].astype(str).str.upper()
    df = df.loc[
        (df["chrom"] == str(chrom))
        & df["pos"].notna()
        & df["ref"].str.len().eq(1)
        & df["alt"].str.len().eq(1)
    ].copy()
    df = df.loc[
        df.apply(lambda r: in_intervals(chrom, int(r["pos"]), cds_ivs), axis=1)
    ]
    df["gene"] = "TP53"
    df["mane_region"] = "CDS"
    df = df.drop_duplicates("variant_id")
    print(f"IARC TP53 haemato in CDS: n={len(df)}", flush=True)
    return df[
        ["variant_id", "chrom", "pos", "ref", "alt", "gene", "mane_region"]
    ]


def _row(r: pd.Series, gene: str, label: str, source: str, sm_source: str) -> dict:
    return {
        "variant_id": str(r["variant_id"]),
        "chrom": str(r["chrom"]).replace("chr", ""),
        "pos": int(r["pos"]),
        "ref": str(r["ref"]),
        "alt": str(r["alt"]),
        "gene": gene,
        "label": label,
        "source": source,
        "sm_source": sm_source,
        "mane_region": "CDS",
        "af": r["af"] if "af" in r.index else None,
        "ab_median": r["ab_median"] if "ab_median" in r.index else None,
        "ab_n_carriers_hist": r["ab_n_carriers_hist"]
        if "ab_n_carriers_hist" in r.index
        else None,
        "AS_pab_max": r["AS_pab_max"] if "AS_pab_max" in r.index else None,
    }


def _build_gene(args, m65, m61, tabix: str, gene: str, out_dir: Path) -> dict:
    gtf = Path(args.gtf)
    cds_map, _exons = load_mane_cds_exons(gtf, {gene})
    cds_ivs = cds_map.get(gene, [])
    if not cds_ivs:
        raise SystemExit(f"No MANE CDS intervals for {gene}")
    chrom, lo, hi = _cds_window(cds_ivs)
    mane = m61.load_mane_select_map(gtf, [gene])
    print(
        f"{gene} MANE={mane.get(gene)} CDS intervals={len(cds_ivs)} "
        f"chr{chrom}:{lo}-{hi}",
        flush=True,
    )

    metrics_dir = out_dir / "metrics" / gene
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- gnomAD WGS + WES candidates in CDS ---
    gnomad_dir = Path(args.gnomad_dir)
    wgs_vcf = gnomad_dir / args.wgs_pattern.format(chrom=chrom)
    wes_vcf = gnomad_dir / args.wes_pattern.format(chrom=chrom)
    if not wgs_vcf.is_file():
        raise SystemExit(f"Missing WGS VCF: {wgs_vcf}")
    if not wes_vcf.is_file():
        raise SystemExit(f"Missing WES VCF: {wes_vcf}")

    print("Fetching WGS CDS candidates…", flush=True)
    wgs_rows = _fetch_cds_candidates(
        m65,
        tabix,
        wgs_vcf,
        gene,
        chrom,
        lo,
        hi,
        cds_ivs,
        max_af=args.max_af,
        min_dp=args.min_dp,
        min_ab_n=args.min_ab_n,
        source="gnomad_v4_genomes",
    )
    print(f"  WGS CDS candidates: {len(wgs_rows)}", flush=True)
    print("Fetching WES CDS candidates…", flush=True)
    wes_rows = _fetch_cds_candidates(
        m65,
        tabix,
        wes_vcf,
        gene,
        chrom,
        lo,
        hi,
        cds_ivs,
        max_af=args.max_af,
        min_dp=args.min_dp,
        min_ab_n=args.min_ab_n,
        source="gnomad_v4_exomes",
    )
    print(f"  WES CDS candidates: {len(wes_rows)}", flush=True)

    # Reference AB hist from WGS (same as paper / script 65)
    ref_samples, ref_mean, ref_sd, ref_n, _ = m65._load_reference(
        tabix, gnomad_dir, args.wgs_pattern
    )

    print("Paper MW on WGS CDS…", flush=True)
    wgs_mw = _run_paper_mw(
        m65,
        wgs_rows,
        ref_samples,
        ref_mean,
        ref_sd,
        ref_n,
        fdr_q=args.fdr_q,
        hedges_g_min=args.hedges_g_min,
    )
    print("Paper MW on WES CDS…", flush=True)
    # WES reference: use WGS rs2276599 hist (paper uses genomes reference)
    wes_mw = _run_paper_mw(
        m65,
        wes_rows,
        ref_samples,
        ref_mean,
        ref_sd,
        ref_n,
        fdr_q=args.fdr_q,
        hedges_g_min=args.hedges_g_min,
    )

    def _paper(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.loc[
            df["mw_fdr_q"].notna()
            & (df["mw_fdr_q"] < args.fdr_q)
            & df["hedges_g_vs_rs2276599"].notna()
            & (df["hedges_g_vs_rs2276599"] >= args.hedges_g_min)
            & (df["ab_n_carriers_hist"] > args.min_ab_n)
        ].copy()

    wgs_paper = _paper(wgs_mw)
    wes_paper = _paper(wes_mw)
    print(f"WGS paper MW somatic: {len(wgs_paper)}", flush=True)
    print(f"WES paper MW somatic: {len(wes_paper)}", flush=True)

    wgs_mw.to_csv(metrics_dir / "wgs_all_candidates_paper_mw_metrics.csv", index=False)
    wes_mw.to_csv(metrics_dir / "wes_all_candidates_paper_mw_metrics.csv", index=False)
    wgs_paper.to_csv(metrics_dir / "wgs_somatic_paper_mw_abn_gt5.csv", index=False)
    wes_paper.to_csv(metrics_dir / "wes_somatic_paper_mw_abn_gt5.csv", index=False)

    # Germline candidates = all CDS candidates with ab_n>5 (MW filter applied later)
    wgs_germ = wgs_mw.loc[wgs_mw["ab_n_carriers_hist"] > args.min_ab_n].copy() if not wgs_mw.empty else pd.DataFrame()
    wes_germ = wes_mw.loc[wes_mw["ab_n_carriers_hist"] > args.min_ab_n].copy() if not wes_mw.empty else pd.DataFrame()

    # COSMIC MutantCensus CDS
    cos = _extract_cosmic_cds(
        gene=gene,
        mutations=Path(args.cosmic_mutations),
        classification=Path(args.cosmic_classification),
        cds_ivs=cds_ivs,
        chrom=chrom,
        lo=lo,
        hi=hi,
        min_haem=args.min_haem,
        min_haem_conf=args.min_haem_confirmed,
    )
    cos.to_csv(
        metrics_dir
        / f"cosmic_mutantcensus_haem_min{args.min_haem}_ge{args.min_haem_confirmed}conf.csv",
        index=False,
    )

    iarc = pd.DataFrame()
    if gene == "TP53" and not args.no_iarc:
        iarc = _iarc_cds(
            cds_ivs, chrom, tumor_only=not args.include_iarc_germline_only
        )
        if not iarc.empty:
            iarc.to_csv(metrics_dir / "iarc_hemato.csv", index=False)

    # --- assemble ---
    wgs_p_ids = set(wgs_paper.variant_id.astype(str)) if len(wgs_paper) else set()
    wes_p_ids = set(wes_paper.variant_id.astype(str)) if len(wes_paper) else set()
    cos_ids = set(cos.variant_id.astype(str)) if len(cos) else set()
    iarc_ids = set(iarc.variant_id.astype(str)) if len(iarc) else set()
    som_ids = wgs_p_ids | wes_p_ids | cos_ids | iarc_ids

    frames = []
    for df, pri in (
        (iarc.assign(_pri=-1) if len(iarc) else pd.DataFrame(), -1),
        (cos.assign(_pri=0) if len(cos) else pd.DataFrame(), 0),
        (wes_germ.assign(_pri=1) if len(wes_germ) else pd.DataFrame(), 1),
        (wgs_germ.assign(_pri=2) if len(wgs_germ) else pd.DataFrame(), 2),
        (wes_paper.assign(_pri=3) if len(wes_paper) else pd.DataFrame(), 3),
        (wgs_paper.assign(_pri=4) if len(wgs_paper) else pd.DataFrame(), 4),
    ):
        if len(df):
            frames.append(df)
    if not frames:
        raise SystemExit("No variants collected")
    meta = (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values("_pri")
        .drop_duplicates("variant_id", keep="last")
        .set_index("variant_id", drop=False)
    )

    som_rows = []
    for vid in som_ids:
        tags = []
        if vid in wgs_p_ids:
            tags.append("paper_mw_wgs_abn_gt5")
        if vid in wes_p_ids:
            tags.append("paper_mw_wes_abn_gt5")
        if vid in cos_ids:
            tags.append("cosmic_mutantcensus_haem_ge1_conf")
        if vid in iarc_ids:
            tags.append("iarc_hemato_mane_cds")
        som_rows.append(_row(meta.loc[vid], gene, "somatic", "+".join(tags), "+".join(tags)))
    som = pd.DataFrame(som_rows)

    wgs_g_ids = set(wgs_germ.variant_id.astype(str)) if len(wgs_germ) else set()
    wes_g_ids = set(wes_germ.variant_id.astype(str)) if len(wes_germ) else set()
    germ_rows = []
    if len(wgs_germ):
        for _, r in wgs_germ.drop_duplicates("variant_id").iterrows():
            src = (
                "gnomad_v4_genomes+exomes"
                if r.variant_id in wes_g_ids
                else "gnomad_v4_genomes"
            )
            germ_rows.append(_row(r, gene, "germline", src, ""))
    if len(wes_germ):
        for _, r in wes_germ.drop_duplicates("variant_id").iterrows():
            if r.variant_id in wgs_g_ids:
                continue
            germ_rows.append(_row(r, gene, "germline", "gnomad_v4_exomes", ""))
    germ = pd.DataFrame(germ_rows)
    n_pre = len(germ)

    # Prefer WGS MW q when both assays present
    mw_parts = []
    if len(wgs_mw):
        mw_parts.append(wgs_mw[["variant_id", "mw_fdr_q", "hedges_g_vs_rs2276599"]].assign(_pri=1))
    if len(wes_mw):
        mw_parts.append(wes_mw[["variant_id", "mw_fdr_q", "hedges_g_vs_rs2276599"]].assign(_pri=0))
    if mw_parts and len(germ):
        mw_q = (
            pd.concat(mw_parts, ignore_index=True)
            .sort_values("_pri")
            .drop_duplicates("variant_id", keep="last")
        )
        germ = germ.merge(mw_q, on="variant_id", how="left")
        n_untested = int(germ["mw_fdr_q"].isna().sum())
        n_qlt = int((germ["mw_fdr_q"].notna() & (germ["mw_fdr_q"] < args.fdr_q)).sum())
        germ = germ.loc[germ["mw_fdr_q"].notna() & (germ["mw_fdr_q"] >= args.fdr_q)].copy()
        print(
            f"Germline MW filter: pre={n_pre} untested_drop={n_untested} "
            f"q<0.05_drop={n_qlt} keep={len(germ)}",
            flush=True,
        )
        germ = germ.drop(columns=["mw_fdr_q", "hedges_g_vs_rs2276599", "_pri"], errors="ignore")

    ext_ids = cos_ids | iarc_ids
    ambig = set(germ.variant_id) & ext_ids
    if ambig:
        germ = germ.loc[~germ.variant_id.isin(ambig)].copy()
        som = som.loc[~som.variant_id.isin(ambig)].copy()
    print(f"Dropped germline∩(COSMIC∪IARC) both sides: n={len(ambig)}", flush=True)

    paper_ids = wgs_p_ids | wes_p_ids
    n_paper_g = int(germ.variant_id.isin(paper_ids).sum()) if len(germ) else 0
    if n_paper_g:
        germ = germ.loc[~germ.variant_id.isin(paper_ids)].copy()
        print(f"Removed paper-MW from germline only: n={n_paper_g}", flush=True)

    both = set(germ.variant_id) & set(som.variant_id)
    if both:
        raise SystemExit(f"Residual germ∩som: n={len(both)}")

    comb = pd.concat([germ, som], ignore_index=True) if len(germ) or len(som) else pd.DataFrame()
    if len(cos):
        comb = comb.merge(
            cos[["variant_id", "n_haem"]].drop_duplicates("variant_id"),
            on="variant_id",
            how="left",
        )
    else:
        comb["n_haem"] = np.nan
    comb["haem_ge3"] = (comb["n_haem"].fillna(0).astype(float) >= 3).astype(int)
    comb["gene"] = gene

    for c in COLS:
        if c not in comb.columns:
            comb[c] = None
    comb = comb[COLS]

    germ_out = comb.loc[comb.label == "germline"].copy()
    som_out = comb.loc[comb.label == "somatic"].copy()
    germ_out.to_csv(metrics_dir / "germline_abn_gt5.csv", index=False)
    som_out.to_csv(metrics_dir / "somatic_pooled.csv", index=False)
    comb.to_csv(out_dir / "by_gene" / f"{gene}.csv", index=False)

    summary = {
        "gene": gene,
        "mane_transcript": mane.get(gene),
        "region": "MANE_Select_CDS",
        "chrom": chrom,
        "cds_span": [lo, hi],
        "n_cds_intervals": len(cds_ivs),
        "filters": {
            "gnomad": "PASS SNV; 0<AF<0.01; DP>=10; median_AB>0.05; ab_n>5",
            "paper_mw": f"mw_fdr_q<{args.fdr_q} AND hedges_g>={args.hedges_g_min}",
            "germline_mw": f"mw_fdr_q>={args.fdr_q}",
            "cosmic": f"MutantCensus haem>={args.min_haem} confirmed>={args.min_haem_confirmed}",
            "iarc": gene == "TP53" and not args.no_iarc,
            "iarc_tumor_only": not args.include_iarc_germline_only,
        },
        "n_wgs_candidates": int(len(wgs_mw)) if len(wgs_mw) else 0,
        "n_wes_candidates": int(len(wes_mw)) if len(wes_mw) else 0,
        "n_wgs_paper_mw": int(len(wgs_paper)),
        "n_wes_paper_mw": int(len(wes_paper)),
        "n_cosmic": int(len(cos)),
        "n_iarc": int(len(iarc)),
        "n_germline": int(len(germ_out)),
        "n_somatic": int(len(som_out)),
        "n_total": int(len(comb)),
        "somatic_source_counts": som_out["source"].value_counts().astype(int).to_dict()
        if len(som_out)
        else {},
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}/by_gene/{gene}.csv", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes", default=",".join(ARCH21))
    ap.add_argument("--gene", help="Single gene (overrides --genes)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--gtf", default=str(PROJECT / "data/reference/gencode.v46.annotation.gtf"))
    ap.add_argument("--gnomad-dir", default=str(ROOT))
    ap.add_argument(
        "--wgs-pattern",
        default="gnomad.genomes.v4.1.1.sites.chr{chrom}.vcf.bgz",
    )
    ap.add_argument(
        "--wes-pattern",
        default="gnomad.exomes.v4.1.1.sites.chr{chrom}.vcf.bgz",
    )
    ap.add_argument(
        "--cosmic-mutations",
        default=str(ROOT / "Cosmic_MutantCensus_v104_GRCh38.tsv"),
    )
    ap.add_argument(
        "--cosmic-classification",
        default=str(ROOT / "Cosmic_Classification_v104_GRCh38.tsv"),
    )
    ap.add_argument("--max-af", type=float, default=0.01)
    ap.add_argument("--min-dp", type=int, default=10)
    ap.add_argument("--min-ab-n", type=int, default=5)
    ap.add_argument("--fdr-q", type=float, default=0.05)
    ap.add_argument("--hedges-g-min", type=float, default=0.8)
    ap.add_argument("--min-haem", type=int, default=1)
    ap.add_argument("--min-haem-confirmed", type=int, default=1)
    ap.add_argument("--no-iarc", action="store_true")
    ap.add_argument(
        "--include-iarc-germline-only",
        action="store_true",
        help="Keep IARC haemato alleles seen only in the germline file (default: tumour only)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT / "data/arch_mane_coding_pooled_sm"),
    )
    args = ap.parse_args()

    genes = [args.gene.strip().upper()] if args.gene else [
        gene.strip().upper() for gene in args.genes.split(",") if gene.strip()
    ]
    out_dir = Path(args.out_dir)
    by_gene_dir = out_dir / "by_gene"
    by_gene_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "build_summary.json"
    summaries = {}
    if summary_path.is_file():
        summaries = json.loads(summary_path.read_text())
        if "gene" in summaries:
            summaries = {summaries["gene"]: summaries}

    m65 = _load_m65()
    m61 = _load_m61()
    tabix = m65._find_tabix()
    for gene in genes:
        if args.skip_existing and (by_gene_dir / f"{gene}.csv").is_file():
            print(f"{gene}: cohort exists — skipping", flush=True)
            continue
        summaries[gene] = _build_gene(args, m65, m61, tabix, gene, out_dir)
        summary_path.write_text(json.dumps(summaries, indent=2))
    summary_path.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
