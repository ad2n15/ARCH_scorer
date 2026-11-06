#!/usr/bin/env python3
"""Paper-style SM (MW vs rs2276599 + FDR q<0.05 + Hedges g>=0.8) vs AS_pab method.

Replicates (on gnomAD v4.1 WGS sites histograms, ARCH MANE noncoding windows):
  one-sided Mann–Whitney of candidate AB hist vs germline DNMT3A rs2276599;
  BH FDR; keep q < 0.05 and Hedges g >= 0.8 (g = (mean_ref - mean_cand)/s_pooled * J,
  so g>0 means candidate AB lower than reference).

Compares to existing method:
  AS_pab_max < 0.05 and median AB in (0.05, 0.30].

Requires: module load samtools

Example:
  cd paper/sm_ch_finetune
  python3 scripts/65_compare_paper_mw_sm_vs_aspab.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent

ARCH21 = [
    "ASXL1", "ASXL2", "CBL", "DNMT3A", "ETNK1", "GNAS", "GNB1", "IDH1", "IDH2",
    "JAK2", "KIT", "KRAS", "MPL", "NRAS", "PPM1D", "RUNX1", "SETBP1", "SF3B1",
    "SRSF2", "TET2", "TP53",
]

# Paper reference (gnomAD v4 sites; same rsID / alleles as paper GRCh37 coord)
REF_RSID = "rs2276599"
REF_CHROM = "2"
REF_POS = 25247044
REF_REF = "C"
REF_ALT = "T"
N_BINS = 20


def _load_mod61():
    path = PROJECT / "scripts/61_build_arch_mane_noncoding_lists.py"
    spec = importlib.util.spec_from_file_location("arch61", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _find_tabix() -> str:
    import os
    import shutil

    for cand in (
        os.environ.get("TABIX_BIN"),
        shutil.which("tabix"),
        __import__("os").environ.get("TABIX", "tabix"),
    ):
        if cand and Path(cand).is_file():
            return cand
    raise SystemExit("tabix not found — run: module load samtools")


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


def _parse_hist(freq: str | None) -> np.ndarray | None:
    if not freq or freq in (".", "NA"):
        return None
    try:
        counts = np.array([int(x) for x in freq.split("|")], dtype=np.int64)
    except ValueError:
        return None
    if counts.size < 2 or int(counts.sum()) <= 0:
        return None
    return counts


def _centers(n_bins: int) -> np.ndarray:
    return np.array([(i + 0.5) / n_bins for i in range(n_bins)], dtype=float)


def _expand(counts: np.ndarray, centers: np.ndarray) -> np.ndarray:
    n = min(len(counts), len(centers))
    return np.repeat(centers[:n], counts[:n].astype(int))


def _hist_mean_sd_med(counts: np.ndarray, centers: np.ndarray) -> tuple[float, float, float, int]:
    n = int(counts.sum())
    c = counts.astype(float)
    ctr = centers[: len(counts)]
    if len(ctr) < len(c):
        c = c[: len(ctr)]
    mean = float(np.dot(ctr, c) / n)
    var = float(np.dot(c, (ctr - mean) ** 2) / n)
    sd = math.sqrt(var) if var > 0 else 0.0
    target = n / 2.0
    cum = 0.0
    med = float(ctr[-1])
    for ci, mid in zip(c, ctr):
        cum += ci
        if cum >= target:
            med = float(mid)
            break
    return mean, sd, med, n


def _hedges_g_lower(mean_c: float, sd_c: float, n_c: int, mean_r: float, sd_r: float, n_r: int) -> float | None:
    """Positive g => candidate mean AB lower than reference (paper direction)."""
    if n_c < 2 or n_r < 2:
        return None
    df = n_c + n_r - 2
    if df <= 0:
        return None
    sp2 = ((n_c - 1) * sd_c**2 + (n_r - 1) * sd_r**2) / df
    if sp2 <= 0:
        return None
    d = (mean_r - mean_c) / math.sqrt(sp2)
    j = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    return float(d * j)


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    q = np.full(len(pvals), np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if not ok.any():
        return q
    p = pvals[ok]
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_raw = ranked * m / (np.arange(1, m + 1))
    q_rev = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_rev = np.clip(q_rev, 0, 1)
    out = np.empty(m, dtype=float)
    out[order] = q_rev
    q[ok] = out
    return q


def _parse_dp(im: dict[str, str], allele_i: int) -> int | None:
    as_vdp = im.get("AS_VarDP", "").split(",") if im.get("AS_VarDP") else []
    if allele_i < len(as_vdp) and as_vdp[allele_i] not in ("", "."):
        try:
            return int(float(as_vdp[allele_i]))
        except ValueError:
            pass
    for key in ("AS_VarDP", "VarDP"):
        if key in im and im[key] not in (".", ""):
            try:
                return int(float(str(im[key]).split(",")[0]))
            except ValueError:
                continue
    return None


def _fetch_variant_line(tabix: str, vcf: Path, chrom: str, pos: int) -> str | None:
    for contig in (f"chr{chrom}", chrom):
        proc = subprocess.run(
            [tabix, str(vcf), f"{contig}:{pos}-{pos}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[0]
    return None


def _load_reference(tabix: str, gnomad_dir: Path, pattern: str) -> tuple[np.ndarray, float, float, int, str]:
    vcf = gnomad_dir / pattern.format(chrom=REF_CHROM)
    if not vcf.exists():
        raise SystemExit(f"Missing reference VCF: {vcf}")
    line = _fetch_variant_line(tabix, vcf, REF_CHROM, REF_POS)
    if not line:
        raise SystemExit(f"Could not fetch {REF_RSID} at {REF_CHROM}:{REF_POS}")
    cols = line.split("\t")
    rsid, ref, alt, info = cols[2], cols[3], cols[4], cols[7]
    if REF_ALT not in alt.split(","):
        raise SystemExit(f"Reference allele mismatch at {REF_POS}: {ref}>{alt}")
    im = _info_map(info)
    counts = _parse_hist(im.get("ab_hist_alt_bin_freq"))
    if counts is None:
        raise SystemExit(f"No ab_hist for {REF_RSID}")
    centers = _centers(max(N_BINS, len(counts)))
    mean, sd, _med, n = _hist_mean_sd_med(counts, centers)
    samples = _expand(counts, centers)
    print(
        f"Reference {REF_RSID} ({rsid}) {REF_CHROM}:{REF_POS} {ref}>{REF_ALT}: "
        f"n_hist={n} mean_AB={mean:.4f} sd={sd:.4f}",
        flush=True,
    )
    return samples, mean, sd, n, im.get("ab_hist_alt_bin_freq", "")


def _fetch_candidates(
    tabix: str,
    vcf: Path,
    gene: str,
    iv: dict,
    mod,
    *,
    max_af: float,
    min_dp: int,
    min_ab_n: int,
) -> list[dict]:
    lo, hi = iv["window"]
    chrom = iv["chrom"]
    lines: list[str] = []
    last_err = None
    for contig in (f"chr{chrom}", chrom):
        proc = subprocess.run(
            [tabix, str(vcf), f"{contig}:{lo}-{hi}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            last_err = proc.stderr.strip()
            continue
        lines = proc.stdout.splitlines()
        break
    else:
        if last_err:
            print(f"  WARNING tabix {gene}: {last_err}", flush=True)
        return []

    centers = _centers(N_BINS)
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
        im = _info_map(info)
        alts = alt.split(",")
        afs = im.get("AF", "").split(",") if im.get("AF") else []
        pabs = im.get("AS_pab_max", "").split(",") if im.get("AS_pab_max") else []
        hist = _parse_hist(im.get("ab_hist_alt_bin_freq"))
        if hist is None:
            continue
        # paper: exclusive first bin (0.00–0.05) = artifact
        if int(hist[1:].sum()) == 0:
            continue
        mean, sd, med, n = _hist_mean_sd_med(hist, centers)
        if n <= min_ab_n:
            continue
        for i, a in enumerate(alts):
            ref_u, a_u = ref.upper(), a.upper()
            if len(ref_u) != 1 or len(a_u) != 1:
                continue
            if ref_u not in "ACGT" or a_u not in "ACGT":
                continue
            region = mod.classify_pos(pos, iv)
            if region is None:
                continue
            try:
                af = float(afs[i]) if i < len(afs) and afs[i] not in ("", ".") else float("nan")
            except ValueError:
                af = float("nan")
            if not (pd.isna(af) or (af > 0 and af < max_af)):
                continue
            dp = _parse_dp(im, i)
            if dp is None or dp < min_dp:
                continue
            try:
                pab = float(pabs[i]) if i < len(pabs) and pabs[i] not in ("", ".") else float("nan")
            except ValueError:
                pab = float("nan")
            chrom_n = str(c).replace("chr", "")
            out.append(
                {
                    "variant_id": f"{chrom_n}-{pos}-{ref_u}-{a_u}",
                    "chrom": chrom_n,
                    "pos": pos,
                    "ref": ref_u,
                    "alt": a_u,
                    "gene": gene,
                    "mane_region": region,
                    "mane_transcript": iv["transcript_id"],
                    "rsid": None if rsid == "." else rsid,
                    "af": af,
                    "dp_as_vardp": dp,
                    "ab_median": med,
                    "ab_mean": mean,
                    "ab_sd": sd,
                    "ab_n_carriers_hist": n,
                    "ab_hist_alt_bin_freq": im.get("ab_hist_alt_bin_freq"),
                    "AS_pab_max": pab,
                    "filter": "PASS",
                    "_hist_counts": hist,
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtf", default=str(PROJECT / "data/reference/gencode.v46.annotation.gtf"))
    p.add_argument("--gnomad-dir", default=str(ROOT))
    p.add_argument("--gnomad-pattern", default="gnomad.genomes.v4.1.1.sites.chr{chrom}.vcf.bgz")
    p.add_argument("--max-af", type=float, default=0.01)
    p.add_argument("--min-dp", type=int, default=10)
    p.add_argument("--min-ab-n", type=int, default=5, help="Paper: AB info from more than five individuals")
    p.add_argument("--fdr-q", type=float, default=0.05)
    p.add_argument("--hedges-g-min", type=float, default=0.8)
    p.add_argument("--pab-max", type=float, default=0.05)
    p.add_argument("--ab-artifact-max", type=float, default=0.05)
    p.add_argument("--ab-max-skew", type=float, default=0.30)
    p.add_argument("--genes", default=",".join(ARCH21))
    p.add_argument("--genes-only", default="")
    p.add_argument(
        "--out-dir",
        default=str(PROJECT / "data/arch_mane_noncoding_gnomad_ab"),
    )
    args = p.parse_args()

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    if args.genes_only:
        want = {g.strip() for g in args.genes_only.split(",") if g.strip()}
        genes = [g for g in genes if g in want]

    mod = _load_mod61()
    tabix = _find_tabix()
    centers = _centers(N_BINS)

    ref_samples, ref_mean, ref_sd, ref_n, ref_hist = _load_reference(
        tabix, Path(args.gnomad_dir), args.gnomad_pattern
    )

    gtf = Path(args.gtf)
    mane: dict[str, str] = {}
    want = set(genes)
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
    missing = [g for g in genes if g not in mane]
    if missing:
        raise SystemExit(f"MANE missing: {missing}")

    gene_ivs = {g: mod.load_mane_noncoding_intervals(gtf, mane[g]) for g in genes}
    gnomad_dir = Path(args.gnomad_dir)

    all_rows: list[dict] = []
    for g in genes:
        chrom = gene_ivs[g]["chrom"]
        vcf = gnomad_dir / args.gnomad_pattern.format(chrom=chrom)
        if not vcf.exists():
            raise SystemExit(f"Missing VCF: {vcf}")
        print(f"[{g}] fetching …", flush=True)
        part = _fetch_candidates(
            tabix,
            vcf,
            g,
            gene_ivs[g],
            mod,
            max_af=args.max_af,
            min_dp=args.min_dp,
            min_ab_n=args.min_ab_n,
        )
        print(f"  candidates (PASS/AF/DP/AB_n>{args.min_ab_n}/not-bin0-only): {len(part)}", flush=True)
        all_rows.extend(part)

    if not all_rows:
        raise SystemExit("No candidates")

    # drop duplicate variant_ids (keep first)
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in all_rows:
        vid = r["variant_id"]
        if vid in seen:
            continue
        seen.add(vid)
        uniq.append(r)
    all_rows = uniq

    print(f"Testing Mann–Whitney vs {REF_RSID} for n={len(all_rows)} …", flush=True)
    pvals = np.empty(len(all_rows), dtype=float)
    gvals = np.empty(len(all_rows), dtype=float)
    for i, r in enumerate(all_rows):
        hist = r.pop("_hist_counts")
        cand = _expand(hist, centers)
        # one-sided: candidate AB stochastically less than reference
        try:
            res = mannwhitneyu(cand, ref_samples, alternative="less", method="asymptotic")
            pvals[i] = float(res.pvalue)
        except ValueError:
            pvals[i] = np.nan
        g = _hedges_g_lower(
            float(r["ab_mean"]),
            float(r["ab_sd"]),
            int(r["ab_n_carriers_hist"]),
            ref_mean,
            ref_sd,
            ref_n,
        )
        gvals[i] = np.nan if g is None else g
        if (i + 1) % 25000 == 0:
            print(f"  tested {i+1}/{len(all_rows)}", flush=True)

    qvals = _bh_qvalues(pvals)
    df = pd.DataFrame(all_rows)
    df["mw_pvalue_less_vs_rs2276599"] = pvals
    df["mw_fdr_q"] = qvals
    df["hedges_g_vs_rs2276599"] = gvals  # >0 => lower AB than ref

    # Paper SM
    paper_sm = (
        df["mw_fdr_q"].notna()
        & (df["mw_fdr_q"] < args.fdr_q)
        & df["hedges_g_vs_rs2276599"].notna()
        & (df["hedges_g_vs_rs2276599"] >= args.hedges_g_min)
    )
    # Your method (same as script 63)
    your_sm = (
        df["AS_pab_max"].notna()
        & (df["AS_pab_max"] < args.pab_max)
        & df["ab_median"].notna()
        & (df["ab_median"] > args.ab_artifact_max)
        & (df["ab_median"] <= args.ab_max_skew)
    )
    df["sm_paper_mw"] = paper_sm
    df["sm_aspab"] = your_sm
    df["sm_concordance"] = np.where(
        paper_sm & your_sm,
        "both",
        np.where(paper_sm & ~your_sm, "paper_only", np.where(~paper_sm & your_sm, "aspab_only", "neither")),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_path = out_dir / "mane_noncoding_gnomad_somatic_paper_mw_sm.csv"
    aspab_path = out_dir / "mane_noncoding_gnomad_somatic_aspab_sm_rescored.csv"
    both_path = out_dir / "mane_noncoding_gnomad_sm_paper_aspab_concordance.csv"
    all_path = out_dir / "mane_noncoding_gnomad_all_candidates_paper_mw_metrics.csv"
    summary_path = out_dir / "paper_mw_vs_aspab_summary.json"

    df_paper = df.loc[paper_sm].copy()
    df_aspab = df.loc[your_sm].copy()
    n_both = int((paper_sm & your_sm).sum())
    n_paper = int(paper_sm.sum())
    n_aspab = int(your_sm.sum())
    n_paper_only = int((paper_sm & ~your_sm).sum())
    n_aspab_only = int((~paper_sm & your_sm).sum())

    df.drop(columns=[], errors="ignore").to_csv(all_path, index=False)
    df_paper.to_csv(paper_path, index=False)
    df_aspab.to_csv(aspab_path, index=False)
    df.loc[paper_sm | your_sm].to_csv(both_path, index=False)

    summary = {
        "reference": {
            "rsid": REF_RSID,
            "chrom": REF_CHROM,
            "pos": REF_POS,
            "alleles": f"{REF_REF}>{REF_ALT}",
            "n_hist": ref_n,
            "mean_ab": ref_mean,
            "sd_ab": ref_sd,
            "ab_hist": ref_hist,
        },
        "paper_rule": f"mw_fdr_q < {args.fdr_q} AND hedges_g_vs_rs2276599 >= {args.hedges_g_min}",
        "aspab_rule": (
            f"AS_pab_max < {args.pab_max} AND "
            f"{args.ab_artifact_max} < ab_median <= {args.ab_max_skew}"
        ),
        "n_candidates_tested": int(len(df)),
        "n_paper_sm": n_paper,
        "n_aspab_sm": n_aspab,
        "n_both": n_both,
        "n_paper_only": n_paper_only,
        "n_aspab_only": n_aspab_only,
        "jaccard": (n_both / (n_paper + n_aspab - n_both)) if (n_paper + n_aspab - n_both) else None,
        "paper_by_gene": df_paper.groupby("gene").size().to_dict() if n_paper else {},
        "aspab_by_gene": df_aspab.groupby("gene").size().to_dict() if n_aspab else {},
        "outputs": {
            "all_metrics": str(all_path),
            "paper_sm": str(paper_path),
            "aspab_sm": str(aspab_path),
            "concordance_sm_union": str(both_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Concordance ===", flush=True)
    print(f"Candidates tested: {len(df)}", flush=True)
    print(f"Paper MW SM (q<{args.fdr_q}, g>={args.hedges_g_min}): {n_paper}", flush=True)
    print(f"AS_pab SM: {n_aspab}", flush=True)
    print(f"Both: {n_both}  paper_only: {n_paper_only}  aspab_only: {n_aspab_only}", flush=True)
    if summary["jaccard"] is not None:
        print(f"Jaccard: {summary['jaccard']:.4f}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {paper_path}", flush=True)
    print(f"Wrote {aspab_path}", flush=True)


if __name__ == "__main__":
    main()
