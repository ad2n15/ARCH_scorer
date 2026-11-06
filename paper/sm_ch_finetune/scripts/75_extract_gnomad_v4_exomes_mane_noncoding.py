#!/usr/bin/env python3
"""Apply MANE noncoding germline filters to gnomAD v4.1.1 *exomes* and overlap with WGS.

Filters (match pooled WGS germline):
  PASS, SNV, 0 < AF < 0.01, DP>=10 (AS_VarDP/VarDP),
  median AB from ab_hist_alt_bin_freq > 0.05, ab_n_carriers_hist >= 5,
  in MANE intron/UTR/promoter for ARCH21.

Exome VCFs: gnomad.exomes.v4.1.1.sites.chr{N}.vcf.bgz under alphagenome/.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
VCF_PAT = ROOT / "gnomad.exomes.v4.1.1.sites.chr{chrom}.vcf.bgz"
OUT = PROJECT / "data/arch_mane_noncoding_gnomad_exomes_v4"
WGS_GERM = (
    PROJECT
    / "data/arch_mane_noncoding_pooled_sm/arch_mane_noncoding_germline_abn_ge5.csv"
)
WGS_POOL = (
    PROJECT
    / "data/arch_mane_noncoding_pooled_sm"
    / "arch_mane_noncoding_germline_plus_somatic_pooled.csv"
)
WGS_PAPER = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_ab_minABn2"
    / "mane_noncoding_gnomad_somatic_paper_mw_abn_gt5.csv"
)


def _tabix() -> str:
    for cand in (
        __import__("os").environ.get("TABIX", "tabix"),
        "tabix",
    ):
        p = Path(cand) if cand.startswith("/") else None
        if p and p.is_file():
            return str(p)
        which = subprocess.run(["which", cand], capture_output=True, text=True)
        if which.returncode == 0 and which.stdout.strip():
            return which.stdout.strip()
    raise SystemExit("tabix not found")


def _info_map(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in info.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        elif part:
            out[part] = "1"
    return out


def _ab_stats(freq: str | None) -> tuple[float | None, int]:
    if not freq:
        return None, 0
    try:
        counts = [int(x) for x in freq.split("|")]
    except ValueError:
        return None, 0
    n = sum(counts)
    if n <= 0:
        return None, 0
    # 20 bins of width 0.05 centered at 0.025, 0.075, ...
    mid = [(i + 0.5) / len(counts) for i in range(len(counts))]
    mean = sum(c * m for c, m in zip(counts, mid)) / n
    # median
    target = (n + 1) / 2.0
    cum = 0
    med = None
    for c, m in zip(counts, mid):
        cum += c
        if cum >= target:
            med = m
            break
    return (float(med) if med is not None else float(mean)), n


def _parse_dp(im: dict[str, str]) -> int | None:
    for key in ("AS_VarDP", "VarDP"):
        if key in im and im[key] not in (".", ""):
            try:
                return int(float(str(im[key]).split(",")[0]))
            except ValueError:
                continue
    return None


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "m61", PROJECT / "scripts/61_build_arch_mane_noncoding_lists.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    tabix = _tabix()
    gtf = PROJECT / "data/reference/gencode.v46.annotation.gtf"
    mane = mod.load_mane_select_map(gtf, mod.ARCH21)
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for gene in mod.ARCH21:
        iv = mod.load_mane_noncoding_intervals(gtf, mane[gene])
        chrom = iv["chrom"]
        vcf = Path(str(VCF_PAT).format(chrom=chrom))
        if not vcf.is_file():
            print(f"MISSING VCF {vcf}", flush=True)
            continue
        lo, hi = iv["window"]
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
        n_pass = 0
        for line in lines:
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            _c, pos_s, rsid, ref, alt, _q, filt, info = cols[:8]
            if filt != "PASS":
                continue
            pos = int(pos_s)
            im = _info_map(info)
            alts = alt.split(",")
            afs = im.get("AF", "").split(",") if im.get("AF") else []
            as_vdp = im.get("AS_VarDP", "").split(",") if im.get("AS_VarDP") else []
            ab_med, ab_n = _ab_stats(im.get("ab_hist_alt_bin_freq"))
            dp_site = _parse_dp(im)
            for i, a in enumerate(alts):
                ru, au = ref.upper(), a.upper()
                if len(ru) != 1 or len(au) != 1 or ru not in "ACGT" or au not in "ACGT":
                    continue
                region = mod.classify_pos(pos, iv)
                if region is None:
                    continue
                try:
                    af = float(afs[i]) if i < len(afs) and afs[i] not in ("", ".") else float("nan")
                except ValueError:
                    af = float("nan")
                if not (af == af and 0 < af < 0.01):
                    continue
                # allele-specific DP if available else site DP
                dp = None
                if i < len(as_vdp) and as_vdp[i] not in ("", "."):
                    try:
                        dp = int(float(as_vdp[i]))
                    except ValueError:
                        dp = None
                if dp is None:
                    dp = dp_site
                if dp is None or dp < 10:
                    continue
                if ab_med is None or ab_med <= 0.05:
                    continue
                if ab_n < 5:
                    continue
                n_pass += 1
                chrom_n = chrom
                rows.append(
                    {
                        "variant_id": f"{chrom_n}-{pos}-{ru}-{au}",
                        "chrom": chrom_n,
                        "pos": pos,
                        "ref": ru,
                        "alt": au,
                        "gene": gene,
                        "mane_region": region,
                        "af": af,
                        "dp": dp,
                        "ab_median": ab_med,
                        "ab_n_carriers_hist": ab_n,
                        "rsid": None if rsid == "." else rsid,
                        "source": "gnomad_v4_exomes",
                    }
                )
        print(f"{gene:8s} tabix_lines={len(lines):6d} kept={n_pass}", flush=True)

    wes = pd.DataFrame(rows).drop_duplicates("variant_id")
    wes_path = OUT / "arch_mane_noncoding_germline_exomes_v4_abn_ge5.csv"
    wes.to_csv(wes_path, index=False)

    wgs_g = pd.read_csv(WGS_GERM, usecols=["variant_id", "gene", "mane_region", "label"])
    wgs_g["variant_id"] = wgs_g["variant_id"].astype(str)
    pool = pd.read_csv(WGS_POOL, usecols=["variant_id", "gene", "label", "source", "mane_region"])
    pool["variant_id"] = pool["variant_id"].astype(str)
    # gnomAD-derived WGS alleles currently in pool (germline + paper MW)
    paper = pd.read_csv(WGS_PAPER, usecols=["variant_id"])
    paper["variant_id"] = paper["variant_id"].astype(str)
    wgs_gnomad_ids = set(wgs_g.variant_id) | set(paper.variant_id)
    pool_ids = set(pool.variant_id)
    wes_ids = set(wes.variant_id)

    summary = {
        "filters": (
            "PASS SNV; 0<AF<0.01; DP>=10; median AB>0.05; ab_n>=5; "
            "MANE intron/UTR/promoter ARCH21"
        ),
        "n_wes": int(len(wes)),
        "n_wgs_germline": int(len(wgs_g)),
        "n_wgs_paper_mw": int(len(paper)),
        "n_wgs_gnomad_germ_or_paper": int(len(wgs_gnomad_ids)),
        "n_wgs_full_pool": int(len(pool_ids)),
        "overlap_wes_wgs_germline": int(len(wes_ids & set(wgs_g.variant_id))),
        "overlap_wes_wgs_gnomad": int(len(wes_ids & wgs_gnomad_ids)),
        "overlap_wes_full_pool": int(len(wes_ids & pool_ids)),
        "wes_only_vs_wgs_germline": int(len(wes_ids - set(wgs_g.variant_id))),
        "wes_only_vs_wgs_gnomad": int(len(wes_ids - wgs_gnomad_ids)),
        "wes_only_vs_full_pool": int(len(wes_ids - pool_ids)),
        "wgs_germline_only": int(len(set(wgs_g.variant_id) - wes_ids)),
        "wes_by_region": {
            str(k): int(v) for k, v in wes["mane_region"].value_counts().items()
        },
        "wes_by_gene": {
            str(k): int(v) for k, v in wes["gene"].value_counts().sort_index().items()
        },
        "wes_file": str(wes_path),
    }
    (OUT / "wes_vs_wgs_overlap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {wes_path}", flush=True)


if __name__ == "__main__":
    main()
