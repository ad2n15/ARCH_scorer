"""ARCH21 gene list and MANE noncoding windows."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

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

# CD34+ AlphaGenome track short names used by the api_baseline model
API_TRACK_SHORT = [
    "RNA_SEQ",
    "SPLICE_USAGE",
    "SPLICE_JUNC",
    "DNASE",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me3",
    "H3K9me3",
]

API_TRACK_RAW = {
    "RNA_SEQ": "RNA_SEQ__CL_0001059_polyA_plus_RNA_seq__GeneMaskLFC",
    "SPLICE_USAGE": "SPLICE_SITE_USAGE__usage_CL_0001059_polyA_plus_RNA_seq__GeneMaskSplicing",
    "SPLICE_JUNC": "SPLICE_JUNCTIONS__junction_CL_0001059_polyA_plus_RNA_seq__SpliceJunction",
    "DNASE": "DNASE__CL_0001059_DNase_seq__CenterMask",
    "H3K27ac": "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K27ac__CenterMask",
    "H3K27me3": "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K27me3__CenterMask",
    "H3K36me3": "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K36me3__CenterMask",
    "H3K4me1": "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K4me1__CenterMask",
    "H3K4me3": "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K4me3__CenterMask",
    "H3K9me3": "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K9me3__CenterMask",
}


def _default_intervals_path() -> Path:
    # Prefer package-adjacent data/ shipped in the repo
    here = Path(__file__).resolve()
    repo_data = here.parents[2] / "data" / "arch21_mane_noncoding_intervals.json"
    if repo_data.is_file():
        return repo_data
    env = Path(__file__).resolve().parents[2] / "data" / "arch21_mane_noncoding_intervals.json"
    return env


@lru_cache(maxsize=1)
def load_intervals(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _default_intervals_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing gene intervals JSON: {p}. Expected data/arch21_mane_noncoding_intervals.json"
        )
    return json.loads(p.read_text())


def map_variant_to_gene(
    chrom: str,
    pos: int,
    *,
    gene: str | None = None,
    intervals: dict | None = None,
) -> str:
    """Return ARCH21 gene for chrom:pos (1-based), or raise if outside windows."""
    chrom_n = str(chrom).replace("chr", "")
    pos = int(pos)
    iv = intervals or load_intervals()
    if gene:
        g = gene.upper()
        if g not in iv:
            raise ValueError(f"Unknown gene {g!r}; allowed: {sorted(iv)}")
        meta = iv[g]
        if str(meta["chrom"]).replace("chr", "") != chrom_n:
            raise ValueError(f"{g} is on chr{meta['chrom']}, not chr{chrom_n}")
        start, end = int(meta["window"][0]), int(meta["window"][1])
        if not (start <= pos <= end):
            raise ValueError(f"{chrom}:{pos} outside {g} MANE NC window [{start},{end}]")
        return g
    hits = []
    for g, meta in iv.items():
        if str(meta["chrom"]).replace("chr", "") != chrom_n:
            continue
        start, end = int(meta["window"][0]), int(meta["window"][1])
        if start <= pos <= end:
            hits.append(g)
    if not hits:
        raise ValueError(
            f"{chrom}:{pos} is outside all ARCH21 MANE noncoding windows"
        )
    if len(hits) > 1:
        raise ValueError(f"{chrom}:{pos} overlaps multiple genes: {hits}; pass --gene")
    return hits[0]
