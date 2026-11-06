"""Ellingford et al. (Genome Med 2022) noncoding / splice helpers for MANE Select.

Noncoding = outside protein-coding exons (CDS).
Canonical splice (noncoding) = intronic ±1/±2 only (GT donor / AG acceptor).
Coding-exon ±2 splice flanks are CDS and must be excluded from noncoding pools.
"""
from __future__ import annotations

import re
from pathlib import Path

MANE_TAG = "MANE_Select"


def _chrom_str(x: str) -> str:
    s = str(x).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    return s


def load_mane_cds_exons(
    gtf: Path, genes: set[str]
) -> tuple[dict[str, list[tuple[str, int, int]]], dict[str, list[tuple[str, int, int]]]]:
    """gene -> list of (chrom, start, end) for MANE Select CDS and exons (1-based)."""
    cds: dict[str, list[tuple[str, int, int]]] = {}
    exons: dict[str, list[tuple[str, int, int]]] = {}
    with open(gtf) as f:
        for line in f:
            if line.startswith("#") or MANE_TAG not in line:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] not in ("CDS", "exon"):
                continue
            m = re.search(r'gene_name "([^"]+)"', p[8])
            if not m or m.group(1) not in genes:
                continue
            g = m.group(1)
            iv = (_chrom_str(p[0]), int(p[3]), int(p[4]))
            if p[2] == "CDS":
                cds.setdefault(g, []).append(iv)
            else:
                exons.setdefault(g, []).append(iv)
    return cds, exons


def in_intervals(chrom: str, pos: int, intervals: list[tuple[str, int, int]]) -> bool:
    c = _chrom_str(chrom)
    for ic, s, e in intervals:
        if ic == c and s <= pos <= e:
            return True
    return False


def exonic_flank(
    chrom: str, pos: int, intervals: list[tuple[str, int, int]], *, bp: int = 2
) -> bool:
    """True if pos is among the first/last `bp` bases inside an interval."""
    c = _chrom_str(chrom)
    for ic, s, e in intervals:
        if ic != c or not (s <= pos <= e):
            continue
        if pos <= s + (bp - 1) or pos >= e - (bp - 1):
            return True
    return False


def intronic_canonical_splice(
    chrom: str, pos: int, exons: list[tuple[str, int, int]], *, bp: int = 2
) -> bool:
    """Ellingford canonical splice: first/last `bp` of the intron (outside exon)."""
    c = _chrom_str(chrom)
    for ic, s, e in exons:
        if ic != c:
            continue
        if e + 1 <= pos <= e + bp or s - bp <= pos <= s - 1:
            return True
    return False


def violates_ellingford_noncoding(
    chrom: str,
    pos: int,
    gene: str,
    cds: dict[str, list[tuple[str, int, int]]],
    *,
    bp: int = 2,
) -> bool:
    """True if allele is in MANE CDS (incl. coding-exon splice ±bp)."""
    intervals = cds.get(gene, [])
    if not intervals:
        return False
    return in_intervals(chrom, pos, intervals) or exonic_flank(
        chrom, pos, intervals, bp=bp
    )
