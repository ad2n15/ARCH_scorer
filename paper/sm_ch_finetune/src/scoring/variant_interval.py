"""1 Mb variant-centered intervals for AlphaGenome scoring."""
from __future__ import annotations

from alphagenome.data import genome
from alphagenome.models.dna_client import SEQUENCE_LENGTH_1MB


def norm_chrom(chrom: str) -> str:
    c = str(chrom)
    return c if c.startswith("chr") else f"chr{c}"


def api_chrom(chrom: str) -> str:
    """Chromosome name for AlphaGenome API (always chr-prefixed)."""
    return norm_chrom(chrom)


def chrom_length(chrom: str, chrom_lens: dict[str, int]) -> int | None:
    """Look up contig length regardless of chr prefix in keys."""
    c = str(chrom)
    for k in (c, c.removeprefix("chr"), norm_chrom(c)):
        if k in chrom_lens:
            return chrom_lens[k]
def resolve_chrom(chrom: str, fasta_keys: set[str] | frozenset[str]) -> str:
    """Match dataset chrom name to FASTA contig (with or without chr prefix)."""
    c = str(chrom)
    for cand in (c, c.removeprefix("chr"), f"chr{c.removeprefix('chr')}"):
        if cand in fasta_keys:
            return cand
    return norm_chrom(c)


def variant_scoring_interval(
    chrom: str,
    pos: int,
    *,
    width: int = SEQUENCE_LENGTH_1MB,
    chrom_len: int | None = None,
    fasta_keys: set[str] | frozenset[str] | None = None,
) -> genome.Interval:
    """Return a fixed-width interval centered on a 1-based SNV position."""
    chrom = resolve_chrom(chrom, fasta_keys) if fasta_keys else norm_chrom(chrom)
    center = int(pos) - 1
    start = max(0, center - width // 2)
    end = start + width
    if chrom_len is not None and end > chrom_len:
        start = max(0, chrom_len - width)
        end = start + width
    interval = genome.Interval(chrom, start, end)
    if interval.width != width:
        raise ValueError(f"Interval width {interval.width} != {width} for {chrom}:{pos}")
    return interval


def make_variant(
    chrom: str, pos: int, ref: str, alt: str, *, name: str = "",
    fasta_keys: set[str] | frozenset[str] | None = None,
) -> genome.Variant:
    c = resolve_chrom(chrom, fasta_keys) if fasta_keys else norm_chrom(chrom)
    return genome.Variant(c, int(pos), ref, alt, name=name or f"{chrom}:{pos}:{ref}>{alt}")
