"""GENCODE GTF helpers: CDS intervals, window coverage, per-window masks."""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')


def norm_chrom(c: str) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


def parse_gtf_attrs(s: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(s)}


@dataclass
class GencodeCdsIndex:
    """CDS intervals keyed by (chrom, gene_name). Values are list of (start, end) 1-based inclusive."""

    intervals: dict[tuple[str, str], list[tuple[int, int]]] = field(default_factory=dict)
    gene_spans: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def from_gtf(cls, gtf_path: str | Path, *, cache_path: str | Path | None = None) -> GencodeCdsIndex:
        gtf_path = Path(gtf_path)
        cache_path = Path(cache_path) if cache_path else gtf_path.with_suffix(".cds_index.pkl")
        if cache_path.exists() and cache_path.stat().st_mtime >= gtf_path.stat().st_mtime:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        intervals: dict[tuple[str, str], list[tuple[int, int]]] = {}
        gene_spans: dict[tuple[str, str], tuple[int, int]] = {}
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                chrom, _src, feature, start, end = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
                attrs = parse_gtf_attrs(parts[8])
                gene_name = attrs.get("gene_name")
                if not gene_name:
                    continue
                key = (chrom, gene_name)
                if feature == "CDS":
                    intervals.setdefault(key, []).append((start, end))
                elif feature == "gene":
                    prev = gene_spans.get(key)
                    if prev is None:
                        gene_spans[key] = (start, end)
                    else:
                        gene_spans[key] = (min(prev[0], start), max(prev[1], end))

        idx = cls(intervals=intervals, gene_spans=gene_spans)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        return idx

    def cds_span(self, chrom: str, gene_name: str) -> tuple[int, int] | None:
        ivs = self.intervals.get((norm_chrom(chrom), gene_name))
        if not ivs:
            return None
        return min(s for s, _ in ivs), max(e for _, e in ivs)

    def window_bounds(self, pos: int, window_size: int) -> tuple[int, int, int]:
        """Return (win_start_0based, win_start_1based, win_end_1based inclusive)."""
        half = window_size // 2
        win_start = max(0, pos - half - 1)
        win_start_1 = win_start + 1
        win_end_1 = win_start + window_size
        return win_start, win_start_1, win_end_1

    def cds_fits_in_window(self, chrom: str, pos: int, gene_name: str, window_size: int) -> bool:
        span = self.cds_span(chrom, gene_name)
        if span is None:
            return False
        cds_min, cds_max = span
        _, win_lo, win_hi = self.window_bounds(pos, window_size)
        return cds_min >= win_lo and cds_max <= win_hi

    def gene_fits_in_window(self, chrom: str, pos: int, gene_name: str, window_size: int) -> bool:
        span = self.gene_spans.get((norm_chrom(chrom), gene_name))
        if span is None:
            return False
        g_min, g_max = span
        _, win_lo, win_hi = self.window_bounds(pos, window_size)
        return g_min >= win_lo and g_max <= win_hi

    def cds_mask(self, chrom: str, pos: int, gene_name: str, window_size: int) -> np.ndarray:
        """Boolean mask [window_size] for positions overlapping gene CDS within the variant window."""
        win_start, win_lo, win_hi = self.window_bounds(pos, window_size)
        mask = np.zeros(window_size, dtype=np.bool_)
        key = (norm_chrom(chrom), gene_name)
        for cds_start, cds_end in self.intervals.get(key, []):
            overlap_lo = max(cds_start, win_lo)
            overlap_hi = min(cds_end, win_hi)
            if overlap_lo > overlap_hi:
                continue
            i0 = overlap_lo - win_lo
            i1 = overlap_hi - win_lo
            mask[i0 : i1 + 1] = True
        return mask
