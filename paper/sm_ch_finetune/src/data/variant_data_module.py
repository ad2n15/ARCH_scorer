"""Variant-centered hg38 windows for classification."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np, pandas as pd
try:
    import pyfaidx
except ImportError:
    pyfaidx = None

from data.gencode_cds import GencodeCdsIndex

@dataclass
class VariantBatch:
    sequences: np.ndarray
    labels: np.ndarray
    variant_index: np.ndarray
    variant_ids: list
    cds_mask: np.ndarray | None = None
    age_mean_norm: np.ndarray | None = None
    age_valid: np.ndarray | None = None
    age_target: np.ndarray | None = None

class VariantDataModule:
    def __init__(
        self,
        parquet_path,
        fasta_path,
        window_size=32768,
        batch_size=1,
        shuffle=False,
        seed=42,
        *,
        gtf_path: str | Path | None = None,
        use_cds_mask: bool = False,
        ch_only: bool = False,
        age_target_col: str = "gnomad_age_mean_norm",
    ):
        if pyfaidx is None:
            raise ImportError("pip install pyfaidx")
        self.df = pd.read_parquet(parquet_path)
        if ch_only:
            self.df = self.df[self.df["label"] == 1].reset_index(drop=True)
        self.fasta = pyfaidx.Fasta(str(fasta_path))
        self.window_size, self.batch_size, self.shuffle, self.seed = window_size, batch_size, shuffle, seed
        self.use_cds_mask = use_cds_mask
        self._cds_index = GencodeCdsIndex.from_gtf(gtf_path) if gtf_path and use_cds_mask else None
        self._age_target_col = age_target_col
        self._has_age = age_target_col in self.df.columns

    def _age_arrays(self, rows) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if not self._has_age:
            return None, None, None
        target = np.asarray([float(getattr(r, self._age_target_col)) for r in rows], dtype=np.float32)
        valid = np.asarray([
            bool(getattr(r, "gnomad_has_age", False)) and np.isfinite(target[i])
            for i, r in enumerate(rows)
        ], dtype=np.bool_)
        target = np.where(valid, target, 0.0).astype(np.float32)
        return target, valid, target

    def __len__(self): return int(np.ceil(len(self.df) / self.batch_size))

    def _enc(self, b): return {"A":0,"C":1,"G":2,"T":3,"N":4}.get(b.upper(),4)

    def _fetch(self, chrom, pos, alt):
        ck = chrom if chrom in self.fasta else (f"chr{chrom}" if f"chr{chrom}" in self.fasta else chrom.replace("chr",""))
        if ck not in self.fasta and f"chr{ck}" in self.fasta: ck = f"chr{ck}"
        half = self.window_size // 2
        start = max(0, pos - half - 1)
        seq = str(self.fasta[ck][start:start+self.window_size]).upper().ljust(self.window_size,"N")
        arr = np.array([self._enc(b) for b in seq], dtype=np.int32)
        center = min(pos - 1 - start, self.window_size - 1)
        if 0 <= center < len(arr): arr[center] = self._enc(alt)
        return arr, center

    def _cds_mask_for_row(self, row) -> np.ndarray | None:
        if not self._cds_index:
            return None
        gene = str(row.get("gene", "."))
        if gene in (".", "", "nan"):
            return np.zeros(self.window_size, dtype=np.bool_)
        return self._cds_index.cds_mask(str(row["chrom"]), int(row["pos"]), gene, self.window_size)

    def iter_batches(self, epoch=0):
        idx = np.arange(len(self.df))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch); rng.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            bi = idx[i:i+self.batch_size]
            seqs, labels, centers, ids, masks = [], [], [], [], []
            for j in bi:
                r = self.df.iloc[j]
                s, c = self._fetch(str(r["chrom"]), int(r["pos"]), str(r["alt"]))
                seqs.append(s); labels.append(int(r["label"])); centers.append(c); ids.append(str(r["variant_id"]))
                if self.use_cds_mask:
                    masks.append(self._cds_mask_for_row(r))
            batch_mask = np.stack(masks) if masks else None
            age_mean, age_valid, age_target = self._age_arrays([self.df.iloc[j] for j in bi])
            yield VariantBatch(
                np.stack(seqs), np.array(labels, dtype=np.float32),
                np.array(centers, dtype=np.int32), ids, batch_mask,
                age_mean, age_valid, age_target,
            )
