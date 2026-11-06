"""64 kb variant-centered windows for Borzoi-CH."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from grelu.sequence.format import strings_to_one_hot
from paths import base_to_bin_index

try:
    import pyfaidx
except ImportError as exc:
    raise ImportError("pip install pyfaidx") from exc


@dataclass
class Batch:
    x: torch.Tensor
    labels: torch.Tensor
    bin_index: torch.Tensor
    variant_ids: list[str]


class VariantDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        fasta_path: str | Path,
        *,
        window_size: int = 65536,
    ):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.fasta = pyfaidx.Fasta(str(fasta_path))
        self.window_size = int(window_size)

    def __len__(self) -> int:
        return len(self.df)

    def _fetch_seq(self, chrom: str, pos: int, alt: str) -> tuple[str, int]:
        ck = chrom
        if ck not in self.fasta:
            ck = f"chr{chrom}" if f"chr{chrom}" in self.fasta else chrom.replace("chr", "")
        if ck not in self.fasta and f"chr{ck}" in self.fasta:
            ck = f"chr{ck}"
        half = self.window_size // 2
        start = max(0, pos - half - 1)
        seq = str(self.fasta[ck][start : start + self.window_size]).upper()
        seq = seq.ljust(self.window_size, "N")
        center = min(pos - 1 - start, self.window_size - 1)
        if 0 <= center < len(seq):
            seq = seq[:center] + alt.upper() + seq[center + 1 :]
        return seq, center

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        seq, center = self._fetch_seq(str(row["chrom"]), int(row["pos"]), str(row["alt"]))
        one_hot = strings_to_one_hot(seq, add_batch_axis=False)  # [4, L]
        return {
            "x": one_hot,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "bin_index": torch.tensor(base_to_bin_index(center), dtype=torch.long),
            "variant_id": str(row["variant_id"]),
        }


def collate(batch: list[dict]) -> Batch:
    return Batch(
        x=torch.stack([b["x"] for b in batch]),
        labels=torch.stack([b["label"] for b in batch]),
        bin_index=torch.stack([b["bin_index"] for b in batch]),
        variant_ids=[b["variant_id"] for b in batch],
    )
