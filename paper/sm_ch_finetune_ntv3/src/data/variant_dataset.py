"""Variant-centered windows for NTv3-CH (reuses sm_ch_finetune parquet splits)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import pyfaidx
except ImportError as exc:
    raise ImportError("pip install pyfaidx") from exc


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    variant_index: torch.Tensor
    variant_ids: list[str]


class VariantDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        fasta_path: str | Path,
        tokenizer,
        *,
        window_size: int = 65536,
        pad_to_multiple_of: int = 128,
    ):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.fasta = pyfaidx.Fasta(str(fasta_path))
        self.tokenizer = tokenizer
        self.window_size = int(window_size)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

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
        tok = self.tokenizer(
            seq,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.window_size,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"][0]
        # Character-level tokenizer: center base index matches genomic center when no extra tokens.
        var_idx = min(center, input_ids.shape[0] - 1)
        return {
            "input_ids": input_ids,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "variant_index": torch.tensor(var_idx, dtype=torch.long),
            "variant_id": str(row["variant_id"]),
        }


def collate(batch: list[dict]) -> Batch:
    return Batch(
        input_ids=torch.stack([b["input_ids"] for b in batch]),
        labels=torch.stack([b["label"] for b in batch]),
        variant_index=torch.stack([b["variant_index"] for b in batch]),
        variant_ids=[b["variant_id"] for b in batch],
    )
