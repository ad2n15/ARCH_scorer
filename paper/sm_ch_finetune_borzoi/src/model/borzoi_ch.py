"""Borzoi-CH: frozen embedding + variant-bin MLP classifier."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariantClassifierHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def load_borzoi_embedding(ckpt_path: str | Path, *, disable_crop: bool = True) -> tuple[nn.Module, int]:
    """Load Borzoi trunk from human_rep0.ckpt.

    The published model crops both ends for native 524 kb windows. That crop
    exceeds the sequence length at 64 kb and raises ``narrow(): length must be
    non-negative``. For our AlphaGenome/NTv3-matched 64 kb protocol we disable
    the crop (set crop_len=0).
    """
    from grelu.lightning import LightningModel
    from grelu.model.layers import Crop

    pl_model = LightningModel.load_from_checkpoint(
        str(ckpt_path), map_location="cpu", weights_only=False,
    )
    embedding = pl_model.model.embedding
    in_channels = int(pl_model.model.head.in_channels)
    if disable_crop and hasattr(embedding, "crop"):
        old = getattr(embedding.crop, "layer", None)
        print(f"Borzoi crop before disable: {old}", flush=True)
        embedding.crop = Crop(crop_len=0)
        print("Borzoi crop disabled (crop_len=0) for non-native window length", flush=True)
    return embedding, in_channels


class BorzoiCHModel(nn.Module):
    """Frozen Borzoi embedding trunk + trainable variant-site bin classifier."""

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        hidden_dim: int = 256,
        freeze_embedding: bool = True,
        disable_crop: bool = True,
    ):
        super().__init__()
        self.embedding, self.in_channels = load_borzoi_embedding(
            ckpt_path, disable_crop=disable_crop,
        )
        self.head = VariantClassifierHead(self.in_channels, hidden_dim=hidden_dim)
        if freeze_embedding:
            for p in self.embedding.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, bin_index: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, L] one-hot → h: [B, C, L_bins] (bin_size=32)
        h = self.embedding(x)
        b = torch.arange(h.shape[0], device=h.device)
        idx = bin_index.clamp(0, h.shape[2] - 1)
        x_bin = h[b, :, idx]  # [B, C]
        return self.head(x_bin)

    def trainable_parameters(self):
        return self.head.parameters()
