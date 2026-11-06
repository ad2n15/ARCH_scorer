"""NTv3-CH: heads-only binary classifier on variant-site embeddings."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from model.ntv3_tokenizer import load_char_tokenizer

_LOAD_KW = {"trust_remote_code": True, "local_files_only": True}


def load_ntv3_core(model_path: str | Path):
    """Load post-trained NTv3 from a local snapshot dir (core only, no task heads).

    Matches InstaDeep tutorial 03: strip task heads, keep discrete-conditioned core.
    Must load from ``ntv3.core.state_dict()`` (or a wrapper with attribute ``core``).
    Loading ``ntv3.state_dict()`` into the bare core silently fails (``core.*`` key prefix).
    """
    model_path = str(model_path)
    config = AutoConfig.from_pretrained(model_path, **_LOAD_KW)
    ntv3 = AutoModel.from_pretrained(model_path, config=config, **_LOAD_KW)
    discrete_cls = type(ntv3.core).__bases__[0]
    core = discrete_cls(config)
    # Strip BigWig/BED heads (unexpected keys) — keep trunk weights.
    res = core.load_state_dict(ntv3.core.state_dict(), strict=False)
    if res.missing_keys:
        raise RuntimeError(
            f"NTv3 core weight load incomplete: {len(res.missing_keys)} missing keys "
            f"(sample: {res.missing_keys[:5]}). Pretrained weights were not applied."
        )
    return core, config


def species_token_ids(config, species: str = "human") -> torch.LongTensor:
    if species in getattr(config, "species_to_token_id", {}):
        return torch.LongTensor([config.species_to_token_id[species]])
    return torch.LongTensor([2])  # mask token fallback


def pool_variant_embedding(
    emb: torch.Tensor,
    variant_index: torch.Tensor,
    *,
    pool_mode: str = "point",
    bin_size: int = 32,
    pool_radius_bp: int = 16,
    pool_op: str = "mean",
) -> torch.Tensor:
    """Pool NTv3 hidden states at the variant site.

    pool_mode:
      - point: single token at variant_index (default / legacy)
      - borzoi_bin: mean/max over the 32 bp bin containing the variant (Borzoi parity)
      - local_bp: mean/max over [center - pool_radius_bp, center + pool_radius_bp]
    """
    if pool_mode == "point":
        b = torch.arange(emb.shape[0], device=emb.device)
        idx = variant_index.clamp(0, emb.shape[1] - 1)
        return emb[b, idx]

    pooled: list[torch.Tensor] = []
    for i in range(emb.shape[0]):
        c = int(variant_index[i].item())
        c = max(0, min(c, emb.shape[1] - 1))
        if pool_mode == "borzoi_bin":
            lo = (c // bin_size) * bin_size
            hi = min(lo + bin_size, emb.shape[1])
        elif pool_mode == "local_bp":
            lo = max(0, c - pool_radius_bp)
            hi = min(emb.shape[1], c + pool_radius_bp + 1)
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")
        chunk = emb[i, lo:hi]
        if chunk.numel() == 0:
            chunk = emb[i, c : c + 1]
        if pool_op == "max":
            pooled.append(chunk.max(dim=0).values)
        else:
            pooled.append(chunk.mean(dim=0))
    return torch.stack(pooled)


class VariantClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class NTv3CHModel(nn.Module):
    """Frozen NTv3 core + trainable variant-site classifier head."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        species: str = "human",
        hidden_dim: int = 256,
        freeze_core: bool = True,
        pool_mode: str = "point",
        bin_size: int = 32,
        pool_radius_bp: int = 16,
        pool_op: str = "mean",
    ):
        super().__init__()
        self.core, self.config = load_ntv3_core(model_path)
        self.species_ids = species_token_ids(self.config, species)
        self.pool_mode = pool_mode
        self.bin_size = int(bin_size)
        self.pool_radius_bp = int(pool_radius_bp)
        self.pool_op = pool_op
        self.head = VariantClassifierHead(self.config.embed_dim, hidden_dim=hidden_dim)
        if freeze_core:
            for p in self.core.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, variant_index: torch.Tensor) -> torch.Tensor:
        species = self.species_ids.to(input_ids.device)
        species = torch.repeat_interleave(species, input_ids.shape[0])
        outputs = self.core(input_ids, [species], output_hidden_states=True)
        emb = outputs["hidden_states"][-1]  # [B, L, D]
        x = pool_variant_embedding(
            emb,
            variant_index,
            pool_mode=self.pool_mode,
            bin_size=self.bin_size,
            pool_radius_bp=self.pool_radius_bp,
            pool_op=self.pool_op,
        )
        return self.head(x)

    def trainable_parameters(self):
        return self.head.parameters()


def build_ntv3_ch_model(model_path: str | Path, model_cfg: dict) -> NTv3CHModel:
    return NTv3CHModel(
        model_path,
        species=model_cfg.get("species", "human"),
        hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
        freeze_core=bool(model_cfg.get("freeze_core", True)),
        pool_mode=model_cfg.get("pool_mode", "point"),
        bin_size=int(model_cfg.get("bin_size", 32)),
        pool_radius_bp=int(model_cfg.get("pool_radius_bp", 16)),
        pool_op=model_cfg.get("pool_op", "mean"),
    )


def load_tokenizer(model_path: str | Path):
    return load_char_tokenizer(model_path)
