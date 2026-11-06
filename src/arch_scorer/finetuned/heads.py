"""The trained variant-site classifier heads (identical architecture across backbones apart from the input
LayerNorm, which only NTv3 uses): [LayerNorm] -> Linear(d, 256) -> ReLU -> Linear(256, 1) -> sigmoid."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class Head(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256, layer_norm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(self.norm(x)))).squeeze(-1)


def load_head(model_dir: Path) -> tuple[Head, dict]:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    head = Head(cfg["embed_dim"], cfg["hidden_dim"], cfg.get("layer_norm", False))
    head.load_state_dict(load_file(Path(model_dir) / "head.safetensors"))
    return head.eval(), cfg
