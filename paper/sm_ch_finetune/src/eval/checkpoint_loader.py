"""Load fine-tuned checkpoints with correct encoder-only forward path."""
from __future__ import annotations

import json
from pathlib import Path

from alphagenome_ft import load_checkpoint

from heads.registry import resolve_strategy


def patch_encoder_checkpoint_config(checkpoint_dir: Path) -> bool:
    """Fix config.json when alphagenome-ft omitted use_encoder_output=True."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return False
    with open(config_path) as f:
        data = json.load(f)
    if data.get("use_encoder_output"):
        return False
    data["use_encoder_output"] = True
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
    return True


def load_variant_checkpoint(
    checkpoint_dir: Path | str,
    cfg: dict,
    *,
    base_model_version: str,
    organism_settings,
    device,
    base_checkpoint_path,
    init_seq_len: int,
):
    ckpt_dir = Path(checkpoint_dir)
    if resolve_strategy(cfg) == "encoder_only":
        if patch_encoder_checkpoint_config(ckpt_dir):
            print("  Patched checkpoint config: use_encoder_output=True")
    return load_checkpoint(
        ckpt_dir,
        base_model_version=base_model_version,
        organism_settings=organism_settings,
        device=device,
        base_checkpoint_path=base_checkpoint_path,
        init_seq_len=init_seq_len,
    )
