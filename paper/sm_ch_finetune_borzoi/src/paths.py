"""Path helpers for Borzoi-CH (data under ../sm_ch_finetune/)."""
from __future__ import annotations

import json
import os
from pathlib import Path

BORZOI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = BORZOI_ROOT.parent
HF_CACHE = WORKSPACE / ".cache/huggingface"
MODEL_MANIFEST = BORZOI_ROOT / "config/borzoi_model_local.json"
DEFAULT_CKPT = "human_rep0.ckpt"
DEFAULT_REPO = "Genentech/borzoi-model"
BIN_SIZE = 32


def resolve_config(config_arg: str) -> Path:
    p = Path(config_arg)
    for c in (p, BORZOI_ROOT / p, Path.cwd() / p):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Config not found: {config_arg}")


def project_subpath(rel: str | Path) -> Path:
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    return (BORZOI_ROOT / rel).resolve()


def resolve_pilot_dir(cfg: dict) -> Path:
    return project_subpath(cfg["data"]["pilot_dir"])


def resolve_split_dir(cfg: dict) -> Path:
    rel = cfg.get("data", {}).get("split_dir") or cfg.get("data", {}).get("age_filtered_dir")
    return project_subpath(rel)


def setup_hf_offline() -> None:
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def configure_torch_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    if os.environ.get("BORZOI_DISABLE_CUDNN", "1") == "0":
        return
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def resolve_ckpt_path(cfg: dict) -> Path:
    model = cfg.get("model", {})
    for key in ("local_path", "checkpoint_path"):
        rel = model.get(key)
        if rel:
            p = project_subpath(rel) if not Path(rel).is_absolute() else Path(rel)
            if p.is_file():
                return p.resolve()
    if MODEL_MANIFEST.is_file():
        data = json.loads(MODEL_MANIFEST.read_text())
        p = Path(data["local_path"])
        if p.is_file():
            return p.resolve()
    slug = "models--Genentech--borzoi-model"
    snap_root = HF_CACHE / slug / "snapshots"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            ckpt = snap / DEFAULT_CKPT
            if ckpt.is_file():
                return ckpt.resolve()
    raise FileNotFoundError(
        "Borzoi checkpoint not found. On login node run:\n"
        "  ./run_binemo_l40.sh sm_ch_finetune_borzoi/scripts/download_model.py"
    )


def write_model_manifest(local_path: Path, *, repo_id: str = DEFAULT_REPO) -> None:
    MODEL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MODEL_MANIFEST.write_text(
        json.dumps({"repo_id": repo_id, "local_path": str(local_path.resolve())}, indent=2) + "\n"
    )


def base_to_bin_index(center_base: int, *, bin_size: int = BIN_SIZE) -> int:
    return int(center_base // bin_size)
