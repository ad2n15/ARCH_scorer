"""Path helpers for NTv3-CH pipeline (data live under ../sm_ch_finetune/)."""
from __future__ import annotations

import json
import os
from pathlib import Path

NTV3_ROOT = Path(__file__).resolve().parents[1]
SM_CH_ROOT = NTV3_ROOT.parent / "sm_ch_finetune"
WORKSPACE = NTV3_ROOT.parent
HF_CACHE = WORKSPACE / ".cache/huggingface"
MODEL_MANIFEST = NTV3_ROOT / "config/ntv3_model_local.json"
DEFAULT_HF_MODEL = "InstaDeepAI/NTv3_650M_post"


def resolve_config(config_arg: str) -> Path:
    p = Path(config_arg)
    candidates = [p, NTV3_ROOT / p, Path.cwd() / p]
    if p.is_absolute():
        candidates.insert(0, p)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Config not found: {config_arg}")


def project_subpath(rel: str | Path) -> Path:
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    # Paths starting with ../sm_ch_finetune resolve from alphagenome root.
    if str(rel).startswith(".."):
        return (NTV3_ROOT / rel).resolve()
    return (NTV3_ROOT / rel).resolve()


def resolve_pilot_dir(cfg: dict) -> Path:
    rel = cfg.get("data", {}).get("pilot_dir")
    if not rel:
        raise KeyError("config data.pilot_dir required")
    return project_subpath(rel)


def resolve_split_dir(cfg: dict) -> Path:
    rel = cfg.get("data", {}).get("split_dir") or cfg.get("data", {}).get("age_filtered_dir")
    if not rel:
        raise KeyError("config data.split_dir or age_filtered_dir required")
    return project_subpath(rel)


def setup_hf_offline() -> None:
    """Force HuggingFace to use local cache only (compute nodes have no internet)."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _snapshot_from_cache(repo_id: str) -> Path | None:
    slug = "models--" + repo_id.replace("/", "--")
    snap_root = HF_CACHE / slug / "snapshots"
    if not snap_root.is_dir():
        return None
    snaps = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in snaps:
        if snap.is_dir() and (snap / "config.json").is_file():
            return snap.resolve()
    return None


def resolve_hf_model_path(cfg: dict) -> Path:
    """Resolve local snapshot dir for NTv3 (never hits the network)."""
    model = cfg.get("model", {})
    for key in ("local_path", "hf_local_path"):
        rel = model.get(key)
        if rel:
            p = project_subpath(rel) if not Path(rel).is_absolute() else Path(rel)
            if p.is_dir() and (p / "config.json").is_file():
                return p.resolve()

    if MODEL_MANIFEST.is_file():
        data = json.loads(MODEL_MANIFEST.read_text())
        p = Path(data["local_path"])
        if p.is_dir() and (p / "config.json").is_file():
            return p.resolve()

    repo_id = model.get("hf_model", DEFAULT_HF_MODEL)
    snap = _snapshot_from_cache(repo_id)
    if snap is not None:
        return snap

    raise FileNotFoundError(
        "NTv3 weights not found locally.\n"
        "On a login node (internet), run:\n"
        "  ./run_binemo_l40.sh sm_ch_finetune_ntv3/scripts/download_model.py\n"
        f"Expected cache under {HF_CACHE}"
    )


def write_model_manifest(local_path: Path, *, repo_id: str = DEFAULT_HF_MODEL) -> None:
    MODEL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MODEL_MANIFEST.write_text(
        json.dumps({"repo_id": repo_id, "local_path": str(local_path.resolve())}, indent=2) + "\n"
    )


def configure_torch_cuda() -> None:
    """Avoid cuDNN mismatch between PyTorch wheels (cu121) and cluster CUDA 12.6."""
    import torch

    if not torch.cuda.is_available():
        return
    if os.environ.get("NTV3_DISABLE_CUDNN", "1") == "0":
        return
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
