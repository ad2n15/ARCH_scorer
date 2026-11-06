#!/usr/bin/env python3
"""Download Genentech/borzoi-model human_rep0.ckpt for offline compute nodes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from paths import DEFAULT_CKPT, DEFAULT_REPO, HF_CACHE, write_model_manifest


def main() -> None:
    cache = HF_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = hf_hub_download(
        repo_id=DEFAULT_REPO,
        filename=DEFAULT_CKPT,
        cache_dir=str(cache),
        token=token,
    )
    local = Path(path)
    write_model_manifest(local)
    print(f"Downloaded -> {local}")
    print(f"Manifest   -> {PROJECT / 'config/borzoi_model_local.json'}")


if __name__ == "__main__":
    main()
