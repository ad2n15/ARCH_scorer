#!/usr/bin/env python3
"""Download NTv3 weights + base-model code for offline compute nodes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from paths import DEFAULT_HF_MODEL, HF_CACHE, write_model_manifest

BASE_MODEL = "InstaDeepAI/ntv3_base_model"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_repo(repo_id: str, cache: Path) -> Path:
    print(f"Downloading {repo_id} ...")
    path = snapshot_download(repo_id=repo_id, cache_dir=str(cache), token=_token())
    print(f"  -> {path}")
    return Path(path)


def warm_transformers_cache(local_snapshot: Path) -> None:
    """Load model/tokenizer once so dynamic modules land in HF modules cache."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))
    from transformers import AutoConfig, AutoModel

    from model.ntv3_tokenizer import load_char_tokenizer

    print("Warming transformers module cache (downloads modeling code once) ...")
    cfg = AutoConfig.from_pretrained(str(local_snapshot), trust_remote_code=True)
    _ = AutoModel.from_pretrained(str(local_snapshot), trust_remote_code=True, config=cfg)
    _ = load_char_tokenizer(local_snapshot)
    print("Warm cache OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_HF_MODEL)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--skip-warm", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    cache = Path(args.cache_dir) if args.cache_dir else root / ".cache/huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))

    if not _token():
        print(
            "Note: set HF_TOKEN in .env for gated repos.\n"
            "Accept: https://huggingface.co/InstaDeepAI/NTv3_650M_post"
        )

    download_repo(args.base_model, cache)
    local = download_repo(args.model, cache)
    write_model_manifest(local, repo_id=args.model)

    if not args.skip_warm:
        warm_transformers_cache(local)

    print(f"\nReady for offline use:\n  weights: {local}")
    print(f"  manifest: {PROJECT / 'config/ntv3_model_local.json'}")


if __name__ == "__main__":
    main()
