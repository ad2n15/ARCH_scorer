#!/usr/bin/env python3
"""Download AlphaGenome fold_1 JAX weights (Kaggle) for local DIFF embeddings.

Requires:
  - Accepted model terms: https://www.kaggle.com/models/google/alphagenome
  - Kaggle credentials in `.env` (KAGGLE_USERNAME + KAGGLE_KEY) or ~/.kaggle/kaggle.json

Usage:
  python scripts/install_fold1.py
  python scripts/install_fold1.py --version fold_1

Writes the resolved path to artifacts/fold1_checkpoint_path.txt and prints
export ARCH_FOLD1_CKPT=...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_TXT = ROOT / "artifacts" / "fold1_checkpoint_path.txt"
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from arch_scorer.env import load_dotenv

    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="fold_1", help="Kaggle model version (default: fold_1)")
    args = ap.parse_args()

    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.is_file():
            print(
                "Warning: no KAGGLE_USERNAME/KAGGLE_KEY in env/.env and no "
                f"{kaggle_json}. See README / .env.example.",
                flush=True,
            )

    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "Install kagglehub: pip install -e '.[emb]'  (or pip install kagglehub)"
        ) from e

    slug = f"google/alphagenome/jax/{args.version.lower()}"
    try:
        path = Path(kagglehub.model_download(slug)).resolve()
    except Exception as e:
        if "403" in str(e):
            raise SystemExit(
                "Kaggle 403: accept model terms at "
                "https://www.kaggle.com/models/google/alphagenome "
                "with the same account as your Kaggle API token."
            ) from e
        raise

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(str(path) + "\n")
    print(path)
    print()
    print("Done. Add to your environment:")
    print(f"  export ARCH_FOLD1_CKPT={path}")
    print(f"(also saved to {OUT_TXT})")


if __name__ == "__main__":
    main()
