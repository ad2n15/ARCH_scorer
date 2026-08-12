#!/usr/bin/env python3
"""Download AlphaGenome fold_1 JAX weights (Kaggle) for local DIFF embeddings.

Requires:
  - Accepted model terms: https://www.kaggle.com/models/google/alphagenome
  - Kaggle credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY

Usage:
  python scripts/install_fold1.py
  python scripts/install_fold1.py --version fold_1

Writes the resolved path to artifacts/fold1_checkpoint_path.txt and prints
export ARCH_FOLD1_CKPT=...
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_TXT = ROOT / "artifacts" / "fold1_checkpoint_path.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="fold_1", help="Kaggle model version (default: fold_1)")
    args = ap.parse_args()
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
