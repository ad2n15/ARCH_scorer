#!/usr/bin/env python3
"""Download AlphaGenome JAX weights from Kaggle (requires accepted model terms)."""
from __future__ import annotations
import argparse
import sys

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="fold_1", help="e.g. fold_1 or all_folds")
    args = p.parse_args()
    try:
        import kagglehub
        path = kagglehub.model_download(f"google/alphagenome/jax/{args.version.lower()}")
    except Exception as e:
        if "403" in str(e):
            raise SystemExit(
                "Kaggle 403: accept model terms at "
                "https://www.kaggle.com/models/google/alphagenome "
                "(same Kaggle account as KAGGLE_API_TOKEN in .env)"
            ) from e
        raise
    print(path)
    print("\nOptional: set in config/pilot.yaml:")
    print(f'  checkpoint_path: "{path}"')

if __name__ == "__main__":
    main()
