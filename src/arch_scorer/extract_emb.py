"""Extract fold_1 ALT−REF DIFF embedding (1536-d) for one SNV.

Requires GPU + AlphaGenome research stack + fold_1 weights (same as
sm_ch_finetune/scripts/101_extract_fold1_variant_embeddings.py).

Set ALPHAGENOME_ROOT to the parent of sm_ch_finetune, or keep ARCH_scorer
as a sibling of sm_ch_finetune under that root.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _find_sm() -> Path:
    env = os.environ.get("ALPHAGENOME_ROOT")
    cands = []
    if env:
        cands.append(Path(env) / "sm_ch_finetune")
    here = Path(__file__).resolve()
    # ARCH_scorer/src/arch_scorer → parents[3] = alphagenome
    cands += [
        here.parents[3] / "sm_ch_finetune",
        Path.cwd() / "sm_ch_finetune",
        Path.cwd().parent / "sm_ch_finetune",
    ]
    for sm in cands:
        if (sm / "scripts" / "101_extract_fold1_variant_embeddings.py").is_file():
            return sm
    raise FileNotFoundError(
        "Cannot find sm_ch_finetune. Set ALPHAGENOME_ROOT or pass --emb with a "
        "precomputed DIFF vector."
    )


def _load_101(sm: Path):
    sys.path.insert(0, str(sm / "src"))
    spec = importlib.util.spec_from_file_location(
        "ext101", sm / "scripts" / "101_extract_fold1_variant_embeddings.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def extract_fold1_diff(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    *,
    out_path: str | Path | None = None,
    window_size: int = 65536,
    fasta: str | Path | None = None,
) -> np.ndarray:
    """Return emb_alt − emb_ref (shape 1536,)."""
    import jax
    import jax.numpy as jnp
    import pyfaidx
    from train.variant_trainer import tokens_to_onehot

    sm = _find_sm()
    mod = _load_101(sm)
    fasta_path = Path(
        fasta
        or os.environ.get("ARCH_FASTA")
        or (sm / "data" / "reference" / "hg38.fa")
    )
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Missing FASTA {fasta_path} (set ARCH_FASTA)")

    fa = pyfaidx.Fasta(str(fasta_path))
    model, org_idx, _device = mod.load_fold1(fasta_path)
    embedder = mod.VariantSiteEmbedder(model.params, model.state)

    tok_ref, center = mod.fetch_window(
        fa, chrom, int(pos), str(ref).upper(), window_size=window_size
    )
    tok_alt, _ = mod.fetch_window(
        fa, chrom, int(pos), str(alt).upper(), window_size=window_size
    )
    oh_ref = tokens_to_onehot(tok_ref[None, ...])
    oh_alt = tokens_to_onehot(tok_alt[None, ...])
    org = jnp.asarray([org_idx], dtype=jnp.int32)
    vi = jnp.asarray([center], dtype=jnp.int32)
    e_ref = np.asarray(embedder(jnp.asarray(oh_ref), org, vi)[0], dtype=np.float32)
    e_alt = np.asarray(embedder(jnp.asarray(oh_alt), org, vi)[0], dtype=np.float32)
    diff = e_alt - e_ref
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, diff)
    return diff


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--out", default="emb_diff.npy")
    ap.add_argument("--window-size", type=int, default=65536)
    ap.add_argument("--fasta", default=None)
    args = ap.parse_args()
    diff = extract_fold1_diff(
        args.chrom,
        args.pos,
        args.ref,
        args.alt,
        out_path=args.out,
        window_size=args.window_size,
        fasta=args.fasta,
    )
    print(f"Wrote {args.out} shape={diff.shape} dtype={diff.dtype}")


if __name__ == "__main__":
    main()
