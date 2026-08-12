"""Extract fold_1 ALT−REF DIFF embedding (1536-d) for one or many SNVs.

Same protocol as training (variant-centered 64 kb window, 1 bp site pooling,
frozen fold_1 JAX weights). Does **not** use fine-tuned checkpoints.

Requires GPU + research stack:
  pip install -e ".[emb]"
  pip install git+https://github.com/google-deepmind/alphagenome_research.git
  python scripts/install_fold1.py
  bash scripts/install_reference.sh ag   # or igenomes
  export ARCH_FOLD1_CKPT=... ARCH_FASTA=...
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np

from .fold1 import find_fasta, find_fold1_checkpoint

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def _enc(b: str) -> int:
    return BASE.get(b.upper(), 4)


def tokens_to_onehot(tokens: np.ndarray) -> np.ndarray:
    out = np.zeros((*tokens.shape, 4), dtype=np.float32)
    for i in range(4):
        out[..., i] = (tokens == i).astype(np.float32)
    return out


def fetch_window(
    fasta,
    chrom: str,
    pos: int,
    allele: str,
    *,
    window_size: int,
) -> tuple[np.ndarray, int]:
    """Return int tokens [L] with `allele` at the variant center index."""
    keys = set(fasta.keys())
    ck = chrom if chrom in keys else (
        f"chr{chrom}" if f"chr{chrom}" in keys else chrom.replace("chr", "")
    )
    if ck not in keys and f"chr{ck}" in keys:
        ck = f"chr{ck}"
    half = window_size // 2
    start = max(0, int(pos) - half - 1)
    seq = str(fasta[ck][start : start + window_size]).upper().ljust(window_size, "N")
    arr = np.array([_enc(b) for b in seq], dtype=np.int32)
    center = min(int(pos) - 1 - start, window_size - 1)
    if 0 <= center < len(arr):
        arr[center] = _enc(allele)
    return arr, int(center)


def _pool_variant(x, variant_index):
    import jax.numpy as jnp

    return x[jnp.arange(x.shape[0]), variant_index]


def build_variant_embedding_fn():
    import haiku as hk
    import jmp
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import model as model_lib
    from alphagenome_research.model.metadata import metadata as metadata_lib

    metadata = {org: metadata_lib.load(org) for org in ag_dna_model.Organism}
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)

    @hk.transform_with_state
    def forward(dna_sequence, organism_index, variant_index):
        alphagenome = model_lib.AlphaGenome(metadata)
        _, embeddings = alphagenome(dna_sequence, organism_index)
        x = embeddings.get_sequence_embeddings(resolution=1)
        return _pool_variant(x, variant_index)

    return forward


class VariantSiteEmbedder:
    def __init__(self, params, state):
        import jax
        import jax.numpy as jnp

        self._transform = build_variant_embedding_fn()
        self._params = params
        self._state = state

        @partial(jax.jit)
        def _run(sequences, organism_index, variant_index):
            pooled, _ = self._transform.apply(
                self._params,
                self._state,
                None,
                sequences,
                organism_index,
                variant_index,
            )
            return pooled

        self._run = _run
        self._jnp = jnp

    def __call__(self, sequences_oh, organism_index, variant_index):
        return self._run(sequences_oh, organism_index, variant_index)


def load_fold1(
    fasta: Path,
    *,
    checkpoint: str | Path | None = None,
    model_version: str = "fold_1",
):
    import jax
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model

    ckpt = find_fold1_checkpoint(explicit=checkpoint, model_version=model_version)
    organism = ag_dna_model.Organism.HOMO_SAPIENS
    organism_settings = {
        organism: research_dna_model.OrganismSettings(fasta_path=str(fasta)),
    }
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    device = gpus[0] if gpus else jax.devices("cpu")[0]
    print(f"Loading {model_version} from {ckpt} on {device}")
    model = research_dna_model.create(
        str(ckpt),
        organism_settings=organism_settings,
        device=device,
    )
    org_idx = int(research_dna_model.convert_to_organism_index(organism))
    return model, org_idx, device


def extract_fold1_diff(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    *,
    out_path: str | Path | None = None,
    window_size: int = 65536,
    fasta: str | Path | None = None,
    checkpoint: str | Path | None = None,
    model=None,
    org_idx: int | None = None,
    embedder: VariantSiteEmbedder | None = None,
    fa=None,
) -> np.ndarray:
    """Return emb_alt − emb_ref (shape 1536,). Reuse model/embedder for batches."""
    import jax.numpy as jnp
    import pyfaidx

    fasta_path = find_fasta(fasta)
    if fa is None:
        fa = pyfaidx.Fasta(str(fasta_path))
    if model is None or embedder is None or org_idx is None:
        model, org_idx, _device = load_fold1(fasta_path, checkpoint=checkpoint)
        embedder = VariantSiteEmbedder(model.params, model.state)

    tok_ref, center = fetch_window(
        fa, chrom, int(pos), str(ref).upper(), window_size=window_size
    )
    tok_alt, _ = fetch_window(
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


def extract_fold1_diff_batch(
    rows: Iterable[dict],
    *,
    window_size: int = 65536,
    fasta: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> list[np.ndarray]:
    """Extract DIFF vectors for dict rows with chrom/pos/ref/alt (model loaded once)."""
    import pyfaidx

    fasta_path = find_fasta(fasta)
    fa = pyfaidx.Fasta(str(fasta_path))
    model, org_idx, _ = load_fold1(fasta_path, checkpoint=checkpoint)
    embedder = VariantSiteEmbedder(model.params, model.state)
    out: list[np.ndarray] = []
    for r in rows:
        out.append(
            extract_fold1_diff(
                str(r["chrom"]),
                int(r["pos"]),
                str(r["ref"]),
                str(r["alt"]),
                window_size=window_size,
                fasta=fasta_path,
                model=model,
                org_idx=org_idx,
                embedder=embedder,
                fa=fa,
            )
        )
    return out


def main() -> None:
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chrom", help="Single-variant chromosome")
    g.add_argument("--batch", help="TSV/CSV with chrom,pos,ref,alt")
    ap.add_argument("--pos", type=int, default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--alt", default=None)
    ap.add_argument("--out", default="emb_diff.npy")
    ap.add_argument("--window-size", type=int, default=65536)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    if args.batch is None:
        if args.pos is None or args.ref is None or args.alt is None:
            ap.error("single-variant mode needs --pos --ref --alt")
        diff = extract_fold1_diff(
            args.chrom,
            args.pos,
            args.ref,
            args.alt,
            out_path=args.out,
            window_size=args.window_size,
            fasta=args.fasta,
            checkpoint=args.checkpoint,
        )
        print(f"Wrote {args.out} shape={diff.shape} dtype={diff.dtype}")
        return

    path = Path(args.batch)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    for need in ("chrom", "pos", "ref", "alt"):
        if need not in cols:
            raise SystemExit(f"Batch missing column '{need}'")
    rows = [
        {
            "chrom": row[cols["chrom"]],
            "pos": row[cols["pos"]],
            "ref": row[cols["ref"]],
            "alt": row[cols["alt"]],
        }
        for _, row in df.iterrows()
    ]
    diffs = extract_fold1_diff_batch(
        rows,
        window_size=args.window_size,
        fasta=args.fasta,
        checkpoint=args.checkpoint,
    )
    out = Path(args.out)
    if out.suffix == ".npy":
        arr = np.stack(diffs, axis=0)
        np.save(out, arr)
        print(f"Wrote {out} shape={arr.shape}")
    else:
        # one npy per row next to out stem
        out.parent.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(diffs):
            np.save(out.parent / f"{out.stem}_{i}.npy", d)
        print(f"Wrote {len(diffs)} files under {out.parent}")


if __name__ == "__main__":
    main()
