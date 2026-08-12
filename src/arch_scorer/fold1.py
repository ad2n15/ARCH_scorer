"""Resolve fold_1 checkpoint and reference FASTA for local emb extract."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_fold1_checkpoint(
    *,
    explicit: str | Path | None = None,
    model_version: str = "fold_1",
) -> Path:
    """Return directory containing fold_1 JAX weights."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"ARCH_FOLD1_CKPT not a directory: {p}")
        return p

    env = os.environ.get("ARCH_FOLD1_CKPT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p

    txt = repo_root() / "artifacts" / "fold1_checkpoint_path.txt"
    if txt.is_file():
        p = Path(txt.read_text().strip()).expanduser().resolve()
        if p.is_dir():
            return p

    # Common kagglehub cache locations
    ver = model_version.lower()
    candidates = [
        repo_root() / ".cache/kagglehub/models/google/alphagenome/jax" / ver,
        Path.home() / ".cache/kagglehub/models/google/alphagenome/jax" / ver,
        Path.cwd() / ".cache/kagglehub/models/google/alphagenome/jax" / ver,
    ]
    # also alphagenome sibling workspace
    here = Path(__file__).resolve()
    if len(here.parents) >= 4:
        candidates.append(
            here.parents[3] / ".cache/kagglehub/models/google/alphagenome/jax" / ver
        )
    for cache in candidates:
        if cache.is_dir():
            versions = sorted(p for p in cache.iterdir() if p.is_dir())
            if versions:
                return versions[-1].resolve()

    raise FileNotFoundError(
        "Missing fold_1 weights. Run:\n"
        "  python scripts/install_fold1.py\n"
        "then: export ARCH_FOLD1_CKPT=$(cat artifacts/fold1_checkpoint_path.txt)"
    )


def find_fasta(explicit: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"FASTA not found: {p}")
        return p
    env = os.environ.get("ARCH_FASTA")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    for cand in (
        repo_root() / "reference" / "GRCh38.p13.genome.fa",
        repo_root() / "data" / "reference" / "hg38.fa",
    ):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "Missing reference FASTA. Run:\n"
        "  bash scripts/install_reference.sh ag\n"
        "or igenomes:\n"
        "  bash scripts/install_reference.sh igenomes\n"
        "then: export ARCH_FASTA=/path/to/genome.fa"
    )
