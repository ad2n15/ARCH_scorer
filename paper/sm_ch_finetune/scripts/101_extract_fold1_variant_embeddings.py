#!/usr/bin/env python3
"""Extract AlphaGenome fold_1 1536-d variant-site embeddings (ALT and REF).

Uses frozen fold_1 JAX weights (local Kaggle cache). For each allele builds a
variant-centered window, substitutes ALT (or REF) at the variant base, and
reads the 1 bp decoder embedding at that position → [1536].

Default cohort: pooled-SM MANE noncoding by gene. Pass several genes with
``--genes``; the model is loaded once and each gene is written under
``results/fold1_embeddings_pooled_sm/<GENE>/``.

Requires GPU (L4/L40/A100). Examples:
  sbatch run_binemo_l4.sh sm_ch_finetune/scripts/101_extract_fold1_variant_embeddings.py \\
      --genes TP53 --window-size 65536 --batch-size 2

  sbatch run_binemo_l4.sh sm_ch_finetune/scripts/101_extract_fold1_variant_embeddings.py \\
      --genes ASXL1,ASXL2,CBL,DNMT3A,ETNK1,PPM1D,RUNX1,TET2 \\
      --window-size 65536 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pandas as pd
import pyfaidx

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model
from alphagenome_research.model import model as model_lib
from alphagenome_research.model.metadata import metadata as metadata_lib

from heads.variant_classifier import pool_sequence_embedding
from paths import find_kaggle_checkpoint, project_subpath
from train.variant_trainer import tokens_to_onehot

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def _enc(b: str) -> int:
    return BASE.get(b.upper(), 4)


def fetch_window(
    fasta: pyfaidx.Fasta,
    chrom: str,
    pos: int,
    allele: str,
    *,
    window_size: int,
) -> tuple[np.ndarray, int]:
    """Return int tokens [L] with `allele` at the variant center index."""
    ck = chrom if chrom in fasta else (
        f"chr{chrom}" if f"chr{chrom}" in fasta else chrom.replace("chr", "")
    )
    if ck not in fasta and f"chr{ck}" in fasta:
        ck = f"chr{ck}"
    half = window_size // 2
    start = max(0, int(pos) - half - 1)
    seq = str(fasta[ck][start : start + window_size]).upper().ljust(window_size, "N")
    arr = np.array([_enc(b) for b in seq], dtype=np.int32)
    center = min(int(pos) - 1 - start, window_size - 1)
    if 0 <= center < len(arr):
        arr[center] = _enc(allele)
    return arr, int(center)


def build_variant_embedding_fn():
    metadata = {org: metadata_lib.load(org) for org in ag_dna_model.Organism}
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    hk.mixed_precision.set_policy(model_lib.AlphaGenome, policy)

    @hk.transform_with_state
    def forward(dna_sequence, organism_index, variant_index):
        alphagenome = model_lib.AlphaGenome(metadata)
        _, embeddings = alphagenome(dna_sequence, organism_index)
        x = embeddings.get_sequence_embeddings(resolution=1)
        return pool_sequence_embedding(
            x, cds_mask=None, variant_index=variant_index, pooling="variant"
        )

    return forward


class VariantSiteEmbedder:
    def __init__(self, params, state):
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

    def __call__(
        self,
        sequences_oh: jnp.ndarray,
        organism_index: jnp.ndarray,
        variant_index: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._run(sequences_oh, organism_index, variant_index)


def load_fold1(fasta: Path, *, model_version: str = "fold_1"):
    workspace = PROJECT.parent
    ckpt = find_kaggle_checkpoint(workspace, model_version)
    if ckpt is None:
        raise SystemExit(
            f"Missing {model_version} weights. Download on a login node:\n"
            f"  ./run_binemo_l40.sh sm_ch_finetune/scripts/download_checkpoint.py "
            f"--version {model_version}"
        )
    organism = ag_dna_model.Organism.HOMO_SAPIENS
    organism_settings = {
        organism: research_dna_model.OrganismSettings(fasta_path=str(fasta)),
    }
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    device = gpus[0] if gpus else jax.devices("cpu")[0]
    print(f"Loading fold_1 from {ckpt} on {device}")
    model = research_dna_model.create(
        ckpt,
        organism_settings=organism_settings,
        device=device,
    )
    org_idx = int(research_dna_model.convert_to_organism_index(organism))
    return model, org_idx, device


def _parse_genes(gene: str | None, genes: str | None) -> list[str]:
    raw = genes if genes else (gene or "TP53")
    out = [g.strip().upper() for g in raw.replace(" ", ",").split(",") if g.strip()]
    if not out:
        raise SystemExit("Provide --gene or --genes")
    seen: set[str] = set()
    uniq: list[str] = []
    for g in out:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def extract_gene(
    *,
    gene: str,
    cohort_path: Path,
    out_dir: Path,
    fasta: pyfaidx.Fasta,
    model,
    org_idx: int,
    embedder: VariantSiteEmbedder,
    modes: set[str],
    window_size: int,
    batch_size: int,
    model_version: str,
    resume: bool,
    max_variants: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta_table = out_dir / "variants_meta.parquet"
    out_alt = out_dir / "emb_alt.npy"
    out_ref = out_dir / "emb_ref.npy"
    out_diff = out_dir / "emb_diff.npy"
    out_json = out_dir / "extract_meta.json"

    df = pd.read_csv(cohort_path)
    df["variant_id"] = df["variant_id"].astype(str)
    df = df.drop_duplicates("variant_id").reset_index(drop=True)
    if max_variants:
        df = df.head(int(max_variants)).copy()
    y = (df["label"].astype(str).str.lower() == "somatic").astype(int)
    print(
        f"\n=== {gene}: n={len(df)} germline={(y == 0).sum()} "
        f"somatic={(y == 1).sum()} window={window_size}"
    )

    start_i = 0
    prev_alt: np.ndarray | None = None
    prev_ref: np.ndarray | None = None
    if resume and out_meta_table.exists():
        prev = pd.read_parquet(out_meta_table)
        n_prev = len(prev)
        if n_prev and list(prev["variant_id"].astype(str)) == list(
            df["variant_id"].astype(str).iloc[:n_prev]
        ):
            start_i = n_prev
            if "alt" in modes and out_alt.exists():
                prev_alt = np.load(out_alt)
            if "ref" in modes and out_ref.exists():
                prev_ref = np.load(out_ref)
            print(f"Resume from index {start_i}/{len(df)}")
        else:
            print("Resume skipped (variant order mismatch); re-extracting")

    todo = df.iloc[start_i:].reset_index(drop=True)
    if todo.empty:
        print(f"Nothing to do — {out_dir}")
        return

    bs = max(1, int(batch_size))
    ws = int(window_size)
    new_alt: list[np.ndarray] = []
    new_ref: list[np.ndarray] = []

    def _cat(prev: np.ndarray | None, parts: list[np.ndarray]) -> np.ndarray | None:
        if not parts and prev is None:
            return None
        chunks = ([] if prev is None else [prev]) + parts
        return np.concatenate(chunks, axis=0)

    def flush(n_complete: int):
        meta = df.iloc[:n_complete][
            [
                c
                for c in ["variant_id", "chrom", "pos", "ref", "alt", "label", "source"]
                if c in df.columns
            ]
        ].copy()
        meta["gene"] = gene
        meta["y"] = (meta["label"].astype(str).str.lower() == "somatic").astype(int)
        meta.to_parquet(out_meta_table, index=False)
        alt_arr = _cat(prev_alt, new_alt)
        ref_arr = _cat(prev_ref, new_ref)
        if alt_arr is not None:
            np.save(out_alt, alt_arr)
        if ref_arr is not None:
            np.save(out_ref, ref_arr)
        if alt_arr is not None and ref_arr is not None:
            np.save(out_diff, alt_arr - ref_arr)
        info = {
            "gene": gene,
            "cohort": str(cohort_path),
            "n": n_complete,
            "window_size": ws,
            "model_version": model_version,
            "modes": sorted(modes),
            "embedding_dim": 1536,
            "pooling": "variant_site_1bp",
            "files": {
                "meta": out_meta_table.name,
                "emb_alt": out_alt.name if alt_arr is not None else None,
                "emb_ref": out_ref.name if ref_arr is not None else None,
                "emb_diff": out_diff.name
                if (alt_arr is not None and ref_arr is not None)
                else None,
            },
        }
        out_json.write_text(json.dumps(info, indent=2))

    with model._device_context:
        for i0 in range(0, len(todo), bs):
            chunk = todo.iloc[i0 : i0 + bs]
            mats: dict[str, np.ndarray] = {}
            for mode in sorted(modes):
                tokens, centers = [], []
                for _, r in chunk.iterrows():
                    allele = str(r["alt"] if mode == "alt" else r["ref"])
                    tok, center = fetch_window(
                        fasta,
                        str(r["chrom"]),
                        int(r["pos"]),
                        allele,
                        window_size=ws,
                    )
                    tokens.append(tok)
                    centers.append(center)
                seq_oh = jnp.asarray(tokens_to_onehot(np.stack(tokens)))
                org = jnp.full((len(chunk),), org_idx, dtype=jnp.int32)
                vidx = jnp.asarray(centers, dtype=jnp.int32)
                mats[mode] = np.asarray(
                    embedder(seq_oh, org, vidx), dtype=np.float32
                )
            if "alt" in mats:
                new_alt.append(mats["alt"])
            if "ref" in mats:
                new_ref.append(mats["ref"])

            n_complete = start_i + min(i0 + bs, len(todo))
            print(
                f"  [{gene}] extracted {min(i0 + bs, len(todo))}/{len(todo)} "
                f"(total {n_complete})"
            )
            if (i0 // bs) % 5 == 0:
                flush(n_complete)

    flush(len(df))
    print(f"Wrote embeddings under {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene", default=None, help="Single gene (legacy)")
    ap.add_argument(
        "--genes",
        default=None,
        help="Comma-separated genes, e.g. ASXL1,TET2,TP53",
    )
    ap.add_argument(
        "--cohort",
        default=None,
        help="Only with a single gene: CSV path. Multi-gene uses by_gene/<GENE>.csv",
    )
    ap.add_argument(
        "--cohort-dir",
        default="data/arch_mane_noncoding_pooled_sm/by_gene",
        help="Directory of <GENE>.csv when using --genes",
    )
    ap.add_argument("--fasta", default="data/reference/hg38.fa")
    ap.add_argument("--window-size", type=int, default=65536)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--model-version", default="fold_1")
    ap.add_argument(
        "--modes",
        default="alt,ref",
        help="Comma list: alt and/or ref (diff computed if both).",
    )
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument(
        "--out-root",
        default="results/fold1_embeddings_pooled_sm",
        help="Per-gene outs under <out-root>/<GENE>/",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Only with a single gene: override output directory",
    )
    args = ap.parse_args()

    gene_list = _parse_genes(args.gene, args.genes)
    if args.cohort is not None and len(gene_list) > 1:
        raise SystemExit("--cohort only allowed with a single gene")
    if args.out_dir is not None and len(gene_list) > 1:
        raise SystemExit("--out-dir only allowed with a single gene; use --out-root")

    modes = {m.strip().lower() for m in args.modes.split(",") if m.strip()}
    if not modes & {"alt", "ref"}:
        raise SystemExit("--modes must include alt and/or ref")

    fasta_path = project_subpath(PROJECT, args.fasta)
    fasta = pyfaidx.Fasta(str(fasta_path))
    model, org_idx, _device = load_fold1(fasta_path, model_version=args.model_version)
    embedder = VariantSiteEmbedder(model._params, model._state)

    print(f"Genes ({len(gene_list)}): {', '.join(gene_list)}")
    for gene in gene_list:
        if args.cohort and len(gene_list) == 1:
            cohort_path = project_subpath(PROJECT, args.cohort)
        else:
            cohort_path = project_subpath(PROJECT, args.cohort_dir) / f"{gene}.csv"
        if not cohort_path.is_file():
            print(f"SKIP {gene}: missing cohort {cohort_path}")
            continue
        if args.out_dir and len(gene_list) == 1:
            out_dir = project_subpath(PROJECT, args.out_dir)
        else:
            out_dir = project_subpath(PROJECT, args.out_root) / gene
        extract_gene(
            gene=gene,
            cohort_path=cohort_path,
            out_dir=out_dir,
            fasta=fasta,
            model=model,
            org_idx=org_idx,
            embedder=embedder,
            modes=modes,
            window_size=int(args.window_size),
            batch_size=int(args.batch_size),
            model_version=args.model_version,
            resume=bool(args.resume),
            max_variants=args.max_variants,
        )
    print("\nDone all genes.")


if __name__ == "__main__":
    main()
