"""Load local all_folds AlphaGenome model for GPU variant scoring."""
from __future__ import annotations

from pathlib import Path

import jax
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from paths import find_kaggle_checkpoint


def load_local_model(
    *,
    workspace: Path,
    checkpoint_path: Path | None,
    model_version: str,
    fasta_path: Path,
    gtf_feather: Path,
    splice_starts: Path | None = None,
    splice_ends: Path | None = None,
    organism: str = "HOMO_SAPIENS",
    device: jax.Device | None = None,
) -> research_dna_model.AlphaGenomeModel:
    ckpt = checkpoint_path
    if ckpt is None or not ckpt.exists():
        ckpt = find_kaggle_checkpoint(workspace, model_version)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint for {model_version}. On login node run:\n"
            f"  ./run_binemo_l40.sh sm_ch_finetune/scripts/download_checkpoint.py --version {model_version}"
        )
    if not gtf_feather.exists():
        raise FileNotFoundError(f"Missing GTF feather: {gtf_feather}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"Missing FASTA: {fasta_path}")

    kwargs = {
        "fasta_path": str(fasta_path),
        "gtf_feather_path": str(gtf_feather),
    }
    if splice_starts and splice_starts.exists():
        kwargs["splice_site_starts_feather_path"] = str(splice_starts)
    if splice_ends and splice_ends.exists():
        kwargs["splice_site_ends_feather_path"] = str(splice_ends)

    org = ag_dna_model.Organism[organism]
    if device is None:
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpus:
            raise RuntimeError(
                "No GPU found. Run on a GPU node:\n"
                "  sbatch run_binemo_l40.sh sm_ch_finetune/scripts/05_score_variant_tracks_local.py --smoke-test"
            )
        device = gpus[0]

    print(f"Loading local checkpoint: {ckpt} on {device}")
    return research_dna_model.create(
        str(ckpt),
        organism_settings={org: research_dna_model.OrganismSettings(**kwargs)},
        device=device,
    )
