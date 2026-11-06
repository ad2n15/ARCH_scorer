#!/usr/bin/env python3
"""Head-only fine-tuning for SM vs CH classification using alphagenome-ft."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import os

from jax_cuda import configure_jax_cuda
configure_jax_cuda()

from paths import find_kaggle_checkpoint, project_subpath, resolve_config, resolve_pilot_dir, resolve_split_dir
from train.variant_trainer import train_variant_classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pilot.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    fasta = project_subpath(PROJECT, cfg["data"]["fasta_path"])
    if not fasta.exists():
        raise SystemExit(f"Missing hg38 FASTA: {fasta}\nDownload to data/reference/hg38.fa")

    try:
        from alphagenome.models import dna_model as ag_dna_model
        from alphagenome.models import dna_output
        from alphagenome_research.model import dna_model as research_dna_model
        from alphagenome_ft import create_model_with_heads
    except ImportError as e:
        raise SystemExit(
            f"Missing Python dependency: {e}\n\n"
            "Install into alphagenome/.local (same container as run):\n"
            "  ./run_binemo_l40.sh sm_ch_finetune/scripts/install_deps.sh"
        )

    from heads.registry import register_variant_head
    from data.variant_data_module import VariantDataModule

    head_id = cfg["model"]["head_id"]
    strategy = register_variant_head(cfg)
    use_encoder_output = strategy == "encoder_only"
    detach_backbone = strategy != "full"

    import jax

    import pandas as pd

    workspace = PROJECT.parent
    ckpt_path = cfg["model"].get("checkpoint_path")
    if ckpt_path:
        ckpt_path = project_subpath(PROJECT, ckpt_path)
    else:
        ckpt_path = find_kaggle_checkpoint(workspace, cfg["model"]["model_version"])

    if not ckpt_path or not ckpt_path.exists():
        msg = (
            "No local pretrained checkpoint found.\n"
            "On a login node (internet required), run once:\n"
            "  ./run_binemo_l40.sh sm_ch_finetune/scripts/download_checkpoint.py --version fold_1\n"
            "SLURM compute nodes have no internet — weights must be cached under alphagenome/.cache/"
        )
        raise SystemExit(msg)
    print(f"Using local checkpoint: {ckpt_path}")

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if gpus:
        device = gpus[0]
        print(f"Using device: {device}")
    elif args.smoke_test:
        device = jax.devices("cpu")[0]
        print(f"WARNING: no GPU in container; smoke test on CPU ({device})")
    else:
        raise SystemExit(
            "No GPU found inside container.\n"
            "  module load cuda/12.6.3   # not cuda/13.x\n"
            "  ./run_binemo_l40.sh -m pip install --user -U \"jax[cuda12]\" "
            "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n"
            "  ./run_binemo_l40.sh -c \"import jax; print(jax.devices())\"  # should show CudaDevice"
        )

    organism = ag_dna_model.Organism[cfg["model"]["organism"]]
    organism_settings = {
        organism: research_dna_model.OrganismSettings(fasta_path=str(fasta)),
    }

    try:
        model = create_model_with_heads(
            cfg["model"]["model_version"], heads=[head_id],
            detach_backbone=detach_backbone,
            use_encoder_output=use_encoder_output,
            checkpoint_path=ckpt_path,
            organism_settings=organism_settings,
            device=device,
            init_seq_len=cfg["model"]["window_size"],
        )
    except Exception as e:
        if "403" in str(e):
            raise SystemExit(
                "Kaggle 403 — accept terms at https://www.kaggle.com/models/google/alphagenome "
                "then download on a login node with download_checkpoint.py"
            ) from e
        if os.environ.get("SLURM_JOB_ID") and "kaggle" in str(e).lower():
            raise SystemExit(
                "Kaggle/network error on compute node. Use cached weights only "
                f"(checkpoint_path or .cache/kagglehub). Original: {e}"
            ) from e
        raise
    if strategy != "full":
        model.freeze_except_head(head_id)
    elif cfg["model"]["window_size"] >= 65536:
        print(
            "WARNING: full-model fine-tuning at 64 kb may OOM on L40 (48 GB). "
            "Tutorial recommends ~76 GB VRAM for full fine-tune; try 32 kb or heads_only."
        )

    proc = resolve_split_dir(PROJECT, cfg)
    pilot_path = resolve_pilot_dir(PROJECT, cfg)
    if pilot_path:
        proc = pilot_path
        if not (proc / "train.parquet").exists():
            raise SystemExit(
                f"Missing {proc}/train.parquet\n"
                "Run: ./run_binemo_l40.sh sm_ch_finetune/scripts/01d_build_pilot500_finetune.py "
                "--config <your-pilot-config.yaml>"
            )
        print(f"Pilot data: {proc} (train={len(pd.read_parquet(proc/'train.parquet'))}, "
              f"valid={len(pd.read_parquet(proc/'valid.parquet'))})")
    elif cfg["data"].get("use_cds_filter") and not (proc / "train.parquet").exists():
        raise SystemExit(
            f"Missing CDS-filtered data at {proc}\n"
            "Run: ./run_binemo_l40.sh sm_ch_finetune/scripts/01b_filter_cds_window.py --config sm_ch_finetune/config/pilot.yaml"
        )
    use_cds_mask = cfg["data"].get("use_cds_mask", cfg["data"].get("use_cds_filter", False))
    gtf = project_subpath(PROJECT, cfg["data"]["gtf_path"]) if use_cds_mask else None
    ch_only = bool(cfg["data"].get("ch_only", False))
    age_target_col = str(cfg["data"].get("age_target_col", "gnomad_age_mean_norm"))
    dm_kw = dict(
        window_size=cfg["model"]["window_size"],
        batch_size=cfg["training"]["batch_size"],
        gtf_path=gtf,
        use_cds_mask=use_cds_mask,
        ch_only=ch_only,
        age_target_col=age_target_col,
    )

    train_dm = VariantDataModule(proc/"train.parquet", fasta, shuffle=True, **dm_kw)
    valid_dm = VariantDataModule(proc/"valid.parquet", fasta, **dm_kw)
    if ch_only:
        print(f"CH-only training: age_target={age_target_col}")

    ckpt = project_subpath(PROJECT, cfg["checkpoint_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)

    max_steps = 20 if args.smoke_test else cfg["training"].get("max_train_steps")
    epochs = 1 if args.smoke_test else cfg["training"]["num_epochs"]

    print(f"Model ready. Train batches={len(train_dm)} Valid={len(valid_dm)}")
    print(f"Strategy={strategy} window={cfg['model']['window_size']} detach_backbone={detach_backbone}")
    print(f"Checkpoints -> {ckpt}")

    train_variant_classifier(
        model, train_dm, valid_dm, head_id,
        organism=cfg["model"]["organism"],
        finetune_strategy=strategy,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        num_epochs=epochs,
        max_train_steps=max_steps,
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 0),
        checkpoint_dir=ckpt,
        seed=cfg["data"].get("seed", 42),
    )

if __name__ == "__main__": main()
