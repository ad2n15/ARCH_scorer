#!/usr/bin/env python3
"""Head-only NTv3-CH fine-tuning (parallel to sm_ch_finetune AlphaGenome pipeline)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_dataset import VariantDataset, collate
from model.ntv3_ch import NTv3CHModel, build_ntv3_ch_model, load_tokenizer
from paths import (
    project_subpath,
    resolve_config,
    resolve_hf_model_path,
    resolve_pilot_dir,
    resolve_split_dir,
    setup_hf_offline,
    configure_torch_cuda,
)
from train.trainer import train_ntv3_ch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(resolve_config(args.config)) as f:
        cfg = yaml.safe_load(f)

    setup_hf_offline()
    model_path = resolve_hf_model_path(cfg)
    fasta = project_subpath(cfg["data"]["fasta_path"])
    if not fasta.exists():
        raise SystemExit(f"Missing FASTA: {fasta}")

    import torch

    configure_torch_cuda()
    if not torch.cuda.is_available() and not args.smoke_test:
        raise SystemExit("CUDA required. Submit via sbatch run_binemo_l40.sh ...")

    proc = resolve_pilot_dir(cfg) if cfg.get("data", {}).get("pilot_dir") else resolve_split_dir(cfg)
    for split in ("train", "valid"):
        if not (proc / f"{split}.parquet").exists():
            raise SystemExit(f"Missing {proc}/{split}.parquet")

    window = int(cfg["model"]["window_size"])
    batch_size = 1 if args.smoke_test else int(cfg["training"]["batch_size"])
    epochs = 1 if args.smoke_test else int(cfg["training"]["num_epochs"])
    max_steps = 5 if args.smoke_test else cfg["training"].get("max_train_steps")

    print(f"Data: {proc} | window={window} bs={batch_size}")
    print(f"Model (offline): {model_path}")
    tokenizer = load_tokenizer(model_path)
    model = build_ntv3_ch_model(model_path, cfg["model"])
    print(
        f"Pool: mode={cfg['model'].get('pool_mode', 'point')} "
        f"bin_size={cfg['model'].get('bin_size', 32)} "
        f"op={cfg['model'].get('pool_op', 'mean')}",
        flush=True,
    )

    train_ds = VariantDataset(proc / "train.parquet", fasta, tokenizer, window_size=window)
    valid_ds = VariantDataset(proc / "valid.parquet", fasta, tokenizer, window_size=window)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0,
    )

    ckpt_dir = project_subpath(cfg["checkpoint_dir"])
    train_ntv3_ch(
        model,
        train_loader,
        valid_loader,
        checkpoint_dir=ckpt_dir,
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        num_epochs=epochs,
        early_stopping_patience=int(cfg["training"].get("early_stopping_patience", 0)),
        max_train_steps=max_steps,
        grad_accum_steps=int(cfg["training"].get("grad_accum_steps", 1)),
        log_every_steps=int(cfg["training"].get("log_every_steps", 10)),
        seed=int(cfg["data"].get("seed", 42)),
    )
    print(f"Done. Checkpoint -> {ckpt_dir / 'best_head.pt'}")


if __name__ == "__main__":
    main()
