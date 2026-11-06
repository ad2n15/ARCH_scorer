#!/usr/bin/env python3
"""Head-only Borzoi-CH fine-tuning."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_dataset import VariantDataset, collate
from model.borzoi_ch import BorzoiCHModel
from paths import (
    configure_torch_cuda,
    project_subpath,
    resolve_ckpt_path,
    resolve_config,
    resolve_pilot_dir,
    resolve_split_dir,
)
from train.trainer import train_borzoi_ch
from torch.utils.data import DataLoader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    with open(resolve_config(args.config)) as f:
        cfg = yaml.safe_load(f)

    configure_torch_cuda()
    ckpt = resolve_ckpt_path(cfg)
    fasta = project_subpath(cfg["data"]["fasta_path"])
    proc = resolve_pilot_dir(cfg) if cfg.get("data", {}).get("pilot_dir") else resolve_split_dir(cfg)

    import torch

    if not torch.cuda.is_available() and not args.smoke_test:
        raise SystemExit("CUDA required — submit via sbatch run_binemo_l40.sh ...")

    window = int(cfg["model"]["window_size"])
    batch_size = 1 if args.smoke_test else int(cfg["training"]["batch_size"])
    epochs = 1 if args.smoke_test else int(cfg["training"]["num_epochs"])
    max_steps = 3 if args.smoke_test else cfg["training"].get("max_train_steps")

    print(f"Data: {proc} | window={window} bs={batch_size}")
    print(f"Checkpoint: {ckpt}")
    model = BorzoiCHModel(ckpt, hidden_dim=int(cfg["model"].get("head_hidden_dim", 256)))

    train_ds = VariantDataset(proc / "train.parquet", fasta, window_size=window)
    valid_ds = VariantDataset(proc / "valid.parquet", fasta, window_size=window)
    if args.smoke_test:
        train_ds.df = train_ds.df.head(4)
        valid_ds.df = valid_ds.df.head(4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    train_borzoi_ch(
        model,
        train_loader,
        valid_loader,
        checkpoint_dir=project_subpath(cfg["checkpoint_dir"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        num_epochs=epochs,
        early_stopping_patience=int(cfg["training"].get("early_stopping_patience", 0)),
        max_train_steps=max_steps,
        grad_accum_steps=int(cfg["training"].get("grad_accum_steps", 1)),
        log_every_steps=int(cfg["training"].get("log_every_steps", 10)),
        seed=int(cfg["data"].get("seed", 42)),
    )


if __name__ == "__main__":
    main()
