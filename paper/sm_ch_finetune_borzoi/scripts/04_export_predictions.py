#!/usr/bin/env python3
"""Export prob_ch parquets for Borzoi-CH."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
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
from torch.utils.data import DataLoader


@torch.no_grad()
def score_split(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch.x.to(device), batch.bin_index.to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
        for vid, p in zip(batch.variant_ids, prob):
            rows.append({"variant_id": str(vid), "prob_ch": float(p)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=("train", "valid", "test", "all"), default="all")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    with open(resolve_config(args.config)) as f:
        cfg = yaml.safe_load(f)

    configure_torch_cuda()
    ckpt = resolve_ckpt_path(cfg)
    fasta = project_subpath(cfg["data"]["fasta_path"])
    proc = resolve_pilot_dir(cfg) if cfg.get("data", {}).get("pilot_dir") else resolve_split_dir(cfg)
    head_path = project_subpath(cfg["checkpoint_dir"]) / "best_head.pt"
    if not head_path.exists():
        raise SystemExit(f"Missing {head_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = int(cfg["model"]["window_size"])
    batch_size = 1 if args.smoke_test else int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"]))
    out_cfg = cfg.get("export", {})
    prefix = out_cfg.get("prefix", "borzoi")
    splits = ["train", "test"] if args.split == "all" else [args.split]
    if args.split == "all" and out_cfg.get("include_valid"):
        splits = ["train", "valid", "test"]

    model = BorzoiCHModel(ckpt).to(device)
    model.head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True)["head"])

    for split in splits:
        pq = proc / f"{split}.parquet"
        if not pq.exists():
            continue
        ds = VariantDataset(pq, fasta, window_size=window)
        if args.smoke_test:
            ds.df = ds.df.head(4)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        df = score_split(model, loader, device)
        out_rel = out_cfg.get(f"{split}_predictions") or f"results/{prefix}_{split}_predictions.parquet"
        out_path = project_subpath(out_rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"{split}: n={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()
