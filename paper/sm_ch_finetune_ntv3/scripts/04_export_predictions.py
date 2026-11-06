#!/usr/bin/env python3
"""Export prob_ch parquets for train/test (and optional valid) splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_dataset import VariantDataset, collate
from model.ntv3_ch import build_ntv3_ch_model, load_tokenizer
from paths import (
    project_subpath,
    resolve_config,
    resolve_hf_model_path,
    resolve_pilot_dir,
    resolve_split_dir,
    setup_hf_offline,
    configure_torch_cuda,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def score_split(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch.input_ids.to(device), batch.variant_index.to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
        for vid, p in zip(batch.variant_ids, prob):
            rows.append({"variant_id": str(vid), "prob_ch": float(p)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="all")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(resolve_config(args.config)) as f:
        cfg = yaml.safe_load(f)

    setup_hf_offline()
    configure_torch_cuda()
    model_path = resolve_hf_model_path(cfg)
    fasta = project_subpath(cfg["data"]["fasta_path"])
    proc = resolve_pilot_dir(cfg) if cfg.get("data", {}).get("pilot_dir") else resolve_split_dir(cfg)
    ckpt_path = project_subpath(cfg["checkpoint_dir"]) / "best_head.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}\nRun 03_train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = int(cfg["model"]["window_size"])
    batch_size = 1 if args.smoke_test else int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"]))

    tokenizer = load_tokenizer(model_path)
    model = build_ntv3_ch_model(model_path, cfg["model"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.head.load_state_dict(ckpt["head"])

    out_cfg = cfg.get("export", {})
    splits = ["train", "test"] if args.split == "all" else [args.split]
    if args.split == "all" and out_cfg.get("include_valid", False):
        splits = ["train", "valid", "test"]

    results_dir = project_subpath(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_cfg.get("prefix", cfg.get("export_prefix", "ntv3"))

    for split in splits:
        pq = proc / f"{split}.parquet"
        if not pq.exists():
            print(f"Skip missing {pq}")
            continue
        ds = VariantDataset(pq, fasta, tokenizer, window_size=window)
        if args.smoke_test:
            ds.df = ds.df.head(4)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        df = score_split(model, loader, device)
        out_key = f"{split}_predictions"
        out_rel = out_cfg.get(out_key) or f"results/{prefix}_{split}_predictions.parquet"
        out_path = project_subpath(out_rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"{split}: n={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()
