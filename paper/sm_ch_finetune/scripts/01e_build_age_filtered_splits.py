#!/usr/bin/env python3
"""Build train/valid/test parquets from age-filtered SM/CH variants (full cohort, no subsampling).

Reads data/processed_age_filtered/sm_ch_age_filtered.parquet (from 00_filter_no_age_sm.py),
applies a random stratified split (default 90/5/5), and writes split parquets for fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.variant_table import build_variant_table, random_split, split_by_chromosome
from paths import project_subpath, resolve_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/pilot.yaml")
    p.add_argument(
        "--input",
        default=None,
        help="Override age-filtered parquet/csv (default: age_filtered_dir/sm_ch_age_filtered.parquet)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output dir (default: data/processed_age_filtered)",
    )
    p.add_argument(
        "--split-method",
        choices=("random", "chromosome"),
        default=None,
        help="Override cfg data.split_method (default: random)",
    )
    args = p.parse_args()

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    data = cfg["data"]
    seed = int(data.get("seed", 42))
    split_method = args.split_method or data.get("split_method", "random")
    out_dir = project_subpath(PROJECT, args.output_dir or data.get("age_filtered_dir", "data/processed_age_filtered"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        src = Path(args.input)
        if not src.is_absolute():
            src = project_subpath(PROJECT, src)
    else:
        src = out_dir / "sm_ch_age_filtered.parquet"
        if not src.exists():
            src = out_dir / "sm_ch_age_filtered.csv"
    if not src.exists():
        raise SystemExit(
            f"Missing {src}\n"
            "Run: ./run_binemo_l40.sh sm_ch_finetune/scripts/00_filter_no_age_sm.py --config <config>"
        )

    print(f"Loading {src} ...")
    if src.suffix == ".parquet":
        raw = pd.read_parquet(src)
    else:
        raw = pd.read_csv(src)

    table = build_variant_table(raw)
    if split_method == "chromosome":
        train_c = set(data["train_chroms"])
        valid_c = set(data["valid_chroms"])
        test_c = set(data["test_chroms"])
        table = split_by_chromosome(
            table, train_chroms=train_c, valid_chroms=valid_c, test_chroms=test_c
        )
        split_meta = {
            "method": "chromosome",
            "train_chroms": sorted(train_c),
            "valid_chroms": sorted(valid_c),
            "test_chroms": sorted(test_c),
        }
    else:
        train_frac = float(data.get("train_frac", 0.9))
        valid_frac = float(data.get("valid_frac", 0.05))
        test_frac = float(data.get("test_frac", 0.05))
        table = random_split(
            table,
            train_frac=train_frac,
            valid_frac=valid_frac,
            test_frac=test_frac,
            seed=seed,
        )
        split_meta = {
            "method": "random",
            "seed": seed,
            "train_frac": train_frac,
            "valid_frac": valid_frac,
            "test_frac": test_frac,
            "stratify_col": "label",
        }
        print(
            f"Random stratified split: {train_frac:.0%} train / "
            f"{valid_frac:.0%} valid / {test_frac:.0%} test (seed={seed})"
        )

    stats = {
        "source": str(src),
        "input_rows": int(len(raw)),
        "snv_parsed": int(len(table)),
        "splits": {},
        **split_meta,
    }
    for split in ("train", "valid", "test"):
        part = table[table["split"] == split].sort_values("variant_id").reset_index(drop=True)
        part.to_parquet(out_dir / f"{split}.parquet", index=False)
        stats["splits"][split] = {
            "n": int(len(part)),
            "n_sm": int((part["label"] == 0).sum()),
            "n_ch": int((part["label"] == 1).sum()),
            "ch_rate": float(part["label"].mean()) if len(part) else None,
        }
        print(
            f"{split}: n={len(part):,}  SM={(part['label']==0).sum():,}  "
            f"CH={(part['label']==1).sum():,}  CH_rate={part['label'].mean():.3f}"
        )

    with open(out_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved train/valid/test -> {out_dir}")
    print("Rebuild pilots after changing splits: 01d_build_pilot500_finetune.py")


if __name__ == "__main__":
    main()
