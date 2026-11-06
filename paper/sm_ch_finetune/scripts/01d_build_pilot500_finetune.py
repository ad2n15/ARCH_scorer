#!/usr/bin/env python3
"""Build train/valid/test parquets for fold_1 pilot on the same variants as API track scoring."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from paths import project_subpath, resolve_config, resolve_split_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pilot_500valid.yaml")
    p.add_argument("--scores-parquet", default=None, help="API wide features (variant_id list)")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument(
        "--n-variants",
        type=int,
        default=None,
        help="Sample this many variants before split (default: all from source)",
    )
    p.add_argument(
        "--source",
        choices=("scores", "processed", "age_filtered"),
        default=None,
        help="scores=API parquet + CDS valid; processed=processed/<pool>; "
        "age_filtered=processed_age_filtered/<pool> (run 01e first)",
    )
    p.add_argument(
        "--pool",
        default=None,
        choices=("train", "valid", "test", "all"),
        help="Split to sample from (default: cfg data.pilot_pool or valid for processed, train for age_filtered)",
    )
    p.add_argument("--output-dir", default=None, help="Override cfg data.pilot500_dir")
    p.add_argument(
        "--func",
        default=None,
        help="Keep variants with Func.refGene equal to this value (e.g. exonic)",
    )
    args = p.parse_args()

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("data", {}).get("seed", 42)
    source = args.source or cfg["data"].get("pilot_source", "scores")

    pool = args.pool or cfg["data"].get("pilot_pool")
    if pool is None:
        pool = "train" if source == "age_filtered" else "valid"

    if source == "age_filtered":
        age_dir = project_subpath(PROJECT, cfg["data"]["age_filtered_dir"])
        if pool == "all":
            parts = []
            for split in ("train", "valid", "test"):
                pth = age_dir / f"{split}.parquet"
                if not pth.exists():
                    raise SystemExit(
                        f"Missing {pth}\n"
                        "Run: ./run_binemo_l40.sh sm_ch_finetune/scripts/01e_build_age_filtered_splits.py"
                    )
                parts.append(pd.read_parquet(pth))
            df = pd.concat(parts, ignore_index=True)
            cds_source = str(age_dir)
        else:
            proc_path = age_dir / f"{pool}.parquet"
            if not proc_path.exists():
                raise SystemExit(
                    f"Missing {proc_path}\n"
                    "Run: ./run_binemo_l40.sh sm_ch_finetune/scripts/01e_build_age_filtered_splits.py"
                )
            df = pd.read_parquet(proc_path)
            cds_source = str(proc_path)
        df["variant_id"] = df["variant_id"].astype(str)
        n_scored = len(df)
    elif source == "processed":
        proc_path = project_subpath(PROJECT, cfg["data"]["processed_dir"]) / f"{pool}.parquet"
        if not proc_path.exists():
            raise SystemExit(f"Missing {proc_path}\nRun 01_prepare_pilot_data.py first")
        df = pd.read_parquet(proc_path)
        df["variant_id"] = df["variant_id"].astype(str)
        cds_source = str(proc_path)
        n_scored = len(df)
    else:
        scores_path = Path(args.scores_parquet) if args.scores_parquet else project_subpath(
            PROJECT, cfg["data"].get("scores_parquet", "results/track_scores/valid_features_wide.parquet")
        )
        if not scores_path.exists():
            raise SystemExit(f"Missing {scores_path}\nRun 05_score_variant_tracks.py first")

        cds_dir = resolve_split_dir(PROJECT, cfg, prefer_mb=False)
        cds_valid = cds_dir / "valid.parquet"
        if not cds_valid.exists():
            raise SystemExit(f"Missing {cds_valid}\nRun 01b_filter_cds_window.py first")

        scored_ids = pd.read_parquet(scores_path, columns=["variant_id"])["variant_id"].astype(str)
        cds = pd.read_parquet(cds_valid)
        cds["variant_id"] = cds["variant_id"].astype(str)
        df = cds[cds["variant_id"].isin(set(scored_ids))].copy()
        n_scored = int(len(scored_ids))
        cds_source = str(cds_valid)
        if len(df) != n_scored:
            missing = n_scored - len(df)
            print(f"Warning: {missing} scored variants missing from {cds_valid} (CDS 32kb filter)")

    func_filter = args.func or cfg.get("data", {}).get("func_filter")
    if func_filter:
        before = len(df)
        df = df[df["func"].astype(str) == func_filter].reset_index(drop=True)
        print(f"func={func_filter!r}: {len(df)}/{before} variants kept")
        if df.empty:
            raise SystemExit(f"No variants with func={func_filter!r}")

    n_take = args.n_variants or cfg["data"].get("pilot_n_variants")
    if n_take is not None:
        if n_take > len(df):
            print(
                f"WARNING: pilot_n_variants={n_take} > pool size {len(df)} "
                f"({source}/{pool}). Using all {len(df)} variants."
            )
            n_take = len(df)
        if n_take < len(df):
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(df), size=int(n_take), replace=False)
            df = df.iloc[pick].reset_index(drop=True)
            print(f"Sampled {n_take} variants (seed={seed}) from {source}/{pool} pool (n={n_scored})")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = max(1, int(len(df) * args.holdout_frac))
    test = df.iloc[idx[:n_test]].sort_values("variant_id").reset_index(drop=True)
    train = df.iloc[idx[n_test:]].sort_values("variant_id").reset_index(drop=True)
    # valid = same holdout for early stopping during training (matches 06 eval test fold)
    valid = test.copy()

    out_rel = args.output_dir or cfg["data"].get("pilot500_dir") or cfg["data"].get("pilot_dir")
    if not out_rel:
        raise SystemExit("Set data.pilot500_dir or data.pilot_dir in config, or pass --output-dir")
    out = project_subpath(PROJECT, out_rel)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    valid.to_parquet(out / "valid.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)

    meta = {
        "n_scored": n_scored,
        "n_selected": int(len(df)),
        "train_n": int(len(train)),
        "valid_n": int(len(valid)),
        "test_n": int(len(test)),
        "holdout_frac": args.holdout_frac,
        "seed": seed,
        "source": source,
        "pool": pool,
        "cds_source": cds_source,
    }
    if func_filter:
        meta["func_filter"] = func_filter
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Selected {len(df)} variants from {source} (pool n={n_scored})")
    print(f"train={len(train)} valid={len(valid)} test={len(test)} (CH rate test={test['label'].mean():.3f})")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
