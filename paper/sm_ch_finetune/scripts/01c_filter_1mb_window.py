#!/usr/bin/env python3
"""Filter variants whose annotated gene/CDS fits fully in a 1 Mb window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data.gencode_cds import GencodeCdsIndex
from paths import project_subpath, resolve_config


def _sequence_length(cfg: dict) -> int:
    if cfg.get("model", {}).get("sequence_length") is not None:
        return int(cfg["model"]["sequence_length"])
    if cfg.get("api", {}).get("sequence_length") is not None:
        return int(cfg["api"]["sequence_length"])
    raise KeyError("config must define model.sequence_length or api.sequence_length")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/track_scoring.yaml")
    args = p.parse_args()

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    ws = _sequence_length(cfg)
    gtf = project_subpath(PROJECT, cfg["data"]["gtf_path"])
    src_dir = project_subpath(PROJECT, cfg["data"]["cds_filtered_dir" if cfg["data"].get("use_cds_filter") else "processed_dir"])
    out = project_subpath(PROJECT, cfg["data"]["mb_filtered_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading GENCODE index from {gtf} (window={ws:,} bp)...")
    idx = GencodeCdsIndex.from_gtf(gtf)

    stats = {"window_size": ws, "splits": {}}
    for split in ("train", "valid", "test", "pilot_balanced_snv"):
        src = src_dir / f"{split}.parquet"
        if not src.exists():
            continue
        df = pd.read_parquet(src)
        keep = []
        for _, row in df.iterrows():
            gene = str(row.get("gene", "."))
            if gene in (".", "", "nan"):
                keep.append(False)
                continue
            keep.append(idx.cds_fits_in_window(str(row["chrom"]), int(row["pos"]), gene, ws))
        df = df.copy()
        df["cds_in_1mb"] = keep
        filt = df[df["cds_in_1mb"]].copy()
        filt.to_parquet(out / f"{split}.parquet", index=False)
        stats["splits"][split] = {"input": int(len(df)), "cds_in_1mb": int(len(filt))}
        print(f"{split}: {len(filt)}/{len(df)} pass CDS-in-1Mb filter")

    with open(out / "filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
