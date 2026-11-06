#!/usr/bin/env python3
"""Remove SM variants from studies without gnomAD age data (Colin-Leitzinger et al. 2025).

Paper: 503,703 SM variants; 107,098 (21.3%) lack age data and could not be tested for CH.
Those stay labeled SM in the supplementary CSV but should be excluded for SM vs CH training.

Expected after filter (~paper Table S3):
  - SM retained (with age): ~307,244
  - CH retained: ~89,361
  - Total: ~396,605
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

from data.gnomad_age_index import build_age_index, parse_variant_id, variant_key
from paths import project_subpath, resolve_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pilot.yaml")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="First 5000 variants only")
    args = p.parse_args()

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    raw_csv = project_subpath(PROJECT, cfg["data"]["raw_csv"])
    out_dir = project_subpath(PROJECT, cfg["data"].get("age_filtered_dir", "data/processed_age_filtered"))
    out_dir.mkdir(parents=True, exist_ok=True)

    gnomad_cfg = cfg.get("gnomad", {})
    vcf_path = Path(gnomad_cfg.get("vcf_path", "../gnomad/gnomad.exomes.r2.1.1.sites.liftover_grch38.vcf.bgz"))
    if not vcf_path.is_absolute():
        vcf_path = (PROJECT / vcf_path).resolve()
    cache_path = project_subpath(
        PROJECT, gnomad_cfg.get("age_index_cache", "data/cache/gnomad_age_index.pkl")
    )
    if args.rebuild_index and cache_path.exists():
        cache_path.unlink()

    print(f"Loading {raw_csv} ...")
    df = pd.read_csv(raw_csv)
    if args.smoke_test:
        df = df.head(5000)

    keys: set[tuple[str, int, str, str]] = set()
    bad = 0
    for v in df["Variants"]:
        parsed = parse_variant_id(v)
        if parsed is None:
            bad += 1
            continue
        keys.add(variant_key(*parsed))
    print(f"Variants: {len(df):,}  unique keys: {len(keys):,}  unparsed: {bad}")

    if not vcf_path.exists():
        raise SystemExit(f"gnomAD VCF not found: {vcf_path}")
    print(f"Building age index from {vcf_path} (one-time full scan, ~1-2 h) ...")
    age_index = build_age_index(vcf_path, keys, cache_path=cache_path)

    has_age = []
    for v in df["Variants"]:
        parsed = parse_variant_id(v)
        has_age.append(False if parsed is None else age_index.get(variant_key(*parsed), False))
    df = df.copy()
    df["has_gnomad_age"] = has_age

    sm = df["SM / CH"] == "SM"
    ch = df["SM / CH"] == "CH"
    sm_no_age = sm & ~df["has_gnomad_age"]
    sm_age = sm & df["has_gnomad_age"]

    filtered = df[ch | sm_age].copy()
    filtered.to_csv(out_dir / "sm_ch_age_filtered.csv", index=False)
    filtered.to_parquet(out_dir / "sm_ch_age_filtered.parquet", index=False)

    stats = {
        "input_total": int(len(df)),
        "input_sm": int(sm.sum()),
        "input_ch": int(ch.sum()),
        "sm_no_age_removed": int(sm_no_age.sum()),
        "sm_with_age_kept": int(sm_age.sum()),
        "ch_kept": int(ch.sum()),
        "output_total": int(len(filtered)),
        "paper_expected": {
            "sm_no_age_removed": 107098,
            "sm_with_age_kept": 307244,
            "ch_kept": 89361,
            "output_total": 396605,
        },
    }
    with open(out_dir / "age_filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Saved {out_dir / 'sm_ch_age_filtered.parquet'}")


if __name__ == "__main__":
    main()
