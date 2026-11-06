#!/usr/bin/env python3
"""Score SM/CH variants via AlphaGenome API (1 Mb, CD34+ tracks). Requires internet + ALPHA_GENOME_API_KEY."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import pyfaidx
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from paths import project_subpath, resolve_config, resolve_data_split_dir
from scoring.api_client import load_api_client
from scoring.scorers import build_scorer_list, scores_to_features
from scoring.variant_interval import api_chrom, chrom_length, make_variant, variant_scoring_interval


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "scored_interval" in out.columns:
        out["scored_interval"] = out["scored_interval"].astype(str)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="AlphaGenome API variant track scoring (not fine-tuning).")
    p.add_argument("--config", default="config/track_scoring.yaml")
    p.add_argument("--split", default=None)
    p.add_argument("--max-variants", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    if "api" not in cfg or "scoring" not in cfg:
        raise SystemExit(
            f"Config {args.config!r} is not a track-scoring config (missing 'api' / 'scoring').\n"
            "Use e.g. config/track_scoring_full_age_filtered.yaml or "
            "config/track_scoring_pilot5k_exonic.yaml — not a pilot_*.yaml fine-tune config."
        )

    api_cfg = cfg["api"]
    client = load_api_client(
        key_env=api_cfg.get("key_env", "ALPHA_GENOME_API_KEY"),
        model_version=api_cfg.get("model_version"),
    )

    scorers = build_scorer_list(
        cfg["scoring"]["scorers"],
        include_polyadenylation=cfg["scoring"].get("include_polyadenylation", False),
    )
    ontology_curies = set(cfg["scoring"]["ontology_terms"])
    raw_only = bool(cfg["scoring"].get("raw_only", False))
    include_quantile = not raw_only
    seq_len = int(api_cfg["sequence_length"])
    scores_dir = project_subpath(PROJECT, cfg.get("scores_dir", "results/track_scores"))
    scores_dir.mkdir(parents=True, exist_ok=True)

    fasta = project_subpath(PROJECT, cfg["data"]["fasta_path"])
    fa = pyfaidx.Fasta(str(fasta))
    fasta_keys = frozenset(fa.keys())
    chrom_lens = {k: len(fa[k]) for k in fa.keys()}

    splits = [args.split] if args.split else cfg["scoring"].get("splits", ["test"])
    if args.smoke_test:
        splits = splits[:1]

    data_dir = resolve_data_split_dir(PROJECT, cfg, prefer_mb=True)
    max_n = 2 if args.smoke_test else (args.max_variants or cfg["scoring"].get("max_variants"))

    for split in splits:
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing {path} — run 01c_filter_1mb_window.py first")
        df = pd.read_parquet(path)
        if max_n:
            df = df.head(int(max_n))

        long_path = scores_dir / f"{split}_scores_long.parquet"
        wide_path = scores_dir / f"{split}_features_wide.parquet"
        done_ids: set[str] = set()
        wide_rows: list[dict] = []

        if cfg["scoring"].get("resume", True) and wide_path.exists():
            prev = pd.read_parquet(wide_path)
            if raw_only:
                drop_q = [c for c in prev.columns if c.endswith("__quantile")]
                if drop_q:
                    prev = prev.drop(columns=drop_q)
            # Keep only alleles still in the current pilot (drop older-pool leftovers)
            keep = set(df["variant_id"].astype(str))
            prev = prev.loc[prev["variant_id"].astype(str).isin(keep)].copy()
            done_ids = set(prev["variant_id"].astype(str))
            wide_rows = prev.to_dict("records")
            print(f"Resuming {split}: {len(done_ids)}/{len(df)} variants already scored")

        long_chunks: list[pd.DataFrame] = []
        if cfg["scoring"].get("resume", True) and long_path.exists():
            prev_long = pd.read_parquet(long_path)
            keep = set(df["variant_id"].astype(str))
            if "variant_id" in prev_long.columns:
                prev_long = prev_long.loc[prev_long["variant_id"].astype(str).isin(keep)]
            if len(prev_long):
                long_chunks.append(prev_long)
        errors = []
        for _, row in df.iterrows():
            vid = str(row["variant_id"])
            if vid in done_ids:
                continue
            chrom = str(row["chrom"])
            ck = api_chrom(chrom)
            try:
                interval = variant_scoring_interval(
                    ck, int(row["pos"]), width=seq_len,
                    chrom_len=chrom_length(chrom, chrom_lens),
                    fasta_keys=None,
                )
                variant = make_variant(
                    ck, int(row["pos"]), str(row["ref"]), str(row["alt"]),
                    name=vid, fasta_keys=None,
                )
                adatas = client.score_variant(interval, variant, scorers)
                long_df, feats = scores_to_features(
                    adatas,
                    gene_name=str(row.get("gene", ".")),
                    ontology_curies=ontology_curies,
                    match_gene_strand=cfg["scoring"].get("match_gene_strand", True),
                    include_quantile=include_quantile,
                )
                if not long_df.empty:
                    long_df["variant_id"] = vid
                    long_df["label"] = int(row["label"])
                    long_df["split"] = split
                    long_chunks.append(_parquet_safe(long_df))

                wide_rows.append({
                    "variant_id": vid,
                    "label": int(row["label"]),
                    "gene": row.get("gene", "."),
                    **feats,
                })
                done_ids.add(vid)

                if len(wide_rows) % 5 == 0 or args.smoke_test:
                    pd.DataFrame(wide_rows).to_parquet(wide_path, index=False)
                print(f"[{split}] {len(done_ids)}/{len(df)} {vid} features={len(feats)}")
            except Exception as e:
                errors.append({"variant_id": vid, "error": str(e)})
                print(f"ERROR {vid}: {e}")
                if args.smoke_test:
                    traceback.print_exc()
                    raise

        pd.DataFrame(wide_rows).to_parquet(wide_path, index=False)
        if long_chunks:
            pd.concat([_parquet_safe(c) for c in long_chunks], ignore_index=True).to_parquet(
                long_path, index=False
            )
        meta = {
            "backend": "api",
            "split": split,
            "n_variants": len(wide_rows),
            "n_features": max(len(r) - 3 for r in wide_rows) if wide_rows else 0,
            "raw_only": raw_only,
            "ontology_terms": list(ontology_curies),
            "scorers": cfg["scoring"]["scorers"],
            "sequence_length": seq_len,
            "errors": errors,
        }
        with open(scores_dir / f"{split}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved {wide_path} ({len(wide_rows)} variants, {meta['n_features']} features)")


if __name__ == "__main__":
    main()
