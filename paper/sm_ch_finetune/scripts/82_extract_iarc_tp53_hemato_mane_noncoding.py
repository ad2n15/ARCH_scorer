#!/usr/bin/env python3
"""Extract IARC TP53 haematological SNVs in MANE Select noncoding windows.

Criteria match the ARCH pooled noncoding definition:
  MANE Select TP53 (ENST00000269305.9): intron + UTR5/UTR3 + 200 bp promoter.
  SNVs only (already enforced in the haemato extract).

Default input: data/pilot_iarc_tp53_hemato/iarc_tp53_hemato_unique_snvs.csv
Default keep: in_tumor == True (haemato somatic / tumour catalogue).

Example:
  cd paper/sm_ch_finetune
  python3 scripts/82_extract_iarc_tp53_hemato_mane_noncoding.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TP53_MANE_TID = "ENST00000269305.9"


def _load_m61():
    path = PROJECT / "scripts/61_build_arch_mane_noncoding_lists.py"
    spec = importlib.util.spec_from_file_location("m61", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--iarc-hemato",
        default=str(
            PROJECT / "data/pilot_iarc_tp53_hemato/iarc_tp53_hemato_unique_snvs.csv"
        ),
    )
    p.add_argument(
        "--gtf",
        default=str(PROJECT / "data/reference/gencode.v46.annotation.gtf"),
    )
    p.add_argument(
        "--mane-transcript",
        default=TP53_MANE_TID,
    )
    p.add_argument(
        "--include-germline-only",
        action="store_true",
        help="Keep haemato alleles seen only in IARC germline file (default: tumor only).",
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT / "data/arch_mane_noncoding"),
    )
    args = p.parse_args()

    src = Path(args.iarc_hemato)
    if not src.is_file():
        raise SystemExit(
            f"Missing IARC haemato extract: {src}\n"
            "Run: python3 scripts/01q_extract_iarc_tp53_hemato.py"
        )

    df = pd.read_csv(src)
    n_in = len(df)
    if "in_tumor" in df.columns and not args.include_germline_only:
        df = df.loc[df["in_tumor"].astype(bool)].copy()
    n_tumor = len(df)

    m61 = _load_m61()
    iv = m61.load_mane_noncoding_intervals(Path(args.gtf), args.mane_transcript)
    df["mane_region"] = df["pos"].map(lambda p: m61.classify_pos(int(p), iv))
    df["mane_transcript"] = args.mane_transcript
    out = df.loc[df["mane_region"].notna()].copy()
    out["label"] = "somatic"
    out["sm_source"] = "iarc_hemato_mane_nc"
    out = out.sort_values(["pos", "ref", "alt"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "arch_mane_noncoding_somatic_iarc_tp53_hemato.csv"
    keep = [
        c
        for c in [
            "chrom",
            "pos",
            "ref",
            "alt",
            "variant_id",
            "gene",
            "label",
            "sm_source",
            "mane_region",
            "mane_transcript",
            "MUT_ID",
            "Short_topo",
            "g_description_GRCh38",
            "c_description",
            "ProtDescription",
            "Effect",
            "n_iarc_records",
            "sources",
            "in_tumor",
            "in_germline",
        ]
        if c in out.columns
    ]
    out[keep].to_csv(out_csv, index=False)

    summary = {
        "input": str(src),
        "mane_transcript": args.mane_transcript,
        "tumor_only": not args.include_germline_only,
        "n_input_unique": n_in,
        "n_after_tumor_filter": n_tumor,
        "n_mane_noncoding": int(len(out)),
        "by_region": out["mane_region"].value_counts().to_dict() if len(out) else {},
        "by_effect": out["Effect"].value_counts(dropna=False).to_dict() if len(out) else {},
        "output": str(out_csv),
    }
    (out_dir / "iarc_tp53_hemato_mane_noncoding_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out_csv} ({len(out)} alleles)", flush=True)


if __name__ == "__main__":
    main()
