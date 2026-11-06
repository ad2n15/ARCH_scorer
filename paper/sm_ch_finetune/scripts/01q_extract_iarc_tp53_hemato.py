#!/usr/bin/env python3
"""Extract unique haematological TP53 SNVs from IARC R21 for AlphaGenome API scoring.

Required API fields: chrom, pos, ref, alt, variant_id (+ gene/label for pipeline).
Uses g_description_GRCh38 (genomic strand), not WT/Mutant coding-strand bases.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SNV_RE = re.compile(r"^g\.(\d+)([ACGT])>([ACGT])$")
DEFAULT_TUMOR = PROJECT.parent / "TumorVariantDownload_r21.csv"
DEFAULT_GERM = PROJECT.parent / "GermlineDownload_r21.csv"


def _parse_snv(g_desc: str) -> tuple[int, str, str] | None:
    m = SNV_RE.fullmatch(str(g_desc).strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def _load_hemato(path: Path, source: str, *, include_lymph: bool) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    topo = df["Short_topo"].fillna("").astype(str)
    keep = topo.eq("HEMATOP. SYSTEM")
    if include_lymph:
        keep |= topo.eq("LYMPH NODES")
    df = df.loc[keep].copy()
    parsed = df["g_description_GRCh38"].map(_parse_snv)
    df = df.loc[parsed.notna()].copy()
    df["pos"] = [p[0] for p in parsed.loc[df.index]]
    df["ref"] = [p[1] for p in parsed.loc[df.index]]
    df["alt"] = [p[2] for p in parsed.loc[df.index]]
    df["chrom"] = "chr17"
    df["variant_id"] = (
        "17-" + df["pos"].astype(str) + "-" + df["ref"] + "-" + df["alt"]
    )
    df["iarc_source"] = source
    df["gene"] = "TP53"
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tumor", type=Path, default=DEFAULT_TUMOR)
    p.add_argument("--germline", type=Path, default=DEFAULT_GERM)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "data/pilot_iarc_tp53_hemato",
    )
    p.add_argument(
        "--include-lymph-nodes",
        action="store_true",
        help="Also keep Short_topo == LYMPH NODES",
    )
    args = p.parse_args()

    parts = []
    if args.tumor.exists():
        parts.append(_load_hemato(args.tumor, "tumor", include_lymph=args.include_lymph_nodes))
    else:
        raise SystemExit(f"Missing tumor CSV: {args.tumor}")
    if args.germline.exists():
        parts.append(
            _load_hemato(args.germline, "germline", include_lymph=args.include_lymph_nodes)
        )
    else:
        raise SystemExit(f"Missing germline CSV: {args.germline}")

    all_rows = pd.concat(parts, ignore_index=True)
    # Prefer keeping both sources as flags; collapse to unique genomic SNVs
    uniq = (
        all_rows.sort_values(["variant_id", "iarc_source"])
        .groupby("variant_id", as_index=False)
        .agg(
            chrom=("chrom", "first"),
            pos=("pos", "first"),
            ref=("ref", "first"),
            alt=("alt", "first"),
            gene=("gene", "first"),
            MUT_ID=("MUT_ID", "first"),
            Short_topo=("Short_topo", "first"),
            g_description_GRCh38=("g_description_GRCh38", "first"),
            c_description=("c_description", "first"),
            ProtDescription=("ProtDescription", "first"),
            Effect=("Effect", "first"),
            n_iarc_records=("variant_id", "size"),
            sources=("iarc_source", lambda s: ",".join(sorted(set(s)))),
            in_tumor=("iarc_source", lambda s: "tumor" in set(s)),
            in_germline=("iarc_source", lambda s: "germline" in set(s)),
        )
    )
    uniq["label"] = 1  # somatic/haem context placeholder for scoring schema
    uniq["label_str"] = "iarc_hemato"
    uniq["split"] = "test"  # score as a single split

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "variant_id",
        "gene",
        "label",
        "label_str",
        "split",
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
    table = uniq[keep_cols].sort_values("pos").reset_index(drop=True)
    table.to_parquet(out / "test.parquet", index=False)
    table.to_csv(out / "iarc_tp53_hemato_unique_snvs.csv", index=False)
    meta = {
        "n_unique_snvs": int(len(table)),
        "n_tumor_only": int((table["sources"] == "tumor").sum()),
        "n_germline_only": int((table["sources"] == "germline").sum()),
        "n_both": int((table["sources"] == "germline,tumor").sum()),
        "include_lymph_nodes": bool(args.include_lymph_nodes),
        "out_dir": str(out),
    }
    pd.Series(meta).to_json(out / "extract_summary.json")
    print(meta)
    print(f"Wrote {out / 'test.parquet'} and CSV ({len(table)} unique SNVs)")


if __name__ == "__main__":
    main()
