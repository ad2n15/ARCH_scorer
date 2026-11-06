#!/usr/bin/env python3
"""Filter pilot variants: 32kb window must fully contain CDS of annotated gene (GENCODE v46)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pilot.yaml")
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(PROJECT / "src"))
    from paths import project_subpath, resolve_config
    from data.gencode_cds import GencodeCdsIndex

    with open(resolve_config(args.config, PROJECT)) as f:
        cfg = yaml.safe_load(f)

    gtf = project_subpath(PROJECT, cfg["data"]["gtf_path"])
    proc = project_subpath(PROJECT, cfg["data"]["processed_dir"])
    out = project_subpath(PROJECT, cfg["data"].get("cds_filtered_dir", "data/processed_cds"))
    out.mkdir(parents=True, exist_ok=True)
    ws = cfg["model"]["window_size"]

    print(f"Loading GENCODE CDS index from {gtf} ...")
    idx = GencodeCdsIndex.from_gtf(gtf)

    stats = {"window_size": ws, "splits": {}}
    for split in ("train", "valid", "test", "pilot_balanced_snv"):
        src = proc / f"{split}.parquet"
        if not src.exists():
            continue
        df = pd.read_parquet(src)
        cds_ok, gene_ok, has_gene = [], [], []
        for _, row in df.iterrows():
            gene = str(row.get("gene", "."))
            if gene in (".", "", "nan"):
                cds_ok.append(False)
                gene_ok.append(False)
                has_gene.append(False)
                continue
            has_gene.append(True)
            cds_ok.append(idx.cds_fits_in_window(str(row["chrom"]), int(row["pos"]), gene, ws))
            gene_ok.append(idx.gene_fits_in_window(str(row["chrom"]), int(row["pos"]), gene, ws))

        df = df.copy()
        df["cds_in_window"] = cds_ok
        df["gene_in_window"] = gene_ok
        filt = df[df["cds_in_window"]].copy()
        filt.to_parquet(out / f"{split}.parquet", index=False)
        stats["splits"][split] = {
            "total": int(len(df)),
            "with_gene": int(sum(has_gene)),
            "cds_in_window": int(filt.shape[0]),
            "gene_in_window": int(df["gene_in_window"].sum()),
            "ch_rate_kept": float(filt["label"].mean()) if len(filt) else None,
        }
        print(
            f"{split}: {len(filt)}/{len(df)} variants with full CDS in {ws}bp window "
            f"(gene body fits: {int(df['gene_in_window'].sum())})"
        )

    with open(out / "cds_filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved filtered parquets -> {out}")


if __name__ == "__main__":
    main()
