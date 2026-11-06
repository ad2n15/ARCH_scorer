#!/usr/bin/env python3
"""Prepare balanced SNV pilot dataset with chromosome-based splits."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import yaml

VARIANT_RE = re.compile(r"^(\d+|X|Y)-(\d+)-([ACGT]+)-([ACGT]+)$")

def parse_variant_id(variant: str) -> dict | None:
    m = VARIANT_RE.match(variant.strip())
    if not m: return None
    chrom, pos, ref, alt = m.groups()
    if len(ref) != 1 or len(alt) != 1: return None
    if not set(ref + alt).issubset({"A","C","G","T"}): return None
    return {"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt, "variant_id": variant.strip()}

def mutation_type(ref, alt): return f"{ref}>{alt}"

def assign_split(chrom, train_c, valid_c, test_c):
    if chrom in train_c: return "train"
    if chrom in valid_c: return "valid"
    if chrom in test_c: return "test"
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pilot.yaml")
    args = parser.parse_args()
    import sys
    project_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_dir / "src"))
    from paths import project_subpath, resolve_config, resolve_raw_csv
    with open(resolve_config(args.config, project_dir)) as f:
        cfg = yaml.safe_load(f)
    try:
        raw_csv = resolve_raw_csv(project_dir, cfg)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    print(f"Source CSV: {raw_csv}")
    out_dir = project_subpath(project_dir, cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv)
    df = df[df["Indel"] == "Not Indel"].copy()
    parsed = []
    for _, row in df.iterrows():
        p = parse_variant_id(row["Variants"])
        if p is None: continue
        p.update({
            "label_str": row["SM / CH"], "label": 1 if row["SM / CH"] == "CH" else 0,
            "gene": row.get("Gene.refGene","."), "exonic_func": row.get("ExonicFunc.refGene","."),
            "func": row.get("Func.refGene","."), "mchip": row.get("M-CHIP regions","No"),
            "lchip": row.get("L-CHIP regions","No"),
            "cadd": pd.to_numeric(row.get("CADD_phred","."), errors="coerce"),
            "mutation_type": mutation_type(p["ref"], p["alt"]),
        })
        parsed.append(p)
    all_df = pd.DataFrame(parsed)
    tc, vc, xc = set(cfg["data"]["train_chroms"]), set(cfg["data"]["valid_chroms"]), set(cfg["data"]["test_chroms"])
    all_df["split"] = all_df["chrom"].map(lambda c: assign_split(c, tc, vc, xc))
    all_df = all_df[all_df["split"].notna()].copy()
    sm = all_df[all_df["label"]==0].sample(n=min(cfg["data"]["n_sm"], (all_df["label"]==0).sum()), random_state=cfg["data"]["seed"])
    ch = all_df[all_df["label"]==1].sample(n=min(cfg["data"]["n_ch"], (all_df["label"]==1).sum()), random_state=cfg["data"]["seed"])
    pilot = pd.concat([sm,ch], ignore_index=True).sample(frac=1, random_state=cfg["data"]["seed"])
    pilot.to_parquet(out_dir / "pilot_balanced_snv.parquet", index=False)
    for split in ("train","valid","test"):
        s = pilot[pilot["split"]==split]
        s.to_parquet(out_dir / f"{split}.parquet", index=False)
        print(f"{split}: {len(s)} (CH={s['label'].sum()})")
    api = pilot.rename(columns={"chrom":"CHROM","pos":"POS","ref":"REF","alt":"ALT"})
    api["CHROM"] = api["CHROM"].apply(lambda x: f"chr{x}" if not str(x).startswith("chr") else x)
    api[["variant_id","CHROM","POS","REF","ALT","label","split"]].to_csv(out_dir/"pilot_for_alphagenome.tsv", sep="\t", index=False)
    print(f"\nSaved {len(pilot)} variants. Labels:\n{pilot['label_str'].value_counts()}")

if __name__ == "__main__": main()
