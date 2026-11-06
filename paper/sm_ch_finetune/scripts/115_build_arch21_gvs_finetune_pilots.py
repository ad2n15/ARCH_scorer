#!/usr/bin/env python3
"""Build train/valid/test for heads-only finetune: ARCH21 germline vs somatic (MANE NC).

Stratifies on gene × Func.refGene × label (rare strata collapsed), then splits into
train / valid / test.

label: 0=germline, 1=somatic (matches VariantDataModule / VariantClassifierHead).

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/115_build_arch21_gvs_finetune_pilots.py \\
      --region noncoding --train-frac 0.80 --valid-frac 0.10 --test-frac 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from paths import project_subpath

_s106 = spec_from_file_location("m106", PROJECT / "scripts/106_cv5_pooled_arch_genes.py")
_m106 = module_from_spec(_s106)
assert _s106.loader is not None
_s106.loader.exec_module(_m106)
ARCH21 = _m106.ARCH21
_load_annovar = _m106._load_annovar
_collapse_rare_strata = _m106._collapse_rare_strata

KEEP = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "label",
    "source",
    "Func.refGene",
    "split",
    "region_class",
]


def _annovar(gene: str, region: str) -> Path:
    if region == "coding":
        return project_subpath(
            PROJECT,
            f"data/arch_mane_coding_pooled_sm/annovar_by_gene/{gene}_annovar.hg38_multianno.csv",
        )
    return project_subpath(
        PROJECT,
        f"data/arch_mane_noncoding_pooled_sm/annovar_by_gene/{gene}_annovar.hg38_multianno.csv",
    )


def _cohort(gene: str, region: str) -> Path:
    if region == "coding":
        return project_subpath(PROJECT, f"data/arch_mane_coding_pooled_sm/by_gene/{gene}.csv")
    return project_subpath(PROJECT, f"data/arch_mane_noncoding_pooled_sm/by_gene/{gene}.csv")


def _load_region(genes: list[str], region: str) -> pd.DataFrame:
    rows = []
    for gene in genes:
        path = _cohort(gene, region)
        if not path.is_file():
            raise SystemExit(f"Missing cohort: {path}")
        df = pd.read_csv(path)
        df["variant_id"] = df["variant_id"].astype(str)
        df["gene"] = gene
        df["region_class"] = region
        ann_p = _annovar(gene, region)
        if ann_p.is_file():
            ann = _load_annovar(ann_p)
            df = df.merge(ann, on="variant_id", how="left")
        if "Func.refGene" not in df.columns:
            df["Func.refGene"] = "unknown"
        df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _split_once(
    idx: np.ndarray,
    strata: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(sss.split(np.zeros(len(idx)), strata))
    return idx[tr], idx[te]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes", default=",".join(ARCH21))
    ap.add_argument(
        "--region",
        choices=("noncoding", "coding", "both"),
        default="noncoding",
        help="noncoding recommended for primary heads-only run",
    )
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        default="data/pilot_arch21_mane_noncoding_gvs_finetune",
    )
    ap.add_argument(
        "--holdout-genes",
        default=None,
        help="Optional comma list excluded from train/valid (entire genes → test only)",
    )
    args = ap.parse_args()

    if abs(args.train_frac + args.valid_frac + args.test_frac - 1.0) > 1e-6:
        raise SystemExit("train+valid+test fracs must sum to 1")

    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    regions = ["noncoding", "coding"] if args.region == "both" else [args.region]
    parts = [_load_region(genes, r) for r in regions]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("variant_id", keep="first").reset_index(drop=True)

    # Binary label for VariantDataModule
    lab = df["label"].astype(str).str.lower()
    if not set(lab.unique()) <= {"germline", "somatic", "0", "1"}:
        raise SystemExit(f"Unexpected labels: {sorted(lab.unique())}")
    df["label"] = lab.map({"germline": 0, "somatic": 1, "0": 0, "1": 1}).astype(int)

    for c in ("chrom", "pos", "ref", "alt"):
        if c not in df.columns:
            raise SystemExit(f"Missing column {c}")
    df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["pos"] = df["pos"].astype(int)

    holdout = []
    if args.holdout_genes:
        holdout = [g.strip().upper() for g in args.holdout_genes.split(",") if g.strip()]
        missing = [g for g in holdout if g not in set(genes)]
        if missing:
            raise SystemExit(f"holdout genes not in --genes: {missing}")

    y = df["label"].to_numpy()
    g0 = df["gene"].astype(str).to_numpy()
    f0 = df["Func.refGene"].astype(str).str.split(";").str[0].to_numpy()
    strata = np.array([f"{g}_{fi}_{yi}" for g, fi, yi in zip(g0, f0, y)], dtype=object)
    # collapse using min count ~10 so stratified shuffle works
    strata = _collapse_rare_strata(strata, y, n_splits=10)

    idx_all = np.arange(len(df))
    if holdout:
        is_ho = np.isin(g0, holdout)
        idx_ho = idx_all[is_ho]
        idx_pool = idx_all[~is_ho]
        strata_pool = strata[~is_ho]
        # Holdout genes → test only; split remaining pool with same fracs
        tr_va, te_pool = _split_once(
            idx_pool, strata_pool, test_size=args.test_frac, seed=args.seed
        )
        strata_trva = strata[tr_va]
        valid_of_trva = args.valid_frac / (args.train_frac + args.valid_frac)
        tr, va = _split_once(
            tr_va, strata_trva, test_size=valid_of_trva, seed=args.seed + 1
        )
        te = np.concatenate([te_pool, idx_ho])
    else:
        tr_va, te = _split_once(
            idx_all, strata, test_size=args.test_frac, seed=args.seed
        )
        strata_trva = strata[tr_va]
        valid_of_trva = args.valid_frac / (args.train_frac + args.valid_frac)
        tr, va = _split_once(
            tr_va, strata_trva, test_size=valid_of_trva, seed=args.seed + 1
        )

    out = project_subpath(PROJECT, args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": tr, "valid": va, "test": te}
    summary = {"n": len(df), "genes": genes, "region": args.region, "holdout_genes": holdout}
    for name, ix in splits.items():
        part = df.iloc[ix].copy()
        part["split"] = name
        cols = [c for c in KEEP if c in part.columns]
        part[cols].to_parquet(out / f"{name}.parquet", index=False)
        summary[name] = {
            "n": int(len(part)),
            "n_somatic": int(part["label"].sum()),
            "n_germline": int((part["label"] == 0).sum()),
            "n_genes": int(part["gene"].nunique()),
        }
        print(
            f"{name}: n={len(part)} somatic={int(part['label'].sum())} "
            f"genes={part['gene'].nunique()}",
            flush=True,
        )

    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
