#!/usr/bin/env python3
"""Univariate germline-vs-somatic on ARCH21 coding ∪ noncoding.

Same three tables as scripts/88_univariate_pooled_noncoding.py, on the union:

  Table 1 — AlphaGenome CD34+ continuous (+ CADD): OR per 1 SD
  Table 2 — Annotated effect / region one-vs-rest (Fisher)
  Table 3 — SBS6 pyrimidine substitutions one-vs-rest (Fisher)

Coding annotated_effect = ANNOVAR Func.refGene primary token (fallback mane_region).
Noncoding annotated_effect = Ellingford MANE region + intronic±2 splicing (same as §10).

Caveat: coding is somatic-rich and noncoding germline-rich, so Func/region ORs
are inflated relative to within-region analyses. Prefer Table 1 + per-region reads.

Example:
  python3 scripts/117_univariate_pooled_coding_noncoding.py
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

_s88 = spec_from_file_location(
    "univ88", PROJECT / "scripts/88_univariate_pooled_noncoding.py"
)
_m88 = module_from_spec(_s88)
assert _s88.loader is not None
_s88.loader.exec_module(_m88)

_pub = _m88._pub
_bh_fdr = _m88._bh_fdr
_key = _m88._key
_to_sbs6 = _m88._to_sbs6
_one_vs_rest_fisher = _m88._one_vs_rest_fisher
_continuous_or_per_sd = _m88._continuous_or_per_sd
_SBS6 = _m88._SBS6
REGION_MAP = _m88.REGION_MAP
load_mane_cds_exons = _m88.load_mane_cds_exons
intronic_canonical_splice = _m88.intronic_canonical_splice
GTF = _m88.GTF

ARCH21 = [
    "ASXL1",
    "ASXL2",
    "CBL",
    "DNMT3A",
    "ETNK1",
    "GNAS",
    "GNB1",
    "IDH1",
    "IDH2",
    "JAK2",
    "KIT",
    "KRAS",
    "MPL",
    "NRAS",
    "PPM1D",
    "RUNX1",
    "SETBP1",
    "SF3B1",
    "SRSF2",
    "TET2",
    "TP53",
]

OUT_DIR = PROJECT / "results/arch_mane_coding_noncoding_pooled_sm_univariate"


def _load_arm(kind: str) -> pd.DataFrame:
    root = PROJECT / f"data/arch_mane_{kind}_pooled_sm"
    rows = []
    for gene in ARCH21:
        path = root / "by_gene" / f"{gene}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        df = pd.read_csv(path, low_memory=False)
        df["variant_id"] = df["variant_id"].astype(str)
        df["gene"] = gene
        df["region_class"] = kind
        df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
        df["y"] = (df["label"].astype(str).str.lower() == "somatic").astype(int)
        df["variant_key"] = _key(df, "chrom", "pos", "ref", "alt")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _attach_cadd_arm(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    cadd_dir = PROJECT / f"data/arch_mane_{kind}_pooled_sm/cadd_by_gene"
    parts = []
    for gene in sorted(df["gene"].unique()):
        local = cadd_dir / f"{gene}_cadd_phred_local.csv"
        api = cadd_dir / f"{gene}_cadd_phred_api.csv"
        path = local if local.is_file() else api
        if not path.is_file():
            continue
        cadd = pd.read_csv(path)
        cadd["variant_id"] = cadd["variant_id"].astype(str)
        cadd["cadd_phred"] = pd.to_numeric(cadd["cadd_phred"], errors="coerce")
        parts.append(cadd[["variant_id", "cadd_phred"]])
    if not parts:
        df["cadd_phred"] = np.nan
        return df
    cadd_all = pd.concat(parts, ignore_index=True).drop_duplicates("variant_id")
    return df.merge(cadd_all, on="variant_id", how="left")


def _attach_api_arm(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list[str]]:
    scores = PROJECT / f"results/track_scores_arch_mane_{kind}_pooled_sm_by_gene"
    parts = []
    for gene in sorted(df["gene"].unique()):
        feat = scores / gene / "test_features_wide.parquet"
        if not feat.is_file():
            raise SystemExit(f"Missing API scores: {feat}")
        api = pd.read_parquet(feat)
        api["variant_id"] = api["variant_id"].astype(str)
        parts.append(api.drop_duplicates("variant_id"))
    api_all = pd.concat(parts, ignore_index=True).drop_duplicates("variant_id")
    feat_cols = [c for c in api_all.columns if c in _pub.SHORT]
    joined = df.merge(api_all[["variant_id"] + feat_cols], on="variant_id", how="inner")
    return joined, feat_cols


def _annotate_noncoding(df: pd.DataFrame) -> pd.DataFrame:
    ann_dir = PROJECT / "data/arch_mane_noncoding_pooled_sm/annovar_by_gene"
    parts = []
    for gene, part in df.groupby("gene", sort=True):
        path = ann_dir / f"{gene}_annovar.hg38_multianno.csv"
        ann = pd.read_csv(path, low_memory=False)
        ann["variant_key"] = _key(ann, "Chr", "Start", "Ref", "Alt")
        ann = ann.drop_duplicates("variant_key")
        parts.append(
            part.merge(
                ann[["variant_key", "Func.refGene"]],
                on="variant_key",
                how="left",
                validate="one_to_one",
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out["annotated_effect"] = out["mane_region"].map(REGION_MAP)
    genes = set(out["gene"].astype(str))
    _cds, exons = load_mane_cds_exons(GTF, genes)
    splice = [
        intronic_canonical_splice(c, int(p), exons.get(g, []))
        for c, p, g in zip(out["chrom"], out["pos"], out["gene"])
    ]
    out.loc[pd.Series(splice, index=out.index), "annotated_effect"] = "splicing"
    if out["annotated_effect"].isna().any():
        bad = out.loc[out["annotated_effect"].isna(), "mane_region"].value_counts()
        raise SystemExit(f"Unmapped NC MANE regions:\n{bad}")
    return out


def _annotate_coding(df: pd.DataFrame) -> pd.DataFrame:
    ann_dir = PROJECT / "data/arch_mane_coding_pooled_sm/annovar_by_gene"
    parts = []
    for gene, part in df.groupby("gene", sort=True):
        path = ann_dir / f"{gene}_annovar.hg38_multianno.csv"
        if path.is_file():
            ann = pd.read_csv(path, low_memory=False)
            ann["variant_key"] = _key(ann, "Chr", "Start", "Ref", "Alt")
            ann = ann.drop_duplicates("variant_key")
            part = part.merge(
                ann[["variant_key", "Func.refGene", "ExonicFunc.refGene"]],
                on="variant_key",
                how="left",
            )
        else:
            part = part.copy()
            part["Func.refGene"] = np.nan
            part["ExonicFunc.refGene"] = np.nan
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    # Prefer ExonicFunc when present; else Func; else mane_region/CDS
    ex = out.get("ExonicFunc.refGene")
    func = out.get("Func.refGene")
    effect = []
    for i in range(len(out)):
        e = str(ex.iloc[i]) if ex is not None else "nan"
        f = str(func.iloc[i]) if func is not None else "nan"
        if e not in ("nan", "None", ".", ""):
            effect.append(e.split(";")[0])
        elif f not in ("nan", "None", ".", ""):
            effect.append(f.split(";")[0])
        else:
            mr = str(out["mane_region"].iloc[i]) if "mane_region" in out.columns else "CDS"
            effect.append(mr if mr and mr != "nan" else "CDS")
    out["annotated_effect"] = effect
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading coding + noncoding …", flush=True)
    nc = _annotate_noncoding(_attach_cadd_arm(_load_arm("noncoding"), "noncoding"))
    cd = _annotate_coding(_attach_cadd_arm(_load_arm("coding"), "coding"))
    # align columns
    keep = [
        "variant_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "label",
        "y",
        "source",
        "region_class",
        "annotated_effect",
        "cadd_phred",
        "variant_key",
    ]
    for d in (nc, cd):
        for c in keep:
            if c not in d.columns:
                d[c] = np.nan
    df = pd.concat([nc[keep], cd[keep]], ignore_index=True)
    df = df.drop_duplicates("variant_id", keep="first")
    df["sbs6"] = [
        _to_sbs6(r, a) for r, a in zip(df["ref"].astype(str), df["alt"].astype(str))
    ]
    y = df["y"].astype(int).to_numpy()
    print(
        f"Union n={len(df):,} somatic={int(y.sum()):,} "
        f"coding={int((df.region_class=='coding').sum()):,} "
        f"noncoding={int((df.region_class=='noncoding').sum()):,} "
        f"with CADD={int(df['cadd_phred'].notna().sum()):,}",
        flush=True,
    )

    # API join per arm then concat (features shared)
    api_nc, feat_nc = _attach_api_arm(df[df.region_class == "noncoding"].copy(), "noncoding")
    api_cd, feat_cd = _attach_api_arm(df[df.region_class == "coding"].copy(), "coding")
    feat_cols = sorted(set(feat_nc) & set(feat_cd))
    if not feat_cols:
        raise SystemExit("No shared API feature columns between coding and noncoding")
    api_df = pd.concat([api_nc, api_cd], ignore_index=True).drop_duplicates("variant_id")
    print(
        f"API-scored union n={len(api_df):,} somatic={int(api_df.y.sum()):,} "
        f"features={len(feat_cols)}",
        flush=True,
    )

    # Table 1
    t1_rows = []
    y1 = api_df["y"].astype(int).to_numpy()
    if api_df["cadd_phred"].notna().sum() >= 50:
        t1_rows.append(
            _continuous_or_per_sd(y1, api_df["cadd_phred"].to_numpy(float), "cadd_phred")
        )
    for col in feat_cols:
        t1_rows.append(
            _continuous_or_per_sd(
                y1, api_df[col].to_numpy(float), _pub.SHORT.get(col, col)
            )
        )
    t1 = pd.DataFrame(t1_rows).sort_values("p_value").reset_index(drop=True)
    t1["q_value_bh"] = _bh_fdr(t1["p_value"])
    t1_path = out_dir / "table1_alphagenome_cd34_continuous.csv"
    t1.to_csv(t1_path, index=False)

    # Table 1b: same continuous, noncoding-only and coding-only slices
    for region, sub in api_df.groupby("region_class"):
        rows = []
        yr = sub["y"].astype(int).to_numpy()
        if sub["cadd_phred"].notna().sum() >= 50:
            rows.append(
                _continuous_or_per_sd(
                    yr, sub["cadd_phred"].to_numpy(float), "cadd_phred"
                )
            )
        for col in feat_cols:
            rows.append(
                _continuous_or_per_sd(
                    yr, sub[col].to_numpy(float), _pub.SHORT.get(col, col)
                )
            )
        tab = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
        tab["q_value_bh"] = _bh_fdr(tab["p_value"])
        tab.to_csv(
            out_dir / f"table1_alphagenome_cd34_continuous_{region}.csv", index=False
        )

    # Table 2: annotated effects + region_class
    t2_effect = _one_vs_rest_fisher(df["annotated_effect"], y, min_n=args.min_n)
    t2_region = _one_vs_rest_fisher(df["region_class"], y, min_n=args.min_n)
    t2_effect["table"] = "annotated_effect"
    t2_region["table"] = "region_class"
    t2 = pd.concat([t2_region, t2_effect], ignore_index=True)
    t2_path = out_dir / "table2_annotated_effects_one_vs_rest.csv"
    t2.to_csv(t2_path, index=False)

    # Table 3: SBS6
    snv = df.loc[df["sbs6"].notna()].copy()
    y3 = snv["y"].astype(int).to_numpy()
    t3 = _one_vs_rest_fisher(snv["sbs6"], y3, min_n=args.min_n)
    t3_path = out_dir / "table3_sbs6_one_vs_rest.csv"
    t3.to_csv(t3_path, index=False)
    rates = (
        snv.groupby("sbs6")["y"]
        .agg(n="count", n_somatic="sum")
        .reindex(_SBS6)
        .reset_index()
        .rename(columns={"sbs6": "feature"})
    )
    rates["somatic_rate"] = rates["n_somatic"] / rates["n"]
    rates.to_csv(out_dir / "table3_sbs6_rates.csv", index=False)

    summary = {
        "n_total": int(len(df)),
        "n_somatic": int(y.sum()),
        "n_coding": int((df.region_class == "coding").sum()),
        "n_noncoding": int((df.region_class == "noncoding").sum()),
        "n_with_cadd": int(df["cadd_phred"].notna().sum()),
        "table1_n_variants": int(len(api_df)),
        "table1_n_somatic": int(api_df["y"].sum()),
        "table1_features": [_pub.SHORT.get(c, c) for c in feat_cols],
        "caveat": (
            "coding somatic-rich / noncoding germline-rich inflates region and "
            "Func ORs; see table1_*_coding.csv / table1_*_noncoding.csv slices"
        ),
        "outputs": {
            "table1": str(t1_path),
            "table2": str(t2_path),
            "table3": str(t3_path),
        },
    }
    (out_dir / "univariate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Table 1: CADD + CD34+ (coding∪noncoding) ===", flush=True)
    show = t1.copy()
    show["OR/SD"] = show.apply(
        lambda r: f"{r['or_per_sd']:.3f} [{r['or_ci_lo']:.3f}-{r['or_ci_hi']:.3f}]",
        axis=1,
    )
    print(
        show[
            ["feature", "n", "n_somatic", "somatic_rate", "OR/SD", "p_value", "q_value_bh"]
        ].to_string(index=False),
        flush=True,
    )
    print("\n=== Table 2 (region_class + annotated_effect) ===", flush=True)
    print(t2.to_string(index=False), flush=True)
    print("\n=== Table 3 SBS6 ===", flush=True)
    print(t3.to_string(index=False), flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
