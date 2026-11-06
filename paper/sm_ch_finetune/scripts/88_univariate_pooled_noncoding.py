#!/usr/bin/env python3
"""Univariate germline-vs-somatic analyses for ARCH MANE noncoding pooled cohort.

Mirrors gnomAD v2.1 / GVS publication tables:

  Table 1 — AlphaGenome CD34+ continuous features (+ CADD): OR per 1 SD
  Table 2 — Annotated effects (MANE region + Ellingford intronic±2 splicing): Fisher one-vs-rest
  Table 3 — Pyrimidine-referenced SBS6 substitutions: Fisher one-vs-rest

Tables 2–3 columns: feature, n, n_somatic, somatic_rate, or_vs_rest, or_ci_lo, or_ci_hi, p_value, q_value_bh
Table 1 columns:    feature, n, n_somatic, somatic_rate, or_per_sd, or_ci_lo, or_ci_hi, p_value, q_value_bh
                    (+ germline/somatic medians)

Example:
  cd paper/sm_ch_finetune
  python3 scripts/88_univariate_pooled_noncoding.py
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))
_pub = import_module("24_ch_genes_gvs_publication_analysis")

from data.mane_ellingford import intronic_canonical_splice, load_mane_cds_exons

POOL = PROJECT / "data/arch_mane_noncoding_pooled_sm"
COHORT = POOL / "arch_mane_noncoding_germline_plus_somatic_pooled.csv"
ANNOVAR_DIR = POOL / "annovar_by_gene"
CADD_DIR = POOL / "cadd_by_gene"
SCORES_DIR = PROJECT / "results/track_scores_arch_mane_noncoding_pooled_sm_by_gene"
OUT_DIR = PROJECT / "results/arch_mane_noncoding_pooled_sm_univariate"
GTF = PROJECT / "data/reference/gencode.v46.annotation.gtf"

REGION_MAP = {
    "intron": "intron",
    "UTR5": "UTR5",
    "UTR3": "UTR3",
    "core_promoter_200": "promoter_200_upstream",
}
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_SBS6 = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]


def _bh_fdr(p_values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    out[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return out


def _key(df: pd.DataFrame, chrom: str, pos: str, ref: str, alt: str) -> pd.Series:
    return (
        df[chrom].astype(str).str.replace("^chr", "", regex=True)
        + ":"
        + df[pos].astype(int).astype(str)
        + ":"
        + df[ref].astype(str).str.upper()
        + ":"
        + df[alt].astype(str).str.upper()
    )


def _to_sbs6(ref: str, alt: str) -> str | None:
    ref, alt = str(ref).upper(), str(alt).upper()
    if len(ref) != 1 or len(alt) != 1 or ref == alt:
        return None
    if ref not in "ACGT" or alt not in "ACGT":
        return None
    if ref in ("A", "G"):
        ref, alt = _COMPLEMENT[ref], _COMPLEMENT[alt]
    sub = f"{ref}>{alt}"
    return sub if sub in _SBS6 else None


def _one_vs_rest_fisher(cat: pd.Series, y: np.ndarray, *, min_n: int) -> pd.DataFrame:
    cat = cat.fillna("unknown").astype(str)
    rows = []
    for level, n in cat.value_counts().items():
        if int(n) < min_n:
            continue
        in_level = cat.eq(level).to_numpy()
        a = int(y[in_level].sum())
        b = int(in_level.sum() - a)
        c = int(y[~in_level].sum())
        d = int((~in_level).sum() - c)
        odds, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        aa, bb, cc, dd = a, b, c, d
        if 0 in (a, b, c, d):
            aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            odds = (aa * dd) / (bb * cc)
        se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        log_or = np.log(odds)
        rows.append(
            {
                "feature": level,
                "n": int(n),
                "n_somatic": a,
                "somatic_rate": float(y[in_level].mean()),
                "or_vs_rest": float(odds),
                "or_ci_lo": float(np.exp(log_or - 1.96 * se)),
                "or_ci_hi": float(np.exp(log_or + 1.96 * se)),
                "p_value": float(p_value),
            }
        )
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    if len(out):
        out["q_value_bh"] = _bh_fdr(out["p_value"])
    return out


def _continuous_or_per_sd(y: np.ndarray, x: np.ndarray, name: str) -> dict:
    row = _pub._univariate_continuous(y, x, name)
    mask = np.isfinite(x)
    y2, x2 = y[mask], x[mask]
    germ = x2[y2 == 0]
    som = x2[y2 == 1]
    return {
        "feature": name,
        "n": int(row.get("n") or 0),
        "n_somatic": int(row.get("n_pos") or 0),
        "somatic_rate": float((row.get("n_pos") or 0) / row["n"]) if row.get("n") else None,
        "or_per_sd": row.get("or_per_sd"),
        "or_ci_lo": row.get("or_ci_lo"),
        "or_ci_hi": row.get("or_ci_hi"),
        "p_value": row.get("p_value"),
        "auroc": row.get("auroc_raw"),
        "germ_median": float(np.median(germ)) if len(germ) else None,
        "germ_min": float(np.min(germ)) if len(germ) else None,
        "germ_max": float(np.max(germ)) if len(germ) else None,
        "som_median": float(np.median(som)) if len(som) else None,
        "som_min": float(np.min(som)) if len(som) else None,
        "som_max": float(np.max(som)) if len(som) else None,
    }


def _load_cohort() -> pd.DataFrame:
    df = pd.read_csv(COHORT, low_memory=False)
    df["variant_id"] = df["variant_id"].astype(str)
    df["chrom"] = df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["y"] = (df["label"].astype(str) == "somatic").astype(int)
    df["variant_key"] = _key(df, "chrom", "pos", "ref", "alt")
    return df


def _attach_annovar(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for gene, part in df.groupby("gene", sort=True):
        path = ANNOVAR_DIR / f"{gene}_annovar.hg38_multianno.csv"
        if not path.is_file():
            raise SystemExit(f"Missing ANNOVAR: {path}")
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
        raise SystemExit(f"Unmapped MANE regions:\n{bad}")
    return out


def _attach_cadd(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for gene in sorted(df["gene"].unique()):
        local = CADD_DIR / f"{gene}_cadd_phred_local.csv"
        api = CADD_DIR / f"{gene}_cadd_phred_api.csv"
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


def _attach_api(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (joined_df_for_scored_genes, feature_cols, scored_genes)."""
    scored_genes = []
    parts = []
    for gene_dir in sorted(SCORES_DIR.iterdir() if SCORES_DIR.is_dir() else []):
        if not gene_dir.is_dir():
            continue
        gene = gene_dir.name
        # skip legacy subset folders
        if gene.startswith("TP53_"):
            continue
        feat = gene_dir / "test_features_wide.parquet"
        if not feat.is_file():
            continue
        api = pd.read_parquet(feat)
        api["variant_id"] = api["variant_id"].astype(str)
        api = api.drop_duplicates("variant_id")
        scored_genes.append(gene)
        parts.append(api)
    if not parts:
        return df.iloc[0:0].copy(), [], []
    api_all = pd.concat(parts, ignore_index=True).drop_duplicates("variant_id")
    feat_cols = [c for c in api_all.columns if c in _pub.SHORT]
    keep = ["variant_id"] + feat_cols
    joined = df.merge(api_all[keep], on="variant_id", how="inner")
    return joined, feat_cols, scored_genes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=30, help="min n for categorical levels")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cohort …", flush=True)
    df = _attach_cadd(_attach_annovar(_load_cohort()))
    df["sbs6"] = [
        _to_sbs6(r, a) for r, a in zip(df["ref"].astype(str), df["alt"].astype(str))
    ]
    y = df["y"].to_numpy(int)
    print(
        f"Cohort n={len(df):,}  somatic={int(y.sum()):,}  "
        f"with CADD={int(df['cadd_phred'].notna().sum()):,}",
        flush=True,
    )

    # ---- Table 1: AlphaGenome CD34+ continuous (+ CADD) ----
    api_df, feat_cols, scored_genes = _attach_api(df)
    print(
        f"API-scored genes ({len(scored_genes)}): {', '.join(scored_genes) or 'none'}",
        flush=True,
    )

    def _table1_continuous(sub: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        rows: list[dict] = []
        if not len(sub) or not feat_cols:
            return pd.DataFrame(rows)
        y1 = sub["y"].to_numpy(int)
        if sub["cadd_phred"].notna().sum() >= 50:
            rows.append(
                _continuous_or_per_sd(
                    y1, sub["cadd_phred"].to_numpy(float), "cadd_phred"
                )
            )
        for col in feat_cols:
            rows.append(
                _continuous_or_per_sd(
                    y1, sub[col].to_numpy(float), _pub.SHORT.get(col, col)
                )
            )
        out = pd.DataFrame(rows)
        if len(out):
            out = out.sort_values("p_value").reset_index(drop=True)
            out["q_value_bh"] = _bh_fdr(out["p_value"])
        return out

    t1 = _table1_continuous(api_df, feat_cols)
    t1_path = out_dir / "table1_alphagenome_cd34_continuous.csv"
    t1.to_csv(t1_path, index=False)

    # Table 1 excluding Ellingford splicing-annotated alleles
    n_splice_api = int((api_df["annotated_effect"] == "splicing").sum()) if len(api_df) else 0
    api_nosplice = (
        api_df.loc[api_df["annotated_effect"] != "splicing"].copy()
        if len(api_df)
        else api_df
    )
    t1_nosplice = _table1_continuous(api_nosplice, feat_cols)
    t1_nosplice_path = out_dir / "table1_alphagenome_cd34_continuous_nosplice.csv"
    t1_nosplice.to_csv(t1_nosplice_path, index=False)

    # Side-by-side combined table for reporting
    def _write_all_vs_nosplice(tab_all: pd.DataFrame, tab_ns: pd.DataFrame, path: Path) -> None:
        if len(tab_all) and len(tab_ns):
            a = tab_all.rename(
                columns={c: f"{c}_all" for c in tab_all.columns if c != "feature"}
            )
            b = tab_ns.rename(
                columns={c: f"{c}_nosplice" for c in tab_ns.columns if c != "feature"}
            )
            a.merge(b, on="feature", how="outer").to_csv(path, index=False)
        else:
            pd.DataFrame().to_csv(path, index=False)

    if len(t1) and len(t1_nosplice):
        t1_combined_path = (
            out_dir / "table1_alphagenome_cd34_continuous_all_vs_nosplice.csv"
        )
        _write_all_vs_nosplice(t1, t1_nosplice, t1_combined_path)
    else:
        t1_combined_path = (
            out_dir / "table1_alphagenome_cd34_continuous_all_vs_nosplice.csv"
        )
        pd.DataFrame().to_csv(t1_combined_path, index=False)

    # Gene-set split: LoF9 / TSG-like vs remaining ARCH genes (oncogene-enriched)
    LOF9 = [
        "ASXL1",
        "ASXL2",
        "CBL",
        "DNMT3A",
        "ETNK1",
        "PPM1D",
        "RUNX1",
        "TET2",
        "TP53",
    ]
    OTHER12 = sorted(set(scored_genes) - set(LOF9))
    gene_set_meta = {}
    for tag, genes in (("lof9", LOF9), ("other12", OTHER12)):
        sub_all = api_df.loc[api_df["gene"].isin(genes)].copy()
        sub_ns = sub_all.loc[sub_all["annotated_effect"] != "splicing"].copy()
        t_all = _table1_continuous(sub_all, feat_cols)
        t_ns = _table1_continuous(sub_ns, feat_cols)
        t_all.to_csv(out_dir / f"table1_alphagenome_cd34_continuous_{tag}.csv", index=False)
        t_ns.to_csv(
            out_dir / f"table1_alphagenome_cd34_continuous_{tag}_nosplice.csv",
            index=False,
        )
        comb = out_dir / f"table1_alphagenome_cd34_continuous_{tag}_all_vs_nosplice.csv"
        _write_all_vs_nosplice(t_all, t_ns, comb)
        gene_set_meta[tag] = {
            "genes": genes,
            "n_all": int(len(sub_all)),
            "n_somatic_all": int(sub_all["y"].sum()) if len(sub_all) else 0,
            "n_splice_excluded": int((sub_all["annotated_effect"] == "splicing").sum())
            if len(sub_all)
            else 0,
            "n_nosplice": int(len(sub_ns)),
            "n_somatic_nosplice": int(sub_ns["y"].sum()) if len(sub_ns) else 0,
        }
        print(
            f"\n=== Table 1 {tag} genes={genes} n={len(sub_all)} "
            f"som={int(sub_all['y'].sum()) if len(sub_all) else 0} ===",
            flush=True,
        )
        if len(t_all):
            show = t_all.copy()
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
        print(
            f"=== Table 1 {tag} nosplice n={len(sub_ns)} "
            f"som={int(sub_ns['y'].sum()) if len(sub_ns) else 0} ===",
            flush=True,
        )
        if len(t_ns):
            show = t_ns.copy()
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

    # ---- Table 2: annotated effects one-vs-rest ----
    t2 = _one_vs_rest_fisher(df["annotated_effect"], y, min_n=args.min_n)
    t2_path = out_dir / "table2_annotated_effects_one_vs_rest.csv"
    t2.to_csv(t2_path, index=False)

    # ---- Table 3: SBS6 one-vs-rest ----
    snv = df.loc[df["sbs6"].notna()].copy()
    y3 = snv["y"].to_numpy(int)
    t3 = _one_vs_rest_fisher(snv["sbs6"], y3, min_n=args.min_n)
    # keep SBS6 order in a companion rates table
    rates = (
        snv.groupby("sbs6")["y"]
        .agg(n="count", n_somatic="sum")
        .reindex(_SBS6)
        .reset_index()
        .rename(columns={"sbs6": "feature"})
    )
    rates["somatic_rate"] = rates["n_somatic"] / rates["n"]
    t3_path = out_dir / "table3_sbs6_one_vs_rest.csv"
    t3.to_csv(t3_path, index=False)
    rates.to_csv(out_dir / "table3_sbs6_rates.csv", index=False)

    summary = {
        "cohort": str(COHORT),
        "n_total": int(len(df)),
        "n_somatic": int(y.sum()),
        "n_with_cadd": int(df["cadd_phred"].notna().sum()),
        "table1_n_variants": int(len(api_df)),
        "table1_n_somatic": int(api_df["y"].sum()) if len(api_df) else 0,
        "table1_n_splicing_excluded": int(n_splice_api),
        "table1_nosplice_n_variants": int(len(api_nosplice)),
        "table1_nosplice_n_somatic": int(api_nosplice["y"].sum()) if len(api_nosplice) else 0,
        "table1_gene_sets": gene_set_meta,
        "table1_scored_genes": scored_genes,
        "table1_method": "logistic OR per 1 SD of continuous CD34+ AlphaGenome features (+ CADD)",
        "table1_nosplice_method": (
            "same as table1 after excluding Ellingford splicing-annotated alleles"
        ),
        "table2_method": (
            "Fisher exact one-vs-rest on annotated_effect "
            "(MANE region; Ellingford intronic±2 splicing)"
        ),
        "table3_method": "Fisher exact one-vs-rest on pyrimidine-referenced SBS6; non-SNVs dropped",
        "table3_n_snv": int(len(snv)),
        "table3_n_somatic_snv": int(y3.sum()),
        "table3_n_nonsnv_dropped": int(df["sbs6"].isna().sum()),
        "outputs": {
            "table1": str(t1_path),
            "table1_nosplice": str(t1_nosplice_path),
            "table1_all_vs_nosplice": str(t1_combined_path),
            "table2": str(t2_path),
            "table3": str(t3_path),
        },
    }
    (out_dir / "univariate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def _fmt_cat(tab: pd.DataFrame, title: str) -> None:
        print(f"\n=== {title} ===", flush=True)
        if not len(tab):
            print("(empty)", flush=True)
            return
        show = tab.copy()
        show["somatic_rate"] = show["somatic_rate"].map(lambda x: f"{100 * x:.2f}%")
        show["OR"] = show.apply(
            lambda r: f"{r['or_vs_rest']:.3f} [{r['or_ci_lo']:.3f}-{r['or_ci_hi']:.3f}]",
            axis=1,
        )
        cols = ["feature", "n", "n_somatic", "somatic_rate", "OR", "p_value", "q_value_bh"]
        print(show[cols].to_string(index=False), flush=True)

    print("\n=== Table 1: AlphaGenome CD34+ continuous (OR per 1 SD) ===", flush=True)
    if len(t1):
        show = t1.copy()
        show["OR/SD"] = show.apply(
            lambda r: (
                f"{r['or_per_sd']:.3f} [{r['or_ci_lo']:.3f}-{r['or_ci_hi']:.3f}]"
                if pd.notna(r["or_per_sd"])
                else "NA"
            ),
            axis=1,
        )
        print(
            show[
                [
                    "feature",
                    "n",
                    "n_somatic",
                    "somatic_rate",
                    "OR/SD",
                    "p_value",
                    "q_value_bh",
                ]
            ].to_string(index=False),
            flush=True,
        )
    else:
        print(
            "No API-scored genes yet. Score genes first, then re-run.",
            flush=True,
        )

    print(
        f"\n=== Table 1 nosplice (excluded {n_splice_api} Ellingford splicing alleles) ===",
        flush=True,
    )
    if len(t1_nosplice):
        show = t1_nosplice.copy()
        show["OR/SD"] = show.apply(
            lambda r: (
                f"{r['or_per_sd']:.3f} [{r['or_ci_lo']:.3f}-{r['or_ci_hi']:.3f}]"
                if pd.notna(r["or_per_sd"])
                else "NA"
            ),
            axis=1,
        )
        print(
            show[
                [
                    "feature",
                    "n",
                    "n_somatic",
                    "somatic_rate",
                    "OR/SD",
                    "p_value",
                    "q_value_bh",
                ]
            ].to_string(index=False),
            flush=True,
        )

    _fmt_cat(t2, "Table 2: Annotated effects (one-vs-rest)")
    _fmt_cat(t3, "Table 3: SBS6 pyrimidine substitutions (one-vs-rest)")
    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
