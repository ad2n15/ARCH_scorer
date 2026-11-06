#!/usr/bin/env python3
"""Rebuild pooled MANE-noncoding cohort: WES∪WGS germline + WES∪WGS paper MW ∪ COSMIC.

Unified carrier rule (paper: “more than five”): ab_n_carriers_hist > 5.

Germline = WGS ∪ WES with ab_n>5 AND Mann–Whitney FDR q >= 0.05
  (not significantly AB-skewed vs rs2276599).
Somatic  = WGS paper MW ∪ WES paper MW
  ∪ COSMIC (NCV ∪ MutantCensus) in MANE noncoding with
    n_haem≥1 AND n_haem_confirmed≥1
  ∪ IARC TP53 haemato MANE-nc.
Overlap rule: germline ∩ (COSMIC ∪ IARC) dropped from BOTH sides.
AS_pab excluded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

WGS_GERM = (
    PROJECT
    / "data/arch_mane_noncoding_pooled_sm"
    / "arch_mane_noncoding_germline_abn_ge5.csv"
)
# Prefer the pre-WES germline snapshot if present; else rebuild from WGS candidates.
WGS_GERM_SNAP = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_ab_minABn2"
    / "mane_noncoding_gnomad_germline_abn_ge5.csv"
)
WGS_CAND = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_ab"
    / "mane_noncoding_gnomad_all_candidates_with_metrics.csv"
)
WGS_PAPER = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_ab_minABn2"
    / "mane_noncoding_gnomad_somatic_paper_mw_abn_gt5.csv"
)
WGS_MW_METRICS = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_ab_minABn2"
    / "mane_noncoding_gnomad_all_candidates_paper_mw_metrics.csv"
)
WES_GERM = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_exomes_v4"
    / "arch_mane_noncoding_germline_exomes_v4_abn_ge5.csv"
)
WES_PAPER = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_exomes_v4"
    / "paper_mw"
    / "mane_noncoding_wes_somatic_paper_mw_abn_gt5.csv"
)
WES_MW_METRICS = (
    PROJECT
    / "data/arch_mane_noncoding_gnomad_exomes_v4"
    / "paper_mw"
    / "mane_noncoding_wes_all_candidates_paper_mw_metrics.csv"
)
COS_NCV_PATH = (
    PROJECT
    / "data/arch_mane_noncoding"
    / "arch_mane_noncoding_somatic_cosmic_ncv_haem_min1.csv"
)
COS_MC_PATH = (
    PROJECT
    / "data/arch_mane_noncoding"
    / "arch_mane_noncoding_somatic_cosmic_mutantcensus_haem_min1_ge1conf.csv"
)
IARC_TP53_PATH = (
    PROJECT
    / "data/arch_mane_noncoding"
    / "arch_mane_noncoding_somatic_iarc_tp53_hemato.csv"
)
# Keep only alleles with ≥1 Confirmed somatic among haem observations.
COS_MIN_HAEM_CONFIRMED = 1

COLS = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "label",
    "source",
    "sm_source",
    "mane_region",
    "af",
    "ab_median",
    "ab_n_carriers_hist",
    "AS_pab_max",
]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["variant_id"] = out["variant_id"].astype(str)
    out["chrom"] = out["chrom"].astype(str).str.replace("^chr", "", regex=True)
    return out


def _row_from(r: pd.Series, label: str, source: str, sm_source: str) -> dict:
    return {
        "variant_id": str(r["variant_id"]),
        "chrom": str(r["chrom"]).replace("chr", ""),
        "pos": int(r["pos"]),
        "ref": str(r["ref"]),
        "alt": str(r["alt"]),
        "gene": r["gene"],
        "label": label,
        "source": source,
        "sm_source": sm_source,
        "mane_region": r["mane_region"] if "mane_region" in r.index else None,
        "af": r["af"] if "af" in r.index else None,
        "ab_median": r["ab_median"] if "ab_median" in r.index else None,
        "ab_n_carriers_hist": r["ab_n_carriers_hist"]
        if "ab_n_carriers_hist" in r.index
        else None,
        "AS_pab_max": r["AS_pab_max"] if "AS_pab_max" in r.index else None,
    }


def _load_wgs_germline() -> pd.DataFrame:
    """WGS germline before this WES merge (avoid circular read of updated pool)."""
    # Rebuild from WGS candidates; ab_n>5 matches paper / MW.
    if WGS_CAND.is_file():
        cand = _norm(pd.read_csv(WGS_CAND))
        germ = cand.loc[cand.ab_n_carriers_hist > 5].copy()
        germ["assay"] = "wgs"
        return germ
    # fallback: previous pool germline (may already be mixed — last resort)
    g = _norm(pd.read_csv(WGS_GERM))
    g = g.loc[g.ab_n_carriers_hist > 5].copy()
    g["assay"] = "wgs"
    return g


def main() -> None:
    pool = PROJECT / "data/arch_mane_noncoding_pooled_sm"
    pilot_root = PROJECT / "data/pilot_arch_mane_noncoding_pooled_sm"
    by_gene_pilot = PROJECT / "data/pilot_arch_mane_noncoding_pooled_sm_by_gene"
    by_gene_csv = pool / "by_gene"

    wgs_paper = _norm(pd.read_csv(WGS_PAPER))
    wes_paper = _norm(pd.read_csv(WES_PAPER))
    # Enforce ab_n>5 on paper MW inputs (should already hold)
    wgs_paper = wgs_paper.loc[wgs_paper.ab_n_carriers_hist > 5].copy()
    wes_paper = wes_paper.loc[wes_paper.ab_n_carriers_hist > 5].copy()
    wes_germ = _norm(pd.read_csv(WES_GERM))
    wes_germ = wes_germ.loc[wes_germ.ab_n_carriers_hist > 5].copy()

    cos_ncv = _norm(pd.read_csv(COS_NCV_PATH))
    if "n_haem_confirmed" not in cos_ncv.columns:
        raise SystemExit(f"Missing n_haem_confirmed in {COS_NCV_PATH}")
    cos_ncv = cos_ncv.loc[
        cos_ncv["n_haem_confirmed"].astype(float) >= COS_MIN_HAEM_CONFIRMED
    ].copy()
    cos_ncv["cosmic_catalog"] = "ncv"

    if not COS_MC_PATH.is_file():
        raise SystemExit(
            f"Missing MutantCensus extract: {COS_MC_PATH}\n"
            "Run: python3 scripts/81_extract_cosmic_mutantcensus_mane_noncoding.py"
        )
    cos_mc = _norm(pd.read_csv(COS_MC_PATH))
    cos_mc = cos_mc.loc[
        cos_mc["n_haem_confirmed"].astype(float) >= COS_MIN_HAEM_CONFIRMED
    ].copy()
    cos_mc["cosmic_catalog"] = "mutantcensus"

    # Union Cosmic catalogues; keep max haem counts per allele
    cos = pd.concat([cos_ncv, cos_mc], ignore_index=True, sort=False)
    cos_ncv_ids = set(cos_ncv.variant_id)
    cos_mc_ids = set(cos_mc.variant_id)
    agg = {
        "chrom": "first",
        "pos": "first",
        "ref": "first",
        "alt": "first",
        "gene": "first",
        "mane_region": "first",
        "n_haem": "max",
        "n_haem_confirmed": "max",
    }
    for c in ("af", "ab_median", "ab_n_carriers_hist", "AS_pab_max"):
        if c in cos.columns:
            agg[c] = "first"
    cos = cos.groupby("variant_id", as_index=False).agg(agg)

    if not IARC_TP53_PATH.is_file():
        raise SystemExit(
            f"Missing IARC TP53 MANE-nc extract: {IARC_TP53_PATH}\n"
            "Run: python3 scripts/82_extract_iarc_tp53_hemato_mane_noncoding.py"
        )
    iarc = _norm(pd.read_csv(IARC_TP53_PATH))
    iarc_ids = set(iarc.variant_id)

    wgs_germ = _load_wgs_germline()

    print(f"WGS paper MW somatic n={len(wgs_paper)}", flush=True)
    print(f"WES paper MW somatic n={len(wes_paper)}", flush=True)
    print(f"COSMIC NCV confirmed n={len(cos_ncv_ids)}", flush=True)
    print(f"COSMIC MutantCensus confirmed n={len(cos_mc_ids)}", flush=True)
    print(
        f"COSMIC union (NCV∪MC, confirmed) n={len(cos)} "
        f"overlap={len(cos_ncv_ids & cos_mc_ids)}",
        flush=True,
    )
    print(f"IARC TP53 haemato MANE-nc n={len(iarc_ids)}", flush=True)
    print(f"WGS germline candidates (ab_n>5) n={len(wgs_germ)}", flush=True)
    print(f"WES germline (ab_n>5) n={len(wes_germ)}", flush=True)

    wgs_p_ids = set(wgs_paper.variant_id)
    wes_p_ids = set(wes_paper.variant_id)
    cos_ids = set(cos.variant_id)
    # Build somatic including catalogues; paper MW / COSMIC / IARC
    som_ids = wgs_p_ids | wes_p_ids | cos_ids | iarc_ids

    # Meta lookup preference: WGS paper > WES paper > WGS germ > WES germ > Cosmic > IARC
    meta_df = (
        pd.concat(
            [
                iarc.assign(_pri=-1),
                cos.assign(_pri=0),
                wes_germ.assign(_pri=1),
                wgs_germ.assign(_pri=2),
                wes_paper.assign(_pri=3),
                wgs_paper.assign(_pri=4),
            ],
            ignore_index=True,
            sort=False,
        )
        .sort_values("_pri")
        .drop_duplicates("variant_id", keep="last")
        .set_index("variant_id", drop=False)
    )

    som_rows = []
    for vid in som_ids:
        tags = []
        if vid in wgs_p_ids:
            tags.append("paper_mw_wgs_abn_gt5")
        if vid in wes_p_ids:
            tags.append("paper_mw_wes_abn_gt5")
        if vid in cos_ncv_ids:
            tags.append("cosmic_ncv_haem_ge1_conf")
        if vid in cos_mc_ids:
            tags.append("cosmic_mutantcensus_haem_ge1_conf")
        if vid in iarc_ids:
            tags.append("iarc_hemato_mane_nc")
        src = "+".join(tags)
        som_rows.append(_row_from(meta_df.loc[vid], "somatic", src, src))
    som = pd.DataFrame(som_rows)

    # Germline = WGS∪WES ab_n>5 (do NOT yet strip somatic catalogues — overlap
    # with COSMIC/IARC is dropped from BOTH sides below). Paper MW alleles have
    # mw_fdr_q<0.05 so they cannot pass the germline MW filter.
    wgs_g = wgs_germ.drop_duplicates("variant_id")
    wes_g = wes_germ.drop_duplicates("variant_id")
    wgs_g_ids = set(wgs_g.variant_id)
    wes_g_ids = set(wes_g.variant_id)

    germ_rows = []
    for _, r in wgs_g.iterrows():
        src = (
            "gnomad_v4_genomes+exomes"
            if r.variant_id in wes_g_ids
            else "gnomad_v4_genomes"
        )
        germ_rows.append(_row_from(r, "germline", src, ""))
    for _, r in wes_g.iterrows():
        if r.variant_id in wgs_g_ids:
            continue
        germ_rows.append(_row_from(r, "germline", "gnomad_v4_exomes", ""))
    germ = pd.DataFrame(germ_rows)
    n_germ_pre_mw = len(germ)

    # Germline MW filter: require tested + FDR q >= 0.05 (prefer WGS q if both)
    wgs_q = pd.read_csv(
        WGS_MW_METRICS, usecols=["variant_id", "mw_fdr_q", "hedges_g_vs_rs2276599"]
    )
    wes_q = pd.read_csv(
        WES_MW_METRICS, usecols=["variant_id", "mw_fdr_q", "hedges_g_vs_rs2276599"]
    )
    wgs_q["variant_id"] = wgs_q["variant_id"].astype(str)
    wes_q["variant_id"] = wes_q["variant_id"].astype(str)
    mw_q = (
        pd.concat([wgs_q.assign(_pri=1), wes_q.assign(_pri=0)], ignore_index=True)
        .sort_values("_pri")
        .drop_duplicates("variant_id", keep="last")
    )
    germ = germ.merge(
        mw_q[["variant_id", "mw_fdr_q", "hedges_g_vs_rs2276599"]],
        on="variant_id",
        how="left",
    )
    n_untested = int(germ["mw_fdr_q"].isna().sum())
    n_q_lt = int(((germ["mw_fdr_q"].notna()) & (germ["mw_fdr_q"] < 0.05)).sum())
    germ = germ.loc[germ["mw_fdr_q"].notna() & (germ["mw_fdr_q"] >= 0.05)].copy()
    n_germ_mw_keep = int(len(germ))
    print(
        f"Germline MW filter: pre={n_germ_pre_mw} untested_drop={n_untested} "
        f"q<0.05_drop={n_q_lt} keep_q>=0.05={n_germ_mw_keep}",
        flush=True,
    )
    germ = germ.drop(columns=["mw_fdr_q", "hedges_g_vs_rs2276599"], errors="ignore")

    # Drop germline ∩ (COSMIC ∪ IARC) from BOTH sides (ambiguous; not relabeled).
    ext_ids = cos_ids | iarc_ids
    ambig = set(germ.variant_id) & ext_ids
    n_ambig_cos = len(ambig & cos_ids)
    n_ambig_iarc = len(ambig & iarc_ids)
    if ambig:
        germ = germ.loc[~germ.variant_id.isin(ambig)].copy()
        som = som.loc[~som.variant_id.isin(ambig)].copy()
    print(
        f"Dropped germline∩(COSMIC∪IARC) both sides: n={len(ambig)} "
        f"(cosmic={n_ambig_cos}, iarc={n_ambig_iarc})",
        flush=True,
    )

    # Paper MW stays somatic. If WES/WGS MW q disagree (paper on one assay,
    # q≥0.05 on the preferred assay), remove from germline only — do not
    # drop paper-MW alleles from the somatic set.
    paper_ids = wgs_p_ids | wes_p_ids
    n_paper_in_germ = int(germ.variant_id.isin(paper_ids).sum())
    if n_paper_in_germ:
        germ = germ.loc[~germ.variant_id.isin(paper_ids)].copy()
        print(
            f"Removed paper-MW alleles from germline only: n={n_paper_in_germ}",
            flush=True,
        )

    both = set(germ.variant_id) & set(som.variant_id)
    if both:
        raise SystemExit(f"Residual germ∩som after overlap rules: n={len(both)}")

    germ = germ[COLS]
    som = som[COLS]
    comb = pd.concat([germ, som], ignore_index=True)

    cos_nh = cos[["variant_id", "n_haem"]].drop_duplicates("variant_id")
    comb = comb.merge(cos_nh, on="variant_id", how="left")
    comb["haem_ge3"] = (comb["n_haem"].fillna(0).astype(float) >= 3).astype(int)
    germ = comb.loc[comb.label == "germline", COLS + ["n_haem", "haem_ge3"]].copy()
    som = comb.loc[comb.label == "somatic", COLS + ["n_haem", "haem_ge3"]].copy()

    pool.mkdir(parents=True, exist_ok=True)
    germ.to_csv(pool / "arch_mane_noncoding_germline_abn_ge5.csv", index=False)
    som.to_csv(pool / "arch_mane_noncoding_somatic_pooled.csv", index=False)
    comb.to_csv(pool / "arch_mane_noncoding_germline_plus_somatic_pooled.csv", index=False)
    with open(pool / "arch_mane_noncoding_pooled.avinput", "w") as f:
        for r in comb.itertuples(index=False):
            c = str(r.chrom).replace("chr", "")
            f.write(f"{c}\t{int(r.pos)}\t{int(r.pos)}\t{r.ref}\t{r.alt}\n")

    pdf = comb.copy()
    pdf["chrom"] = ["chr" + str(c).replace("chr", "") for c in pdf["chrom"]]
    pdf["label_str"] = pdf["label"].astype(str)
    pdf["label"] = (pdf["label_str"] == "somatic").astype(np.int64)
    pdf["split"] = "test"
    pilot_cols = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "variant_id",
        "gene",
        "label",
        "label_str",
        "split",
        "mane_region",
        "source",
        "n_haem",
        "haem_ge3",
    ]
    pilot_root.mkdir(parents=True, exist_ok=True)
    pdf[pilot_cols].to_parquet(pilot_root / "test.parquet", index=False)

    by_gene_pilot.mkdir(parents=True, exist_ok=True)
    by_gene_csv.mkdir(parents=True, exist_ok=True)
    for gene, g in comb.groupby("gene"):
        d = by_gene_pilot / gene
        d.mkdir(exist_ok=True)
        gp = pdf.loc[pdf.gene == gene]
        gp[pilot_cols].to_parquet(d / "test.parquet", index=False)
        g.to_csv(by_gene_csv / f"{gene}.csv", index=False)

    by_gene = []
    for gene, g in comb.groupby("gene"):
        gg = g[g.label == "germline"]
        ss = g[g.label == "somatic"]
        row = {
            "gene": gene,
            "germline": int(len(gg)),
            "somatic_total": int(len(ss)),
            "somatic_paper_mw_wgs": int(
                ss["sm_source"].astype(str).str.contains("paper_mw_wgs", case=False).sum()
            ),
            "somatic_paper_mw_wes": int(
                ss["sm_source"].astype(str).str.contains("paper_mw_wes", case=False).sum()
            ),
            "somatic_COSMIC": int(
                ss["sm_source"].astype(str).str.contains("cosmic", case=False).sum()
            ),
            "somatic_IARC": int(
                ss["sm_source"].astype(str).str.contains("iarc_hemato", case=False).sum()
            ),
            "total": int(len(g)),
        }
        for r in ["intron", "UTR3", "UTR5", "core_promoter_200"]:
            row[f"g_{r}"] = int((gg.mane_region == r).sum())
            row[f"s_{r}"] = int((ss.mane_region == r).sum())
        by_gene.append(row)

    summary = {
        "design": (
            "germline=WGS∪WES (ab_n>5 AND mw_fdr_q>=0.05); "
            "somatic=WGS paper MW ∪ WES paper MW ∪ "
            "COSMIC (NCV∪MutantCensus) MANE-nc haem≥1 Confirmed≥1 ∪ "
            "IARC TP53 haemato MANE-nc; "
            "drop germline∩(COSMIC∪IARC) from BOTH sides; AS_pab excluded"
        ),
        "ab_n_rule": "ab_n_carriers_hist > 5 everywhere (paper: more than five individuals)",
        "germline_mw_rule": "mw_fdr_q >= 0.05 (must be MW-tested; untested dropped)",
        "overlap_rule": "drop germline ∩ (COSMIC ∪ IARC) from both sides (not relabeled)",
        "cosmic_rule": (
            f"NCV∪MutantCensus in MANE noncoding; "
            f"n_haem>=1 AND n_haem_confirmed>={COS_MIN_HAEM_CONFIRMED}"
        ),
        "iarc_rule": (
            "IARC R21 TP53 Short_topo=HEMATOP. SYSTEM tumor SNVs in "
            "MANE Select intron+UTR+prom200"
        ),
        "paper_rule": (
            "mw_fdr_q < 0.05 AND hedges_g_vs_rs2276599 >= 0.8 "
            "AND ab_n_carriers_hist > 5"
        ),
        "germline_mw_filter_counts": {
            "pre_mw_filter": n_germ_pre_mw,
            "dropped_untested": n_untested,
            "dropped_q_lt_0.05": n_q_lt,
            "kept_q_ge_0.05": n_germ_mw_keep,
        },
        "overlap_dropped_both_sides": {
            "n": int(len(ambig)),
            "cosmic": int(n_ambig_cos),
            "iarc": int(n_ambig_iarc),
        },
        "paper_mw_removed_from_germline_only": int(n_paper_in_germ),
        "n_germline": int(len(germ)),
        "n_somatic": int(len(som)),
        "n_combined": int(len(comb)),
        "germline_parts": {
            "wgs_only": int((germ.source == "gnomad_v4_genomes").sum()),
            "wes_only": int((germ.source == "gnomad_v4_exomes").sum()),
            "both": int((germ.source == "gnomad_v4_genomes+exomes").sum()),
        },
        "somatic_parts": {
            "paper_mw_wgs_abn_gt5": len(wgs_p_ids),
            "paper_mw_wes_abn_gt5": len(wes_p_ids),
            "overlap_wgs_wes_paper_mw": len(wgs_p_ids & wes_p_ids),
            "cosmic_ncv_confirmed": len(cos_ncv_ids),
            "cosmic_mutantcensus_confirmed": len(cos_mc_ids),
            "cosmic_overlap_ncv_mc": len(cos_ncv_ids & cos_mc_ids),
            "cosmic_union": len(cos_ids),
            "iarc_tp53_hemato_mane_nc": len(iarc_ids),
            "overlap_iarc_cosmic": len(iarc_ids & cos_ids),
            "overlap_iarc_paper": len(iarc_ids & (wgs_p_ids | wes_p_ids)),
            "overlap_any_paper_cosmic": len((wgs_p_ids | wes_p_ids) & cos_ids),
            "union": int(len(som)),
            "haem_ge3": int(som["haem_ge3"].sum()),
        },
        "sm_source_counts": {
            str(k): int(v) for k, v in som["sm_source"].value_counts().items()
        },
        "by_gene": by_gene,
        "inputs": {
            "wgs_paper": str(WGS_PAPER),
            "wes_paper": str(WES_PAPER),
            "wes_germline": str(WES_GERM),
            "wgs_candidates": str(WGS_CAND),
            "cosmic_ncv": str(COS_NCV_PATH),
            "cosmic_mutantcensus": str(COS_MC_PATH),
            "iarc_tp53_hemato": str(IARC_TP53_PATH),
        },
    }
    (pool / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "by_gene"}, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
