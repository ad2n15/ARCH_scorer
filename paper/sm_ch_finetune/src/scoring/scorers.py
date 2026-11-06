"""Variant scorer selection and feature extraction."""
from __future__ import annotations

import re

import pandas as pd
from alphagenome.models import variant_scorers as vs

_GENE_CENTRIC_PREFIXES = (
    "GeneMaskLFCScorer",
    "GeneMaskActiveScorer",
    "GeneMaskSplicingScorer",
    "SpliceJunctionScorer",
    "PolyadenylationScorer",
)

_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def build_scorer_list(names: list[str], *, include_polyadenylation: bool = False) -> list:
    rec = vs.RECOMMENDED_VARIANT_SCORERS
    out = []
    for name in names:
        key = name.upper()
        if key not in rec:
            raise ValueError(f"Unknown scorer {name!r}. Keys: {sorted(rec.keys())}")
        out.append(rec[key])
    if include_polyadenylation and "POLYADENYLATION" not in {n.upper() for n in names}:
        out.append(rec["POLYADENYLATION"])
    return out


# GEM paper chromHMM proxy: CD34+ accessibility + histone marks (Weinstock et al. 2025).
GEM_CD34_ONTOLOGY = "CL_0001059"
GEM_CHROMATIN_OUTPUTS = ("DNASE", "ATAC", "CHIP_HISTONE")
GEM_HISTONE_MARKS = (
    "H3K27ac", "H3K4me3", "H3K27me3", "H3K36me3", "H3K9me3", "H3K4me1",
)


def gem_chromatin_feature_cols(
    columns: list[str],
    *,
    include_quantile: bool = False,
) -> list[str]:
    """Keep CD34+ CenterMask DNase/ATAC/histone features matching GEM chromatin pillar."""
    out = []
    for col in columns:
        if GEM_CD34_ONTOLOGY not in col:
            continue
        if not col.startswith(GEM_CHROMATIN_OUTPUTS):
            continue
        if "__CenterMask" not in col:
            continue
        if not include_quantile and col.endswith("__quantile"):
            continue
        if col.startswith("CHIP_HISTONE") and not any(mark in col for mark in GEM_HISTONE_MARKS):
            continue
        out.append(col)
    return sorted(out)


def _safe(s: str) -> str:
    return _SAFE.sub("_", str(s)).strip("_")


def _scorer_tag(scorer_str: str) -> str:
    for prefix in _GENE_CENTRIC_PREFIXES:
        if prefix in scorer_str:
            return prefix.replace("Scorer", "")
    if "CenterMaskScorer" in scorer_str:
        return "CenterMask"
    if "ContactMapScorer" in scorer_str:
        return "ContactMap"
    return _safe(scorer_str)[:40]


def filter_tidy_rows(
    tidy: pd.DataFrame,
    *,
    gene_name: str,
    ontology_curies: set[str],
) -> pd.DataFrame:
    """Keep CD34+ tracks; gene-mask rows restricted to the annotated gene."""
    if tidy.empty:
        return tidy
    df = tidy[tidy["ontology_curie"].isin(ontology_curies)].copy()
    if df.empty:
        return df
    has_gene = df["gene_name"].notna() & (df["gene_name"] != "")
    if gene_name and gene_name not in (".", "", "nan"):
        gene_rows = df[has_gene & (df["gene_name"] == gene_name)]
        other_rows = df[~has_gene]
        return pd.concat([gene_rows, other_rows], ignore_index=True)
    return df[~has_gene]


def tidy_to_wide(row_features: pd.DataFrame, *, include_quantile: bool = True) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, r in row_features.iterrows():
        base = f"{r['output_type']}__{_safe(r['track_name'])}__{_scorer_tag(r['variant_scorer'])}"
        out[base] = float(r["raw_score"])
        if include_quantile and "quantile_score" in r and pd.notna(r["quantile_score"]):
            out[f"{base}__quantile"] = float(r["quantile_score"])
    return out


def scores_to_features(
    score_adatas: list,
    *,
    gene_name: str,
    ontology_curies: set[str],
    match_gene_strand: bool = True,
    include_quantile: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return long tidy table and wide feature dict for one variant."""
    tidy = vs.tidy_scores([score_adatas], match_gene_strand=match_gene_strand)
    if tidy is None or tidy.empty:
        return pd.DataFrame(), {}
    filt = filter_tidy_rows(tidy, gene_name=gene_name, ontology_curies=ontology_curies)
    return filt, tidy_to_wide(filt, include_quantile=include_quantile)


def score_variant_batched(model, interval, variant, scorers: list, batch_size: int) -> list:
    """Run score_variant in scorer chunks to reduce peak GPU memory."""
    if batch_size <= 0 or batch_size >= len(scorers):
        return model.score_variant(interval, variant, scorers)
    out: list = []
    for i in range(0, len(scorers), batch_size):
        out.extend(model.score_variant(interval, variant, scorers[i : i + batch_size]))
    return out
