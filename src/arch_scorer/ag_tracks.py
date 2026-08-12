"""AlphaGenome API CD34+ track scores for api+baseline model."""
from __future__ import annotations

import os
from typing import Any

from .genes import API_TRACK_RAW


SCORER_NAMES = [
    "RNA_SEQ",
    "DNASE",
    "CHIP_HISTONE",
    "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
]
ONTOLOGY = {"CL:0001059"}


def score_api_tracks(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    gene: str,
    *,
    api_key: str | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Return wide raw CD34+ track features for one SNV (hosted AG API)."""
    from alphagenome.data import genome
    from alphagenome.models import dna_client
    from alphagenome.models import variant_scorers as vs

    # Local tidy→wide helpers mirrored from sm_ch_finetune scorers (minimal)
    from alphagenome.models.dna_client import SEQUENCE_LENGTH_1MB

    from .env import load_dotenv

    load_dotenv()

    key = api_key or os.environ.get("ALPHA_GENOME_API_KEY") or os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set ALPHA_GENOME_API_KEY in .env (see .env.example), or pass --api-key."
        )
    client = dna_client.create(key)
    chrom_chr = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    variant = genome.Variant(
        chromosome=chrom_chr,
        position=int(pos),
        reference_bases=str(ref).upper(),
        alternate_bases=str(alt).upper(),
    )
    half = SEQUENCE_LENGTH_1MB // 2
    start = max(0, int(pos) - half - 1)
    interval = genome.Interval(chrom_chr, start, start + SEQUENCE_LENGTH_1MB)

    scorers = [vs.RECOMMENDED_VARIANT_SCORERS[n] for n in SCORER_NAMES]
    adatas = client.score_variant(interval, variant, scorers)
    tidy = vs.tidy_scores([adatas], match_gene_strand=True)
    if tidy is None or tidy.empty:
        raise RuntimeError("AlphaGenome API returned empty scores")

    # Filter CD34+; gene-mask rows to annotated gene
    df = tidy[tidy["ontology_curie"].isin(ONTOLOGY)].copy()
    has_gene = df["gene_name"].notna() & (df["gene_name"] != "")
    gene_rows = df[has_gene & (df["gene_name"] == gene)]
    other_rows = df[~has_gene]
    filt = pd_concat(gene_rows, other_rows)

    feats: dict[str, float] = {}
    for _, r in filt.iterrows():
        track = _safe(r.get("track_name", ""))
        out_t = str(r["output_type"])
        tag = _scorer_tag(str(r.get("variant_scorer", "")))
        base = f"{out_t}__{track}__{tag}"
        feats[base] = float(r["raw_score"])

    # Prefer exact training column names when provided
    names = feature_names or list(API_TRACK_RAW.values())
    out: dict[str, float] = {}
    for name in names:
        if name in feats:
            out[name] = feats[name]
            continue
        # fallback: match by short key
        hit = None
        for short, raw in API_TRACK_RAW.items():
            if raw == name or name.startswith(short + "__") or short in name:
                # try any feat containing distinctive parts
                pass
        # substring match against returned keys
        for k, v in feats.items():
            if _compat(name, k):
                hit = v
                break
        if hit is None:
            raise KeyError(f"Missing API feature {name!r}; got {sorted(feats)[:12]}…")
        out[name] = float(hit)
    return out


def pd_concat(a, b):
    import pandas as pd

    if a.empty:
        return b.copy()
    if b.empty:
        return a.copy()
    return pd.concat([a, b], ignore_index=True)


def _safe(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


def _scorer_tag(scorer_str: str) -> str:
    for prefix, tag in (
        ("GeneMaskLFCScorer", "GeneMaskLFC"),
        ("GeneMaskSplicingScorer", "GeneMaskSplicing"),
        ("SpliceJunctionScorer", "SpliceJunction"),
        ("CenterMaskScorer", "CenterMask"),
    ):
        if prefix in scorer_str:
            return tag
    return _safe(scorer_str)[:40]


def _compat(want: str, have: str) -> bool:
    return want == have or want.replace("__", "_") == have.replace("__", "_")
