"""Lookup pilot variants in gnomAD exomes sites VCF."""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path

from data.gnomad_age_index import _info_val, has_age_data, variant_key


@dataclass(frozen=True)
class GnomadHit:
    in_gnomad: bool
    mutation_type: str | None
    ref: str | None
    alt: str | None
    af: float | None
    filters: str | None
    has_age_data: bool


def _parse_gnomad_line(cols: list[str]) -> GnomadHit:
    ref, alt = cols[3], cols[4]
    info = cols[7]
    af_raw = _info_val(info, "AF")
    af = float(af_raw) if af_raw is not None else None
    flt = _info_val(info, "FILTER") or cols[6] if len(cols) > 6 else None
    return GnomadHit(
        in_gnomad=True,
        mutation_type=f"{ref}>{alt}",
        ref=ref,
        alt=alt,
        af=af,
        filters=flt,
        has_age_data=has_age_data(info),
    )


def lookup_gnomad_variants(
    vcf_path: Path,
    query_keys: set[tuple[str, int, str, str]],
    *,
    cache_path: Path | None = None,
) -> dict[tuple[str, int, str, str], GnomadHit]:
    """One VCF pass; return gnomAD annotation per variant key."""
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if query_keys.issubset(cached.keys()):
            return {k: cached[k] for k in query_keys}

    found: dict[tuple[str, int, str, str], GnomadHit] = {}
    remaining = set(query_keys)
    n_lines = 0
    open_fn = gzip.open if str(vcf_path).endswith((".gz", ".bgz")) else open
    with open_fn(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n_lines += 1
            if n_lines % 2_000_000 == 0:
                print(f"  scanned {n_lines:,} VCF rows, matched {len(found):,}/{len(query_keys):,}")
            cols = line.rstrip("\n").split("\t", 8)
            if len(cols) < 8:
                continue
            key = variant_key(cols[0], int(cols[1]), cols[3], cols[4])
            if key not in remaining:
                continue
            found[key] = _parse_gnomad_line(cols)
            remaining.discard(key)
            if not remaining:
                print(f"  all {len(query_keys):,} query variants matched after {n_lines:,} rows")
                break

    miss = GnomadHit(
        in_gnomad=False,
        mutation_type=None,
        ref=None,
        alt=None,
        af=None,
        filters=None,
        has_age_data=False,
    )
    for k in query_keys:
        found.setdefault(k, miss)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(found, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  cached gnomAD lookup -> {cache_path}")
    return found
