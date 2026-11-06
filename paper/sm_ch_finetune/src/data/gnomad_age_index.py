"""gnomAD age histogram helpers (Colin-Leitzinger et al. 2025)."""
from __future__ import annotations

import gzip
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VARIANT_RE = re.compile(r"^(\d+|X|Y|M)-(\d+)-([ACGT]+)-([ACGT]+)$")

# gnomAD exomes r2.1.1: 11 bins at 30,35,...,80 (years).
AGE_BIN_STARTS = tuple(range(30, 81, 5))
AGE_BELOW_CENTER = 27.5
AGE_ABOVE_CENTER = 82.5
AGE_NORM_MIN = 30.0
AGE_NORM_SCALE = 50.0
N_AGE_BINS = len(AGE_BIN_STARTS)


def parse_variant_id(variant: str) -> tuple[str, int, str, str] | None:
    m = VARIANT_RE.match(str(variant).strip())
    if not m:
        return None
    chrom, pos, ref, alt = m.groups()
    return f"chr{chrom}", int(pos), ref, alt


def variant_key(chrom: str, pos: int, ref: str, alt: str) -> tuple[str, int, str, str]:
    c = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    return (c, int(pos), ref.upper(), alt.upper())


def _info_val(info: str, key: str) -> str | None:
    for part in info.split(";"):
        if part.startswith(f"{key}="):
            return part.split("=", 1)[1]
    return None


@dataclass(frozen=True)
class AgeHistFeatures:
    has_age: bool
    n_carriers: int
    n_smaller: int
    n_larger: int
    bin_counts: tuple[int, ...] = field(default_factory=tuple)
    mean_age: float = float("nan")
    median_age: float = float("nan")

    def hist_array(self) -> np.ndarray:
        return np.asarray(self.bin_counts, dtype=np.float32)

    def mean_age_norm(self) -> float:
        if not self.has_age or not np.isfinite(self.mean_age):
            return float("nan")
        return float((self.mean_age - AGE_NORM_MIN) / AGE_NORM_SCALE)

    def as_dict(self) -> dict:
        out = {
            "gnomad_has_age": self.has_age,
            "gnomad_age_n_carriers": self.n_carriers,
            "gnomad_age_n_smaller": self.n_smaller,
            "gnomad_age_n_larger": self.n_larger,
            "gnomad_age_mean": self.mean_age,
            "gnomad_age_median": self.median_age,
        }
        for i, c in enumerate(self.bin_counts):
            out[f"gnomad_age_hist_{i:02d}"] = int(c)
        return out


def _parse_bin_freq(info: str, key: str = "age_hist_het_bin_freq") -> list[int] | None:
    bf = _info_val(info, key)
    if bf is None:
        return None
    try:
        return [int(x) for x in bf.split("|") if x != ""]
    except ValueError:
        return None


def parse_age_hist_het(info: str) -> AgeHistFeatures:
    """Parse heterozygous carrier age histogram from gnomAD INFO."""
    bins = _parse_bin_freq(info)
    ns = int(_info_val(info, "age_hist_het_n_smaller") or 0)
    nl = int(_info_val(info, "age_hist_het_n_larger") or 0)
    if bins is None:
        return AgeHistFeatures(False, 0, 0, 0)

    counts = bins[:N_AGE_BINS]
    if len(counts) < N_AGE_BINS:
        counts = counts + [0] * (N_AGE_BINS - len(counts))
    counts_t = tuple(int(c) for c in counts[:N_AGE_BINS])
    n_carriers = int(sum(counts_t) + ns + nl)
    if n_carriers <= 0:
        return AgeHistFeatures(False, 0, ns, nl, counts_t)

    ages: list[float] = []
    weights: list[int] = []
    if ns > 0:
        ages.append(AGE_BELOW_CENTER)
        weights.append(ns)
    for age_start, c in zip(AGE_BIN_STARTS, counts_t):
        if c > 0:
            ages.append(float(age_start) + 2.5)
            weights.append(c)
    if nl > 0:
        ages.append(AGE_ABOVE_CENTER)
        weights.append(nl)

    ages_arr = np.asarray(ages, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    mean_age = float(np.average(ages_arr, weights=weights_arr))
    cum = np.cumsum(weights_arr)
    half = n_carriers / 2.0
    median_age = float(ages_arr[np.searchsorted(cum, half, side="left")])
    return AgeHistFeatures(True, n_carriers, ns, nl, counts_t, mean_age, median_age)


def has_age_data(info: str) -> bool:
    """True if gnomAD age_hist_het has any carrier ages (paper: studies with age data)."""
    return parse_age_hist_het(info).has_age


def build_age_index(
    vcf_path: Path,
    query_keys: set[tuple[str, int, str, str]],
    *,
    cache_path: Path | None = None,
) -> dict[tuple[str, int, str, str], bool]:
    """One VCF pass: map variant keys to has_age_data."""
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if query_keys.issubset(cached.keys()):
            return {k: cached[k] for k in query_keys}

    found: dict[tuple[str, int, str, str], bool] = {}
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
            found[key] = has_age_data(cols[7])
            remaining.discard(key)
            if not remaining:
                print(f"  all {len(query_keys):,} query variants matched after {n_lines:,} rows")
                break

    for k in query_keys:
        found.setdefault(k, False)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(found, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  cached age index -> {cache_path}")
    return found


def build_age_features_index(
    vcf_path: Path,
    query_keys: set[tuple[str, int, str, str]],
    *,
    cache_path: Path | None = None,
) -> dict[tuple[str, int, str, str], AgeHistFeatures]:
    """One VCF pass: map variant keys to parsed age histogram features."""
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if query_keys.issubset(cached.keys()):
            return {k: cached[k] for k in query_keys}

    found: dict[tuple[str, int, str, str], AgeHistFeatures] = {}
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
            found[key] = parse_age_hist_het(cols[7])
            remaining.discard(key)
            if not remaining:
                print(f"  all {len(query_keys):,} query variants matched after {n_lines:,} rows")
                break

    empty = AgeHistFeatures(False, 0, 0, 0)
    for k in query_keys:
        found.setdefault(k, empty)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(found, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  cached age features -> {cache_path}")
    return found
