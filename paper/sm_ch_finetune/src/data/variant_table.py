"""Parse Colin-Leitzinger variants into finetune-ready rows."""
from __future__ import annotations

import re

import pandas as pd

VARIANT_RE = re.compile(r"^(\d+|X|Y)-(\d+)-([ACGT]+)-([ACGT]+)$")


def parse_variant_id(variant: str) -> dict | None:
    m = VARIANT_RE.match(str(variant).strip())
    if not m:
        return None
    chrom, pos, ref, alt = m.groups()
    if len(ref) != 1 or len(alt) != 1:
        return None
    if not set(ref + alt).issubset({"A", "C", "G", "T"}):
        return None
    return {
        "chrom": chrom,
        "pos": int(pos),
        "ref": ref,
        "alt": alt,
        "variant_id": str(variant).strip(),
    }


def mutation_type(ref: str, alt: str) -> str:
    return f"{ref}>{alt}"


_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_SBS6 = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")


def to_sbs6(ref: str, alt: str) -> str | None:
    """Pyrimidine-referenced SBS6 subtype, or None for non-SNVs."""
    ref, alt = str(ref).upper(), str(alt).upper()
    if len(ref) != 1 or len(alt) != 1 or ref == alt:
        return None
    if ref not in _COMPLEMENT or alt not in _COMPLEMENT:
        return None
    if ref in ("A", "G"):
        ref, alt = _COMPLEMENT[ref], _COMPLEMENT[alt]
    sub = f"{ref}>{alt}"
    return sub if sub in _SBS6 else None


def mutation_type_feature(ref: str, alt: str, scheme: str = "raw") -> str:
    """Feature label: raw REF>ALT (12-class) or collapsed SBS6 (+ non_snv)."""
    scheme = str(scheme).lower().strip()
    if scheme == "raw":
        return mutation_type(str(ref), str(alt))
    if scheme == "sbs6":
        s = to_sbs6(ref, alt)
        return s if s is not None else "non_snv"
    raise ValueError(f"Unknown mutation scheme: {scheme!r} (use raw|sbs6)")


def assign_chrom_split(chrom: str, train_chroms: set[str], valid_chroms: set[str], test_chroms: set[str]) -> str | None:
    if chrom in train_chroms:
        return "train"
    if chrom in valid_chroms:
        return "valid"
    if chrom in test_chroms:
        return "test"
    return None


def build_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert age-filtered (or raw) Colin CSV rows to finetune parquet schema."""
    work = df[df["Indel"] == "Not Indel"].copy()
    parsed = []
    for _, row in work.iterrows():
        p = parse_variant_id(row["Variants"])
        if p is None:
            continue
        p.update(
            {
                "label_str": row["SM / CH"],
                "label": 1 if row["SM / CH"] == "CH" else 0,
                "gene": row.get("Gene.refGene", "."),
                "exonic_func": row.get("ExonicFunc.refGene", "."),
                "func": row.get("Func.refGene", "."),
                "mchip": row.get("M-CHIP regions", "No"),
                "lchip": row.get("L-CHIP regions", "No"),
                "cadd": pd.to_numeric(row.get("CADD_phred", "."), errors="coerce"),
                "mutation_type": mutation_type(p["ref"], p["alt"]),
            }
        )
        parsed.append(p)
    return pd.DataFrame(parsed)


def split_by_chromosome(
    df: pd.DataFrame,
    *,
    train_chroms: set[str],
    valid_chroms: set[str],
    test_chroms: set[str],
) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out["chrom"].map(
        lambda c: assign_chrom_split(c, train_chroms, valid_chroms, test_chroms)
    )
    return out[out["split"].notna()].copy()


def random_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.9,
    valid_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
    stratify_col: str = "label",
) -> pd.DataFrame:
    """Stratified random train/valid/test split (default 90/5/5)."""
    total = train_frac + valid_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    import numpy as np

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for label, group in df.groupby(stratify_col, sort=False):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_valid = int(round(n * valid_frac))
        n_train = n - n_test - n_valid
        for split_name, split_idx in (
            ("test", idx[:n_test]),
            ("valid", idx[n_test : n_test + n_valid]),
            ("train", idx[n_test + n_valid :]),
        ):
            part = df.loc[split_idx].copy()
            part["split"] = split_name
            parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    return out
