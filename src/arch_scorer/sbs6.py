"""SBS6 mutation-type feature (pyrimidine-referenced)."""
from __future__ import annotations

_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_SBS6 = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")


def to_sbs6(ref: str, alt: str) -> str:
    ref, alt = str(ref).upper(), str(alt).upper()
    if len(ref) != 1 or len(alt) != 1 or ref == alt:
        return "non_snv"
    if ref not in _COMPLEMENT or alt not in _COMPLEMENT:
        return "non_snv"
    if ref in ("A", "G"):
        ref, alt = _COMPLEMENT[ref], _COMPLEMENT[alt]
    sub = f"{ref}>{alt}"
    return sub if sub in _SBS6 else "non_snv"
