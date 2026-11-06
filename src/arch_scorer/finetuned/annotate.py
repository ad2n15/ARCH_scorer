"""Variant parsing, reference sequence and annotations from free public APIs (no keys needed).

  sequence       UCSC Genome Browser API (hg38), Ensembl REST as fallback
  Func.refGene   ANNOVAR-style gene-based class computed locally from UCSC RefSeq (refGene) transcripts
  gene           gene(s) of the winning class, as ANNOVAR reports them
  CADD-PHRED     CADD API, GRCh38 v1.7 (the version used in training)
  ExonicFunc     coding consequence from the same CADD call (synonymous / nonsynonymous / stopgain / stoploss)
  mutation type  REF>ALT (12 classes) and pyrimidine-referenced SBS6
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache

import requests

UCSC = "https://api.genome.ucsc.edu/getData"
ENSEMBL = "https://rest.ensembl.org"
CADD = "https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7_inclAnno"
SPLICE_BP, FLANK_BP = 2, 1000  # ANNOVAR defaults: splicing threshold, upstream/downstream distance
CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
_session = requests.Session()
_session.headers["User-Agent"] = "arch-scorer/1.0 (research use)"


@dataclass(frozen=True)
class Variant:
    chrom: str  # no "chr" prefix
    pos: int    # 1-based
    ref: str
    alt: str

    @property
    def id(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"


def parse_variant(text: str) -> Variant:
    """Accepts 20-32437597-T-G, chr20:32437597:T:G, chr20:32437597 T>G, '20 32437597 T G'."""
    parts = [p for p in re.split(r"[\s:\-_>/,]+", text.strip()) if p]
    if len(parts) != 4:
        raise ValueError(f"Could not parse {text!r}; use chrom-pos-ref-alt, e.g. 20-32437597-T-G")
    chrom = re.sub(r"^chr", "", parts[0], flags=re.I).upper()
    ref, alt = parts[2].upper(), parts[3].upper()
    if chrom not in CHROMS:
        raise ValueError(f"Unsupported chromosome {parts[0]!r} (1-22, X, Y)")
    if not (len(ref) == len(alt) == 1 and ref in _COMP and alt in _COMP and ref != alt):
        raise ValueError(f"{text!r}: only single-nucleotide variants are supported")
    return Variant(chrom, int(parts[1]), ref, alt)


def mutation_type(v: Variant) -> str:
    return f"{v.ref}>{v.alt}"


def sbs6(v: Variant) -> str:
    ref, alt = (v.ref, v.alt) if v.ref in "CT" else (_COMP[v.ref], _COMP[v.alt])
    return f"{ref}>{alt}"


def _get(url: str, params: dict | None = None, *, tries: int = 3, timeout: int = 60, **kw):
    for i in range(tries):
        try:
            r = _session.get(url, params=params, timeout=timeout, **kw)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
        time.sleep(2 ** i)
    raise RuntimeError(f"{url} unavailable (HTTP {r.status_code})")


# ---------------------------------------------------------------- sequence
@lru_cache(maxsize=256)
@lru_cache(maxsize=1)
def _fasta():
    import os

    path = os.environ.get("ARCH_FASTA")
    if not path:
        return None
    import pyfaidx

    return pyfaidx.Fasta(path)


def fetch_sequence(chrom: str, start: int, end: int) -> str:
    """hg38 sequence for 0-based half-open [start, end), upper case.
    Uses a local GRCh38 FASTA when ARCH_FASTA is set (offline / batch use), otherwise the UCSC API."""
    fa = _fasta()
    if fa is not None:
        key = next(k for k in (f"chr{chrom}", chrom) if k in fa)
        return str(fa[key][start:end]).upper()
    try:
        r = _get(f"{UCSC}/sequence", {"genome": "hg38", "chrom": f"chr{chrom}", "start": start, "end": end})
        return r.json()["dna"].upper()
    except Exception:
        r = _get(f"{ENSEMBL}/sequence/region/human/{chrom}:{start + 1}..{end}:1",
                 headers={"Content-Type": "text/plain"})
        return r.text.strip().upper()


def variant_window(v: Variant, window: int = 65536) -> tuple[str, int, str]:
    """ALT-centred window exactly as in training (variant_dataset.py). Returns (seq, centre index, ref base)."""
    start = max(0, v.pos - window // 2 - 1)
    seq = fetch_sequence(v.chrom, start, start + window).ljust(window, "N")
    center = min(v.pos - 1 - start, window - 1)
    return seq[:center] + v.alt + seq[center + 1:], center, seq[center]


# ---------------------------------------------------------------- ANNOVAR-style gene annotation
def _tx(rec: dict) -> dict:
    ints = lambda s: [int(x) for x in str(s).strip(",").split(",") if x]  # noqa: E731
    return {"gene": rec["name2"], "strand": rec["strand"], "tx": (int(rec["txStart"]), int(rec["txEnd"])),
            "cds": (int(rec["cdsStart"]), int(rec["cdsEnd"])),
            "exons": list(zip(ints(rec["exonStarts"]), ints(rec["exonEnds"])))}


@lru_cache(maxsize=256)
def refgene_transcripts(chrom: str, pos: int) -> tuple[dict, ...]:
    r = _get(f"{UCSC}/track", {"genome": "hg38", "track": "refGene", "chrom": f"chr{chrom}",
                               "start": max(0, pos - FLANK_BP - 1), "end": pos + FLANK_BP})
    return tuple(_tx(rec) for rec in r.json().get("refGene", []))


def _classify(pos: int, t: dict) -> str | None:
    """One transcript -> ANNOVAR class for a 1-based position (exon starts are 0-based in genePred)."""
    (tx_s, tx_e), (cds_s, cds_e), plus = t["tx"], t["cds"], t["strand"] == "+"
    coding = cds_s != cds_e
    if pos <= tx_s:
        return ("upstream" if plus else "downstream") if tx_s - pos < FLANK_BP else None
    if pos > tx_e:
        return ("downstream" if plus else "upstream") if pos - tx_e <= FLANK_BP else None
    exons = t["exons"]
    for k, (es, ee) in enumerate(exons):
        if es < pos <= ee:
            if not coding:
                return "ncRNA_exonic"
            if pos <= cds_s:
                return "UTR5" if plus else "UTR3"
            if pos > cds_e:
                return "UTR3" if plus else "UTR5"
            return "exonic"
    near = any((k > 0 and es - SPLICE_BP < pos <= es) or (k < len(exons) - 1 and ee < pos <= ee + SPLICE_BP)
               for k, (es, ee) in enumerate(exons))
    if near:
        return "splicing" if coding else "ncRNA_splicing"
    return "intronic" if coding else "ncRNA_intronic"


# ANNOVAR precedence: exonic = splicing > ncRNA > UTR5 = UTR3 > intronic > upstream = downstream > intergenic
_TIERS = [("exonic", "splicing"), ("ncRNA_exonic", "ncRNA_splicing"), ("ncRNA_intronic",),
          ("UTR5", "UTR3"), ("intronic",), ("upstream", "downstream")]


def annovar_func(pos: int, transcripts) -> tuple[str, str]:
    """(Func.refGene, Gene.refGene) for a 1-based position given refGene transcripts."""
    per_tx = [(t["gene"], c) for t in transcripts if (c := _classify(pos, t))]
    # ANNOVAR uses a gene's non-coding (NR_) transcripts only when none of its coding transcripts overlap.
    coding_genes = {g for g, c in per_tx if not c.startswith("ncRNA")}
    hits: dict[str, list[str]] = {}
    for g, c in per_tx:
        if c.startswith("ncRNA") and g in coding_genes:
            continue
        hits.setdefault(c, [])
        if g not in hits[c]:
            hits[c].append(g)
    for tier in _TIERS:
        found = [c for c in tier if c in hits]
        if found:
            if found == ["ncRNA_exonic", "ncRNA_splicing"]:
                func = "ncRNA_exonic;splicing"
            else:
                func = ";".join(found)
            genes = []
            for c in found:
                genes += [g for g in hits[c] if g not in genes]
            return func, ";".join(genes)
    return "intergenic", "NONE"


# ---------------------------------------------------------------- CADD
_EXONIC = {"STOP_GAINED": "stopgain", "STOP_LOST": "stoploss", "NON_SYNONYMOUS": "nonsynonymous SNV",
           "SYNONYMOUS": "synonymous SNV"}


@lru_cache(maxsize=1024)
def cadd(v: Variant) -> tuple[float | None, str | None]:
    """(CADD-PHRED v1.7, most severe coding consequence or None)."""
    rows = _get(f"{CADD}/{v.chrom}:{v.pos}_{v.ref}_{v.alt}").json()
    if not rows:
        return None, None
    phred = float(rows[0]["PHRED"])
    cons = {r.get("Consequence") for r in rows}
    coding = next((_EXONIC[c] for c in _EXONIC if c in cons), None)
    return phred, coding


def annotate(v: Variant) -> dict:
    """All annotations used by the two fusion models; missing values are None."""
    out = {"variant_id": v.id, "chrom": v.chrom, "pos": v.pos, "ref": v.ref, "alt": v.alt,
           "mutation_type_raw": mutation_type(v), "mutation_type_sbs6": sbs6(v), "notes": []}
    try:
        tx = refgene_transcripts(v.chrom, v.pos)
        out["func"], out["gene"] = annovar_func(v.pos, tx)
        # genes whose transcript (+/- 1 kb) covers the variant, e.g. TET2 for a TET2-AS1 intronic variant
        out["overlapping_genes"] = sorted({t["gene"] for t in tx
                                           if t["tx"][0] - FLANK_BP < v.pos <= t["tx"][1] + FLANK_BP})
    except Exception as e:  # noqa: BLE001
        out["func"] = out["gene"] = None
        out["overlapping_genes"] = []
        out["notes"].append(f"RefSeq annotation unavailable ({type(e).__name__})")
    try:
        out["cadd_phred"], coding = cadd(v)
    except Exception as e:  # noqa: BLE001
        out["cadd_phred"], coding = None, None
        out["notes"].append(f"CADD unavailable ({type(e).__name__})")
    func = out.get("func") or ""
    out["exonic_func"] = (coding or "unknown") if "exonic" in func.split(";") else "."
    return out
