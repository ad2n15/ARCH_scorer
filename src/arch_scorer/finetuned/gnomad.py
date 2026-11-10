"""gnomAD v4.1 context for each variant: frequency, allele balance (AB) and carrier age.

Fetched from the public gnomAD GraphQL API (no key). Exome and genome histograms are summed.
AB and age summaries follow the paper's scripts (paper/sm_ch_finetune/scripts/75_*.py and
src/data/gnomad_age_index.py): AB median = midpoint of the 0.05-wide bin holding the median carrier;
age median = midpoint of the 5-year bin holding the median heterozygous carrier (<30 counted as 27.5,
>80 as 82.5).

Germline heterozygous variants have AB near 0.5. Clonal (somatic) blood variants are seen in a minority
of reads, so AB is well below 0.5, and their carriers tend to be older. The paper used these signals to
define somatic labels, so they are context for a variant, not an independent test of the model.
"""
from __future__ import annotations

import time

import numpy as np
import requests

API = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"
BATCH = 20
AB_EDGES = np.round(np.arange(0, 1.0001, 0.05), 2)
AGE_EDGES = np.arange(30, 81, 5)

_FIELDS = """ac an homozygote_count
  age_distribution { het { bin_freq n_smaller n_larger } }
  quality_metrics { allele_balance { alt { bin_freq } } }"""

_session = requests.Session()
_session.headers["User-Agent"] = "arch-scorer/1.0 (research use)"

COLUMNS = ["gnomad_status", "gnomad_af", "gnomad_ac", "gnomad_an", "gnomad_hom",
           "gnomad_ab_median", "gnomad_ab_n", "gnomad_ab_hist",
           "gnomad_age_median", "gnomad_age_n", "gnomad_age_hist"]


def _post(query: str, tries: int = 4, timeout: int = 60) -> dict:
    for i in range(tries):
        try:
            r = _session.post(API, json={"query": query}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
        time.sleep(5 * 2 ** i)  # the API rate-limits bursts
    raise RuntimeError(f"gnomAD API unavailable (HTTP {r.status_code})")


def ab_median(counts) -> float:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n <= 0:
        return np.nan
    mid = (np.arange(len(counts)) + 0.5) / len(counts)
    return float(mid[np.searchsorted(np.cumsum(counts), (n + 1) / 2)])


def age_median(counts, n_smaller: int = 0, n_larger: int = 0) -> float:
    ages = np.concatenate([[27.5], AGE_EDGES[:-1] + 2.5, [82.5]])
    w = np.concatenate([[n_smaller], np.asarray(counts, dtype=float), [n_larger]])
    n = w.sum()
    if n <= 0:
        return np.nan
    return float(ages[np.searchsorted(np.cumsum(w), n / 2)])


def summarise(rec: dict | None) -> dict:
    """One API variant record -> flat columns."""
    if not rec:
        return {"gnomad_status": "not in gnomAD v4.1"}
    ac = an = hom = 0
    ab = np.zeros(len(AB_EDGES) - 1)
    age = np.zeros(len(AGE_EDGES) - 1)
    small = large = 0
    for part in ("exome", "genome"):
        d = rec.get(part)
        if not d:
            continue
        ac, an, hom = ac + (d.get("ac") or 0), an + (d.get("an") or 0), hom + (d.get("homozygote_count") or 0)
        h = ((d.get("age_distribution") or {}).get("het")) or {}
        if h.get("bin_freq"):
            age += np.asarray(h["bin_freq"][: len(age)], dtype=float)
            small, large = small + (h.get("n_smaller") or 0), large + (h.get("n_larger") or 0)
        a = (((d.get("quality_metrics") or {}).get("allele_balance") or {}).get("alt")) or {}
        if a.get("bin_freq"):
            ab += np.asarray(a["bin_freq"][: len(ab)], dtype=float)
    if ac == 0:
        return {"gnomad_status": "not observed in gnomAD v4.1 (AC = 0)", "gnomad_an": an}
    n_age = int(age.sum() + small + large)
    return {
        "gnomad_status": "found",
        "gnomad_af": ac / an if an else np.nan, "gnomad_ac": ac, "gnomad_an": an, "gnomad_hom": hom,
        "gnomad_ab_median": ab_median(ab), "gnomad_ab_n": int(ab.sum()),
        "gnomad_ab_hist": "|".join(str(int(x)) for x in ab),
        "gnomad_age_median": age_median(age, small, large) if n_age else np.nan, "gnomad_age_n": n_age,
        "gnomad_age_hist": "|".join(str(int(x)) for x in [small, *age, large]),
    }


def fetch(variant_ids: list[str], progress=None) -> list[dict]:
    """gnomAD columns for each 'chrom-pos-ref-alt' id, in order. Network errors give a status, not an exception."""
    out: list[dict] = []
    for i in range(0, len(variant_ids), BATCH):
        chunk = variant_ids[i:i + BATCH]
        q = "{" + " ".join(f'v{k}: variant(variantId: "{v}", dataset: {DATASET}) {{ '
                           f"exome {{ {_FIELDS} }} genome {{ {_FIELDS} }} }}" for k, v in enumerate(chunk)) + "}"
        try:
            data = _post(q).get("data") or {}
            out += [summarise(data.get(f"v{k}")) for k in range(len(chunk))]
        except Exception as e:  # noqa: BLE001
            out += [{"gnomad_status": f"unavailable ({type(e).__name__})"} for _ in chunk]
        if progress:
            progress(min(i + BATCH, len(variant_ids)) / len(variant_ids), desc="gnomAD")
    return [{c: r.get(c, np.nan) for c in COLUMNS} for r in out]
