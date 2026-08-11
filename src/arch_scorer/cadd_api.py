"""CADD PHRED via the public CADD web API (no local CADD install required)."""
from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE = "https://cadd.gs.washington.edu/api/v1.0"
DEFAULT_VERSION = "GRCh38-v1.7"


def fetch_cadd_phred(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    *,
    base: str = DEFAULT_BASE,
    version: str = DEFAULT_VERSION,
    timeout: float = 60.0,
    retries: int = 3,
    sleep_s: float = 0.5,
) -> float:
    """Return CADD PHRED for an hg38 SNV."""
    chrom_n = str(chrom).replace("chr", "")
    ref, alt = str(ref).upper(), str(alt).upper()
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError("CADD SNV API requires single-base REF/ALT")
    url = f"{base.rstrip('/')}/{version}/{chrom_n}:{int(pos)}_{ref}_{alt}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "arch-scorer/0.1",
                },
            )
            with urlopen(req, timeout=timeout) as resp:
                data: Any = json.loads(resp.read().decode())
            if not data:
                raise RuntimeError(f"Empty CADD response for {url}")
            row = data[0] if isinstance(data, list) else data
            phred = row.get("PHRED")
            if phred in (None, ""):
                raise RuntimeError(f"No PHRED in CADD response: {row}")
            return float(phred)
        except (HTTPError, URLError, TimeoutError, RuntimeError, KeyError, ValueError) as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"CADD API failed for {chrom}:{pos} {ref}>{alt}: {last_err}")
