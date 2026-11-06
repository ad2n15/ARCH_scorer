#!/usr/bin/env python3
"""Download the released variant-level tables from Zenodo into data/release/.

  python scripts/fetch_data.py

Files (public-derived only; see data/README.md):
  genomewide_age_filtered.csv.gz, arch21_noncoding_public.csv.gz, arch21_coding_public.csv.gz, summary.json
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

ZENODO_RECORD = None  # set to the Zenodo record id once the data are deposited
FILES = ["genomewide_age_filtered.csv.gz", "arch21_noncoding_public.csv.gz", "arch21_coding_public.csv.gz",
         "summary.json"]
OUT = Path(__file__).resolve().parents[1] / "data" / "release"


def main() -> None:
    if ZENODO_RECORD is None:
        sys.exit("The Zenodo record is not public yet. Maintainers: build the files with "
                 "scripts/export_public_data.py.")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        (OUT / name).write_bytes(r.content)
        print(f"{name}  {len(r.content) / 1e6:.1f} MB  md5 {hashlib.md5(r.content).hexdigest()}")


if __name__ == "__main__":
    main()
