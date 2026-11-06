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

ZENODO_RECORD = 24008687  # https://doi.org/10.5281/zenodo.24008687
MD5 = {"genomewide_age_filtered.csv.gz": "774402a7e5f64d66ccd8233c0941db67",
       "arch21_noncoding_public.csv.gz": "544e790e68268ea7bedc9fcecf60279b",
       "arch21_coding_public.csv.gz": "3bbb3d27a0cbe3aa10239ccf26584d7b",
       "summary.json": "517063e69f28f98aee844b92a4dd89b9"}
FILES = list(MD5)
RESTRICTED = ("The files are restricted until the paper is published; reviewers can download them with the share "
              "link provided with the manuscript and place them in data/release/.")
OUT = Path(__file__).resolve().parents[1] / "data" / "release"


def main() -> None:
    if ZENODO_RECORD is None:
        sys.exit("No Zenodo record set.")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
        r = requests.get(url, timeout=300)
        if r.status_code in (401, 403, 404) or "text/html" in r.headers.get("Content-Type", ""):
            sys.exit(f"{name}: not downloadable. {RESTRICTED}")
        r.raise_for_status()
        md5 = hashlib.md5(r.content).hexdigest()
        if md5 != MD5[name]:
            sys.exit(f"{name}: checksum mismatch ({md5}); download again or use the share link.")
        (OUT / name).write_bytes(r.content)
        print(f"{name}  {len(r.content) / 1e6:.1f} MB  md5 ok")


if __name__ == "__main__":
    main()
