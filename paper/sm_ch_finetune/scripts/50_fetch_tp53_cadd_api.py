#!/usr/bin/env python3
"""Fetch CADD PHRED for TP53 sensitivity-analysis variants via the CADD web API.

API docs: https://cadd.bihealth.org/api
Endpoint used: https://cadd.gs.washington.edu/api/v1.0/<version>/<chrom>:<pos>_<ref>_<alt>

Default cohort: results/tp53_cv5_api_only_gnomad_iarc_cosmic/oof_predictions.csv
(joined to GVS / IARC / COSMIC pilots for chrom/pos/ref/alt). SNVs only —
indels are skipped (SNV API). Resumable: re-run skips already scored IDs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from paths import project_subpath

DEFAULT_BASE = "https://cadd.gs.washington.edu/api/v1.0"
DEFAULT_VERSION = "GRCh38-v1.7"


def _load_meta() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    gvs = PROJECT / "data/pilot_ch_genes_gvs_plus_noage_sm/TP53"
    for split in ("train", "valid", "test"):
        p = gvs / f"{split}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    for rel in (
        "data/pilot_iarc_tp53_clean_test/test.parquet",
        "data/pilot_tp53_cosmic_haem/test.parquet",
    ):
        p = PROJECT / rel
        if p.exists():
            parts.append(pd.read_parquet(p))
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("variant_id")
    meta["variant_id"] = meta["variant_id"].astype(str)
    need = ["variant_id", "chrom", "pos", "ref", "alt"]
    missing = [c for c in need if c not in meta.columns]
    if missing:
        raise SystemExit(f"pilot meta missing columns: {missing}")
    return meta[need].copy()


def _is_snv(ref: str, alt: str) -> bool:
    return len(str(ref)) == 1 and len(str(alt)) == 1 and str(ref) in "ACGT" and str(alt) in "ACGT"


def _fetch_one(
    *,
    base: str,
    version: str,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    timeout: float,
) -> dict | None:
    chrom_n = str(chrom).replace("chr", "")
    url = f"{base.rstrip('/')}/{version}/{chrom_n}:{int(pos)}_{ref}_{alt}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "alphagenome-tp53-cadd/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    if not data:
        return None
    row = data[0] if isinstance(data, list) else data
    return {
        "cadd_raw": float(row["RawScore"]) if row.get("RawScore") not in (None, "") else float("nan"),
        "cadd_phred": float(row["PHRED"]) if row.get("PHRED") not in (None, "") else float("nan"),
        "cadd_api_chrom": str(row.get("Chrom", chrom_n)),
        "cadd_api_pos": int(row.get("Pos", pos)),
        "cadd_api_ref": str(row.get("Ref", ref)),
        "cadd_api_alt": str(row.get("Alt", alt)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort",
        default="results/tp53_cv5_api_only_gnomad_iarc_cosmic/oof_predictions.csv",
        help="CSV with variant_id (+ optional source) defining the cohort",
    )
    p.add_argument(
        "--out",
        default="results/tp53_cv5_api_only_gnomad_iarc_cosmic/cadd_phred_api.csv",
    )
    p.add_argument("--base-url", default=DEFAULT_BASE, help="CADD API v1 base (see cadd.bihealth.org/api)")
    p.add_argument("--version", default=DEFAULT_VERSION, help="e.g. GRCh38-v1.7")
    p.add_argument("--sleep", type=float, default=0.15, help="Seconds between API calls")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Score only first N pending (0=all)")
    args = p.parse_args()

    cohort_path = project_subpath(PROJECT, args.cohort)
    out_path = project_subpath(PROJECT, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(cohort_path)
    cohort["variant_id"] = cohort["variant_id"].astype(str)
    coord_cols = [c for c in ("chrom", "pos", "ref", "alt", "source") if c in cohort.columns]
    cohort = cohort[["variant_id"] + coord_cols].drop_duplicates("variant_id")

    # Prefer coordinates already present in the cohort CSV; fill gaps from pilots.
    meta = _load_meta()
    if all(c in cohort.columns for c in ("chrom", "pos", "ref", "alt")):
        df = cohort.copy()
        miss = df["pos"].isna()
        if miss.any():
            df = df.merge(meta, on="variant_id", how="left", suffixes=("", "_meta"))
            for c in ("chrom", "pos", "ref", "alt"):
                df[c] = df[c].where(df[c].notna(), df.get(f"{c}_meta"))
    else:
        df = cohort.merge(meta, on="variant_id", how="left")
    n_miss = int(df["pos"].isna().sum())
    if n_miss:
        print(f"WARNING: {n_miss} cohort variants missing chrom/pos/ref/alt")

    done: dict[str, dict] = {}
    cohort_ids = set(df["variant_id"].astype(str))
    if out_path.exists():
        prev = pd.read_csv(out_path)
        prev["variant_id"] = prev["variant_id"].astype(str)
        for _, r in prev.iterrows():
            vid = str(r["variant_id"])
            # Drop obsolete alleles from older cohort builds
            if vid not in cohort_ids:
                continue
            if pd.notna(r.get("cadd_phred")):
                done[vid] = r.to_dict()
        print(f"Resume: {len(done)} already scored in {out_path} (cohort n={len(cohort_ids)})")

    rows_out: list[dict] = list(done.values())
    pending = df.loc[~df["variant_id"].isin(done)].copy()
    pending["is_snv"] = [
        _is_snv(r, a) if pd.notna(r) and pd.notna(a) else False
        for r, a in zip(pending["ref"], pending["alt"])
    ]
    n_indel = int((~pending["is_snv"]).sum())
    for _, r in pending.loc[~pending["is_snv"]].iterrows():
        rows_out.append(
            {
                "variant_id": r["variant_id"],
                "source": r.get("source"),
                "chrom": r.get("chrom"),
                "pos": r.get("pos"),
                "ref": r.get("ref"),
                "alt": r.get("alt"),
                "cadd_phred": float("nan"),
                "cadd_raw": float("nan"),
                "cadd_status": "skipped_indel_or_missing",
                "cadd_version": args.version,
            }
        )
    pending = pending.loc[pending["is_snv"]].reset_index(drop=True)
    if args.limit and args.limit > 0:
        pending = pending.head(int(args.limit))

    print(
        f"Cohort n={len(df)} already={len(done)} indel/skip={n_indel} "
        f"to_query={len(pending)} version={args.version}"
    )

    ok = fail = 0
    for i, r in pending.iterrows():
        status = "ok"
        hit = None
        last_err = ""
        for attempt in range(1, int(args.max_retries) + 1):
            try:
                hit = _fetch_one(
                    base=args.base_url,
                    version=args.version,
                    chrom=str(r["chrom"]),
                    pos=int(r["pos"]),
                    ref=str(r["ref"]),
                    alt=str(r["alt"]),
                    timeout=float(args.timeout),
                )
                if hit is None:
                    status = "empty"
                break
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
                last_err = str(e)
                status = "error"
                time.sleep(min(2.0 * attempt, 10.0))
        row = {
            "variant_id": r["variant_id"],
            "source": r.get("source"),
            "chrom": r.get("chrom"),
            "pos": r.get("pos"),
            "ref": r.get("ref"),
            "alt": r.get("alt"),
            "cadd_phred": hit["cadd_phred"] if hit else float("nan"),
            "cadd_raw": hit["cadd_raw"] if hit else float("nan"),
            "cadd_status": status if hit or status == "empty" else f"error:{last_err[:120]}",
            "cadd_version": args.version,
        }
        if hit:
            row.update({k: hit[k] for k in hit if k.startswith("cadd_api_")})
            ok += 1
        else:
            fail += 1
        rows_out.append(row)

        if (ok + fail) % 25 == 0 or (ok + fail) == len(pending):
            pd.DataFrame(rows_out).drop_duplicates("variant_id", keep="last").to_csv(
                out_path, index=False
            )
            print(f"  progress {ok + fail}/{len(pending)} ok={ok} fail/empty={fail} -> {out_path}")

        time.sleep(float(args.sleep))

    out = pd.DataFrame(rows_out).drop_duplicates("variant_id", keep="last")
    # Restrict output to current cohort only (drop leftover alleles from older pools)
    out = out.loc[out["variant_id"].astype(str).isin(cohort_ids)].copy()
    out.to_csv(out_path, index=False)
    n_phred = int(out["cadd_phred"].notna().sum())
    print(
        f"Wrote {out_path} n={len(out)} cadd_phred={n_phred} "
        f"({n_phred / max(len(out), 1):.1%}) ok={ok} fail/empty={fail}"
    )
    if "source" in out.columns:
        print(out.groupby("source")["cadd_phred"].apply(lambda s: f"{s.notna().sum()}/{len(s)}"))


if __name__ == "__main__":
    main()
