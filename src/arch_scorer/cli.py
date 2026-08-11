"""CLI: arch-score — ARCH21 NC somatic probability."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .pipeline import ArchScorer


def _load_emb(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix in {".csv", ".tsv"}:
        sep = "\t" if p.suffix == ".tsv" else ","
        import pandas as pd

        df = pd.read_csv(p, sep=sep)
        cols = [c for c in df.columns if str(c).startswith("e") or str(c).isdigit()]
        if not cols:
            if df.shape[1] == 1536:
                cols = list(df.columns)
            elif df.shape[1] > 1536:
                cols = list(df.columns[-1536:])
            else:
                raise SystemExit(f"Cannot find 1536 emb columns in {p}")
        arr = df[cols].to_numpy(dtype=np.float64)
    else:
        raise SystemExit(f"Unsupported emb file: {p}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise SystemExit(f"Expected 1-d emb vector, got shape {arr.shape}")
    return arr


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "ARCH_scorer: somatic vs germline for ARCH21 MANE noncoding alleles. "
            "Models: emb_baseline (fold_1 DIFF+baseline) or api_baseline (AG API tracks+baseline)."
        )
    )
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--gene", default=None)
    ap.add_argument(
        "--model",
        default="emb_baseline",
        choices=("emb_baseline", "api_baseline"),
        help="Production model (default: emb_baseline)",
    )
    ap.add_argument(
        "--emb",
        default=None,
        help="fold_1 ALT−REF DIFF .npy (required for emb_baseline unless --extract-emb)",
    )
    ap.add_argument(
        "--extract-emb",
        action="store_true",
        help="Run scripts/extract_fold1_emb_diff.py then score (needs GPU + AG research stack)",
    )
    ap.add_argument("--emb-out", default=None, help="Where to write extracted emb .npy")
    ap.add_argument("--cadd-phred", type=float, default=None)
    ap.add_argument("--func", default=None, help="ANNOVAR Func.refGene")
    ap.add_argument("--annovar-dir", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    emb = None
    if args.model == "emb_baseline":
        if args.extract_emb:
            from .extract_emb import extract_fold1_diff

            out = Path(args.emb_out) if args.emb_out else Path("emb_diff.npy")
            emb = extract_fold1_diff(
                args.chrom, args.pos, args.ref, args.alt, out_path=out
            )
        elif args.emb:
            emb = _load_emb(args.emb)
        else:
            ap.error("emb_baseline requires --emb PATH or --extract-emb")

    scorer = ArchScorer(
        args.model,
        args.artifact,
        annovar_dir=args.annovar_dir,
        api_key=args.api_key,
    )
    res = scorer.score(
        args.chrom,
        args.pos,
        args.ref,
        args.alt,
        emb_diff=emb,
        gene=args.gene,
        cadd_phred=args.cadd_phred,
        func_refgene=args.func,
    )
    payload = {
        "chrom": res.chrom,
        "pos": res.pos,
        "ref": res.ref,
        "alt": res.alt,
        "gene": res.gene,
        "cadd_phred": res.cadd_phred,
        "mutation_type": res.mutation_type,
        "Func.refGene": res.func_refgene,
        "p_somatic": res.p_somatic,
        "call_somatic_pm": res.call_somatic_pm,
        "pm_threshold": res.pm_threshold,
        "model": res.model_name,
    }
    if res.api_tracks:
        payload["api_tracks"] = res.api_tracks
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"[{res.model_name}] {res.gene} {res.chrom}:{res.pos} {res.ref}>{res.alt}  "
            f"P(somatic)={res.p_somatic:.4f}  "
            f"PM_call={'somatic' if res.call_somatic_pm else 'germline'}  "
            f"(thr={res.pm_threshold:.4f})"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
