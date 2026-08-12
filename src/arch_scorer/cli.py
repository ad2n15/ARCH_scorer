"""CLI: arch-score — ARCH21 NC somatic probability (+ api_only SHAP ranks)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import ArchScorer, ScoreResult


def _load_emb(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix in {".csv", ".tsv"}:
        sep = "\t" if p.suffix == ".tsv" else ","
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


def _result_payload(res: ScoreResult) -> dict:
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
    if res.feature_ranks:
        payload["feature_ranks"] = res.feature_ranks
    return payload


def _print_result(res: ScoreResult, *, as_json: bool) -> None:
    payload = _result_payload(res)
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    print(
        f"[{res.model_name}] {res.gene} {res.chrom}:{res.pos} {res.ref}>{res.alt}  "
        f"P(somatic)={res.p_somatic:.4f}  "
        f"PM_call={'somatic' if res.call_somatic_pm else 'germline'}  "
        f"(thr={res.pm_threshold:.4f})"
    )
    if res.feature_ranks:
        print("feature_ranks (signed SHAP → somatic):")
        print(f"{'rank':>4}  {'feature_short':<14}  {'shap_value':>12}  feature")
        for r in res.feature_ranks:
            print(
                f"{r['rank']:>4}  {r['feature_short']:<14}  "
                f"{r['shap_value']:>12.6f}  {r['feature']}"
            )


def _score_one(
    scorer: ArchScorer,
    *,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    gene: str | None,
    emb: np.ndarray | None,
    cadd_phred: float | None,
    func: str | None,
    explain: bool,
) -> ScoreResult:
    return scorer.score(
        chrom,
        pos,
        ref,
        alt,
        emb_diff=emb,
        gene=gene,
        cadd_phred=cadd_phred,
        func_refgene=func,
        explain=explain,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "ARCH_scorer: somatic vs germline for ARCH21 MANE noncoding alleles. "
            "Models: emb_baseline | api_baseline | api_only (RF interpretability)."
        )
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chrom", help="Chromosome (single-variant mode)")
    g.add_argument(
        "--batch",
        default=None,
        help="TSV/CSV with columns chrom,pos,ref,alt[,gene]",
    )
    ap.add_argument("--pos", type=int, default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--alt", default=None)
    ap.add_argument("--gene", default=None)
    ap.add_argument(
        "--model",
        default="emb_baseline",
        choices=("emb_baseline", "api_baseline", "api_only"),
        help="emb_baseline (best) | api_baseline (no local emb) | api_only (SHAP ranks)",
    )
    ap.add_argument(
        "--explain",
        action="store_true",
        help="With api_only: print top-10 signed SHAP ranks of API tracks",
    )
    ap.add_argument(
        "--emb",
        default=None,
        help="fold_1 ALT−REF DIFF .npy (required for emb_baseline unless --extract-emb)",
    )
    ap.add_argument(
        "--extract-emb",
        action="store_true",
        help="Run fold_1 emb extract then score (needs GPU + AG research stack)",
    )
    ap.add_argument("--emb-out", default=None, help="Where to write extracted emb .npy")
    ap.add_argument("--cadd-phred", type=float, default=None)
    ap.add_argument("--func", default=None, help="ANNOVAR Func.refGene")
    ap.add_argument("--annovar-dir", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default=None, help="Write batch results TSV/JSONL")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.explain and args.model != "api_only":
        ap.error("--explain requires --model api_only")

    if args.batch is None:
        if args.pos is None or args.ref is None or args.alt is None:
            ap.error("single-variant mode requires --chrom --pos --ref --alt")

    emb = None
    if args.model == "emb_baseline":
        if args.batch is not None:
            ap.error("emb_baseline batch scoring needs per-row emb; use api_* models for --batch")
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
        fetch_cadd=args.model != "api_only",
        fetch_annovar=args.model != "api_only",
    )

    if args.batch is None:
        res = _score_one(
            scorer,
            chrom=args.chrom,
            pos=args.pos,
            ref=args.ref,
            alt=args.alt,
            gene=args.gene,
            emb=emb,
            cadd_phred=args.cadd_phred,
            func=args.func,
            explain=args.explain,
        )
        _print_result(res, as_json=args.json)
        return

    path = Path(args.batch)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    batch = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in batch.columns}
    for need in ("chrom", "pos", "ref", "alt"):
        if need not in cols:
            raise SystemExit(f"Batch file missing column '{need}'")
    gene_col = cols.get("gene")
    rows_out: list[dict] = []
    for i, row in batch.iterrows():
        gene = None if gene_col is None else (
            None if pd.isna(row[gene_col]) else str(row[gene_col])
        )
        res = _score_one(
            scorer,
            chrom=str(row[cols["chrom"]]),
            pos=int(row[cols["pos"]]),
            ref=str(row[cols["ref"]]),
            alt=str(row[cols["alt"]]),
            gene=gene,
            emb=None,
            cadd_phred=None,
            func=None,
            explain=args.explain,
        )
        payload = _result_payload(res)
        rows_out.append(payload)
        if not args.out:
            _print_result(res, as_json=args.json)
            if not args.json:
                print("---")

    if args.out:
        outp = Path(args.out)
        if outp.suffix.lower() == ".jsonl" or args.json:
            with outp.open("w") as fh:
                for p in rows_out:
                    fh.write(json.dumps(p) + "\n")
        else:
            flat = []
            for p in rows_out:
                base = {k: v for k, v in p.items() if k != "feature_ranks"}
                ranks = p.get("feature_ranks") or []
                if ranks:
                    for r in ranks:
                        flat.append({**base, **{f"rank_{k}" if k != "rank" else k: v for k, v in r.items()}})
                else:
                    flat.append(base)
            pd.DataFrame(flat).to_csv(
                outp, sep="\t" if outp.suffix.lower() in {".tsv", ".txt"} else ",", index=False
            )
        print(f"Wrote {len(rows_out)} variants → {outp}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
