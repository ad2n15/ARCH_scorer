"""Score GRCh38 SNVs with the fine-tuned heads, alone and fused with annotations.

    from arch_scorer.finetuned import Scorer
    s = Scorer("ntv3")                       # or "alphagenome", "borzoi"; downloads the backbone on first use
    s.score(["20-32437597-T-G"])             # DataFrame, one row per variant, one column pair per task

    arch-score-ft variants.txt --backbone ntv3 -o scores.csv     # one variant per line, or a VCF
    arch-score-ft variants.txt --html report.html                # also write a readable HTML report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import annotate as A
from . import gnomad as G
from .backbones import BACKBONE_CLASSES, WINDOW
from .fusion import FusionLR
from .heads import load_head
from .report import summary_table, write_html

MODELS = Path(__file__).resolve().parents[3] / "models" / "finetuned"
BACKBONES = tuple(BACKBONE_CLASSES)
TASKS = ("arch21_noncoding", "arch21_cnc", "genomewide")
MISSENSE = {"nonsynonymous SNV", "stopgain", "stoploss"}


def fusion_frame(task: str, ann: pd.DataFrame, prob: np.ndarray, arch_genes: list[str] | None = None) -> pd.DataFrame:
    """Inputs for a task's fusion LR, built the way its training table was built."""
    if task.startswith("arch21"):
        overlap = ann["overlapping_genes"] if "overlapping_genes" in ann else ann["gene"].fillna("").str.split(";")
        return pd.DataFrame({
            "finetune_prob": prob,
            "cadd_phred": ann["cadd_phred"].astype(float).values,
            "mutation_type": ann["mutation_type_sbs6"].values,
            "Func.refGene": ann["func"].fillna("unknown").values,
            "gene": [next((g for g in arch_genes if g in (o or [])), "unknown") for o in overlap],
        })
    # Genome-wide training CADD came from the Colin-Leitzinger table, which scores missense and stop variants only.
    cadd = np.where(ann["exonic_func"].isin(MISSENSE), ann["cadd_phred"].astype(float), np.nan)
    return pd.DataFrame({
        "finetune_prob": prob,
        "cadd": cadd,
        "exonic_func": ann["exonic_func"].values,
        "gene": ann["gene"].fillna("unknown").values,
        "mutation_type": ann["mutation_type_raw"].values,
    })


def read_variants(path: Path) -> list[str]:
    """Plain list (one variant per line, e.g. 20-32437597-T-G) or VCF (CHROM POS ID REF ALT)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        out.append(f"{f[0]}-{f[1]}-{f[3]}-{f[4]}" if len(f) >= 5 else line.strip())
    return out


class Scorer:
    def __init__(self, backbone: str = "ntv3", tasks=TASKS, model_path: str | None = None,
                 device: str | None = None, models_dir: Path = MODELS, gnomad: bool = True):
        if backbone not in BACKBONE_CLASSES:
            raise ValueError(f"backbone must be one of {BACKBONES}")
        self.name = backbone
        self.gnomad = gnomad  # add gnomAD v4.1 frequency, allele balance and carrier age (public API)
        self.tasks, self.meta = {}, {}
        for t in tasks:
            d = Path(models_dir) / backbone / t
            head, cfg = load_head(d)
            q = json.loads((d / "score_quantiles.json").read_text())
            fus = d / "fusion.json"
            self.tasks[t] = {"head": head, "cfg": cfg, "fusion": FusionLR.load(fus) if fus.is_file() else None,
                             "quantiles": {k: np.asarray(v) for k, v in q.items()}}
            m = json.loads((d / "test_metrics.json").read_text())
            prev = m["n_pos"] / m["n"]
            # prevalence-matched threshold (as in the paper): the score at the (1 - prevalence) test quantile
            thr = {k: float(np.interp(100 * (1 - prev), np.linspace(0, 100, len(v)), v)) for k, v in q.items()}
            self.meta[t] = {**m, "prevalence": prev, "threshold": thr}
        self.backbone = BACKBONE_CLASSES[backbone](model_path, device)

    @staticmethod
    def percentile(q: np.ndarray, p: float) -> float:
        """Percentile of p among the held-out test-set scores (q = 0th..100th percentiles)."""
        return float(np.interp(p, q, np.linspace(0, 100, len(q)))) if np.isfinite(p) else np.nan

    def embed(self, variants: list[A.Variant], progress=None) -> tuple[torch.Tensor, list[str]]:
        embs, refs = [], []
        bs = self.backbone.batch_size
        for i in range(0, len(variants), bs):
            wins = [A.variant_window(v, WINDOW) for v in variants[i:i + bs]]
            refs += [w[2] for w in wins]
            embs.append(self.backbone.embed([w[0] for w in wins], [w[1] for w in wins]))
            if progress:
                progress(0.5 + min(i + bs, len(variants)) / (2 * len(variants)), desc=f"Running {self.name}")
        return torch.cat(embs), refs

    @torch.no_grad()
    def score(self, variants: list[str | A.Variant], progress=None) -> pd.DataFrame:
        parsed, bad = [], []
        for v in variants:
            try:
                parsed.append(v if isinstance(v, A.Variant) else A.parse_variant(v))
            except ValueError as e:
                bad.append({"variant_id": str(v), "notes": str(e)})
        if not parsed:
            return pd.DataFrame(bad)
        ann = []
        for k, v in enumerate(parsed):
            ann.append(A.annotate(v))
            if progress:
                progress((k + 1) / (2 * len(parsed)), desc="Annotating")
        ann = pd.DataFrame(ann)
        emb, refs = self.embed(parsed, progress)
        ann["ref_matches_genome"] = [r == v.ref for r, v in zip(refs, parsed)]
        out = ann.drop(columns=["notes"]).copy()
        notes = [list(n) for n in ann["notes"]]
        for i, ok in enumerate(ann["ref_matches_genome"]):
            if not ok:
                notes[i].append(f"REF does not match hg38 (reference base is {refs[i]}), so not scored")
        ref_ok = ann["ref_matches_genome"].to_numpy()
        annotated = (ann["func"].notna() & ann["cadd_phred"].notna()).to_numpy()
        for t, m in self.tasks.items():
            p = torch.sigmoid(m["head"](emb)).numpy()
            applies = ref_ok.copy()
            fused = np.full(len(p), np.nan)
            X = fusion_frame(t, ann, p, m["cfg"].get("genes"))
            if m["fusion"] is not None:
                fused = np.where(annotated, m["fusion"].predict(X), np.nan)
            if t.startswith("arch21"):
                for i, g in enumerate(X["gene"]):
                    if g == "unknown":
                        applies[i] = False
                        if f"outside the 21 ARCH genes" not in " ".join(notes[i]):
                            notes[i].append("outside the 21 ARCH genes, so no 21-gene score")
                    elif t == "arch21_noncoding" and "exonic" in str(ann["func"].iloc[i]).split(";"):
                        notes[i].append("coding variant: arch21_noncoding was trained on non-coding variants")
            p, fused = np.where(applies, p, np.nan), np.where(applies, fused, np.nan)
            col = f"{t}_{self.name}"
            out[col] = p
            out[f"{col}_plus_annotations"] = fused
            qf, qp = m["quantiles"].get("fused"), m["quantiles"]["finetuned"]
            out[f"{t}_percentile"] = [self.percentile(qf, f) if (qf is not None and np.isfinite(f))
                                      else self.percentile(qp, x) for f, x in zip(fused, p)]
        out["notes"] = ["; ".join(n) for n in notes]
        # headline score per task (fused where available, as the percentile), and a flag at the paper's threshold
        for t, m in self.tasks.items():
            col = f"{t}_{self.name}"
            fused, ft = out[f"{col}_plus_annotations"].to_numpy(), out[col].to_numpy()
            use_fused = np.isfinite(fused) & (m["quantiles"].get("fused") is not None)
            thr = self.meta[t]["threshold"]
            score = np.where(use_fused, fused, ft)
            cut = np.where(use_fused, thr.get("fused", np.nan), thr["finetuned"])
            out[f"{t}_score"] = score
            out[f"{t}_score_type"] = np.where(np.isfinite(score), np.where(use_fused, "fused", "finetuned"), "")
            out[f"{t}_flag"] = np.where(np.isfinite(score), np.where(score >= cut, "high", "-"), "")
        if self.gnomad:
            g = pd.DataFrame(G.fetch([v.id for v in parsed], progress=progress), columns=G.COLUMNS)
            for c in G.COLUMNS:
                out[c] = g[c].to_numpy()
        return pd.concat([out, pd.DataFrame(bad)], ignore_index=True) if bad else out


def _progress(frac: float, desc: str = "") -> None:
    end = "\n" if frac >= 1 else ""
    print(f"\r{desc:<20} {100 * frac:5.1f}%", end=end, file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", help="file with one variant per line (e.g. 20-32437597-T-G) or a VCF")
    ap.add_argument("--backbone", choices=BACKBONES, default="ntv3")
    ap.add_argument("--task", choices=TASKS, action="append", help="repeat to choose tasks (default: all)")
    ap.add_argument("-o", "--out", default="arch_scores.csv", help="CSV with every column (default: %(default)s)")
    ap.add_argument("--html", default=None, metavar="PATH", help="also write a readable HTML report")
    ap.add_argument("--model-path", default=None, help="local backbone weights (default: download)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-gnomad", action="store_true",
                    help="skip gnomAD v4.1 frequency, allele balance and carrier age (fetched from the public API)")
    view = ap.add_mutually_exclusive_group()
    view.add_argument("--full", action="store_true", help="print every column instead of the summary")
    view.add_argument("--quiet", action="store_true", help="print no table, only where the files went")
    a = ap.parse_args()
    tasks = tuple(a.task or TASKS)
    variants = read_variants(Path(a.variants))
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, file=sys.stderr, flush=True))
    log(f"Loading {a.backbone} (the first run downloads the backbone weights; this can take a few minutes)")
    scorer = Scorer(a.backbone, tasks, a.model_path, a.device, gnomad=not a.no_gnomad)
    log(f"Scoring {len(variants)} variant(s) on {scorer.backbone.device}")
    df = scorer.score(variants, progress=None if a.quiet else _progress)
    df.to_csv(a.out, index=False)
    if a.full:
        print(df.to_string(index=False))
    elif not a.quiet:
        print(summary_table(df, tasks, scorer.meta))
    print(f"Wrote {len(df)} variants -> {a.out}", file=sys.stderr)
    if a.html:
        try:
            from importlib.metadata import version
            ver = version("arch-scorer")
        except Exception:  # noqa: BLE001  (running from a source checkout)
            ver = ""
        write_html(df, tasks, scorer.meta, a.backbone, a.html, version=ver)
        print(f"Wrote HTML report -> {a.html}", file=sys.stderr)


if __name__ == "__main__":
    main()
