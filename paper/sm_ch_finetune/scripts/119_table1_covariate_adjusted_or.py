#!/usr/bin/env python3
"""Table 1 / Fig. 2 and Supplementary Table S4: covariate-adjusted AlphaGenome CD34+ CMP odds ratios.

Rebuilt in Nov 2026 (the original script was lost). Reproduces the stored outputs to ~1e-8.

Table 1b: one logistic regression per feature,
    y ~ feature_z + cadd_z + SBS6 (ref C>T) + Func.refGene (first term) + gene (21 fixed effects)
    (CADD-PHRED itself: y ~ cadd_z + covariates). OR per 1 SD of the feature; BH q across features.
Table 1c: one joint model with CADD-PHRED and all ten track scores, same covariates.

Fit: scikit-learn LogisticRegression (lbfgs, C = 1, class_weight="balanced"); Wald SE from the
class-weighted Fisher information (`_logit_wald` in 24_ch_genes_gvs_publication_analysis.py).
z-scores use the population SD; categorical references are C>T and the alphabetically first level.

Strata: noncoding_all (n = 71,544; 1,914 somatic), noncoding_no_splice (canonical +/-2 bp intronic
splice positions removed; n = 71,433), coding (n = 6,037).

  cd paper/sm_ch_finetune
  python3 scripts/119_table1_covariate_adjusted_or.py
"""
from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from data.variant_table import mutation_type_feature  # noqa: E402

OUT = PROJECT / "results/arch_mane_table1_covariate_adjusted"
FEATS = ["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3", "DNASE", "RNA_SEQ",
         "SPLICE_JUNC", "SPLICE_USAGE"]


def _load(name: str, fn: str):
    spec = spec_from_file_location(name, PROJECT / "scripts" / fn)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load("u117", "117_univariate_pooled_coding_noncoding.py")
P = U._pub


def arm(kind: str) -> pd.DataFrame:
    df = U._load_arm(kind)
    df, _ = U._attach_api_arm(df, kind)
    df = df.rename(columns=P.SHORT)
    df = U._attach_cadd_arm(df, kind)
    df = df.loc[df.cadd_phred.notna()].drop_duplicates("variant_id")
    df = U._annotate_noncoding(df) if kind == "noncoding" else U._annotate_coding(df)
    df["Func.refGene"] = df["Func.refGene"].fillna("unknown").astype(str).str.split(";").str[0]
    df["mt"] = [mutation_type_feature(str(r), str(a), "sbs6") for r, a in zip(df.ref, df.alt)]
    return df.reset_index(drop=True)


def zs(s: pd.Series) -> np.ndarray:
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy(float)


def dummies(d: pd.DataFrame) -> np.ndarray:
    out = []
    for col, ref in [("mt", "C>T"), ("Func.refGene", None), ("gene", None)]:
        s = d[col].astype(str)
        levels = sorted(s.unique())
        r = ref or levels[0]
        out += [(s == lv).to_numpy(float) for lv in levels if lv != r]
    return np.column_stack(out)


def _row(d, o, lo, hi, p, tag):
    return {"n": len(d), "n_somatic": int(d.y.sum()), "somatic_rate": d.y.mean(), f"or_per_sd_{tag}": o,
            f"or_ci_lo_{tag}": lo, f"or_ci_hi_{tag}": hi, f"p_value_{tag}": p}


def table1b(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in ["cadd_phred"] + FEATS:
        cont = [f] if f == "cadd_phred" else [f, "cadd_phred"]
        d = df.dropna(subset=cont)
        X = np.column_stack([zs(d[c]) for c in cont] + [dummies(d)])
        *_, p, _, _, o, lo, hi = P._logit_wald(X, d.y.to_numpy(int))
        rows.append({"feature": f, **_row(d, o[0], lo[0], hi[0], p[0], "adj")})
    t = pd.DataFrame(rows)
    t["q_value_bh_adj"] = U._bh_fdr(t.p_value_adj)
    return t


def table1c(df: pd.DataFrame) -> pd.DataFrame:
    cont = ["cadd_phred"] + FEATS
    d = df.dropna(subset=cont)
    X = np.column_stack([zs(d[c]) for c in cont] + [dummies(d)])
    *_, p, _, _, o, lo, hi = P._logit_wald(X, d.y.to_numpy(int))
    t = pd.DataFrame([{"feature": f, **_row(d, o[i], lo[i], hi[i], p[i], "joint")} for i, f in enumerate(cont)])
    t["q_value_bh_joint"] = U._bh_fdr(t.p_value_joint)
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nc, co = arm("noncoding"), arm("coding")
    strata = {"noncoding_all": nc, "noncoding_no_splice": nc[nc.annotated_effect != "splicing"], "coding": co}
    summary = {}
    for k, d in strata.items():
        table1b(d).to_csv(OUT / f"table1b_covariate_adjusted_{k}.csv", index=False)
        table1c(d).to_csv(OUT / f"table1c_joint_all_features_{k}.csv", index=False)
        summary[k] = {"n": len(d), "n_somatic": int(d.y.sum())}
        print(k, summary[k])
    (OUT / "table1_rebuild_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
