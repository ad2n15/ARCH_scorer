"""Stratified evaluation by categorical columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.classification_metrics import prevalence_matched_table, ranking_metrics


def eval_by_stratum(
    test: pd.DataFrame,
    prob,
    *,
    stratify_col: str,
    min_n: int,
) -> list[dict]:
    """Per-stratum ranking metrics using global model predictions on test."""
    if stratify_col not in test.columns:
        return []
    prob = np.asarray(prob)
    test = test.reset_index(drop=True)
    rows: list[dict] = []
    for stratum, g in test.groupby(stratify_col, dropna=False):
        name = str(stratum) if pd.notna(stratum) else "."
        n = len(g)
        if n < min_n:
            rows.append({
                stratify_col: name,
                "n": int(n),
                "skipped": True,
                "reason": f"n < {min_n}",
            })
            continue
        y = g["label"].astype(int).values
        p = prob[g.index.values]
        if len(np.unique(y)) < 2:
            rows.append({
                stratify_col: name,
                "n": int(n),
                "n_ch": int(y.sum()),
                "n_sm": int(len(y) - y.sum()),
                "skipped": True,
                "reason": "single class in stratum",
            })
            continue
        rank = ranking_metrics(y, p)
        prev = prevalence_matched_table(y, p)
        rows.append({
            stratify_col: name,
            "n": int(n),
            "n_ch": int(y.sum()),
            "n_sm": int(len(y) - y.sum()),
            "ch_rate": float(y.mean()),
            "skipped": False,
            **rank,
            "prevalence_matched": prev,
        })
    rows.sort(key=lambda r: r.get("auroc", 0), reverse=True)
    return rows


def eval_by_genes(
    test: pd.DataFrame,
    prob,
    *,
    gene_col: str,
    genes: list[str],
    min_n: int,
) -> list[dict]:
    """Per-gene metrics: filter test to each gene in `genes`."""
    if gene_col not in test.columns:
        return []
    prob = np.asarray(prob)
    test = test.reset_index(drop=True)
    rows: list[dict] = []
    for gene in genes:
        g = test[test[gene_col].astype(str) == str(gene)]
        n = len(g)
        if n < min_n:
            rows.append({
                "gene": str(gene),
                "n": int(n),
                "skipped": True,
                "reason": f"n < {min_n}",
            })
            continue
        y = g["label"].astype(int).values
        p = prob[g.index.values]
        if len(np.unique(y)) < 2:
            rows.append({
                "gene": str(gene),
                "n": int(n),
                "n_ch": int(y.sum()),
                "n_sm": int(len(y) - y.sum()),
                "skipped": True,
                "reason": "single class in stratum",
            })
            continue
        rank = ranking_metrics(y, p)
        prev = prevalence_matched_table(y, p)
        rows.append({
            "gene": str(gene),
            "n": int(n),
            "n_ch": int(y.sum()),
            "n_sm": int(len(y) - y.sum()),
            "ch_rate": float(y.mean()),
            "skipped": False,
            **rank,
            "prevalence_matched": prev,
        })
    return rows


def print_stratum_table(
    model: str,
    stratify_col: str,
    rows: list[dict],
    *,
    min_n: int,
) -> None:
    print(f"\n{model} by {stratify_col} (global model, test n>={min_n}):")
    label_key = "gene" if stratify_col == "gene" or all("gene" in r for r in rows) else stratify_col
    for row in rows:
        name = row.get(label_key, row.get(stratify_col, "?"))
        if row.get("skipped"):
            print(f"  {name}: n={row['n']} (skipped: {row['reason']})")
        else:
            print(
                f"  {name}: AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} "
                f"n={row['n']} (CH={row['n_ch']}, SM={row['n_sm']}, ch_rate={row['ch_rate']:.3f})"
            )


def stratum_payload(model: str, stratify_col: str, rows: list[dict], *, min_n: int) -> dict:
    label_key = "gene" if stratify_col == "gene" or (rows and "gene" in rows[0]) else stratify_col
    return {
        "model": model,
        "stratify_col": stratify_col,
        "min_stratum_n": min_n,
        "rows": [{k: v for k, v in row.items() if k != "prevalence_matched"} for row in rows],
        "prevalence_matched": {
            row[label_key]: row["prevalence_matched"]
            for row in rows
            if not row.get("skipped")
        },
    }
