#!/usr/bin/env python3
"""ROC + germline/somatic score distributions for baseline vs emb+baseline XGB.

Per-gene figures (default ASXL1, PPM1D, TP53): panels a–c = ROC, baseline XGB, emb+baseline XGB.

Usage:
  python sm_ch_finetune/scripts/127_plot_emb_baseline_xgb_discrimination.py
  python sm_ch_finetune/scripts/127_plot_emb_baseline_xgb_discrimination.py --genes ASXL1,PPM1D,TP53
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT = Path(__file__).resolve().parents[1]

API_DIR = "results/pooled_arch21_cv5_stratify_gene_func_genefeat_onehot"
EMB_DIR = (
    "results/pooled_arch21_cv5_fold1_emb_baseline_emb_diff_pca64_"
    "stratify_gene_func_genefeat_onehot"
)

COLORS = {
    "baseline_xgb": "#2A6F6F",
    "emb_base_xgb": "#C45C26",
}
LABELS = {
    "baseline_xgb": "Baseline XGB",
    "emb_base_xgb": "Emb + baseline XGB",
}


def _load() -> pd.DataFrame:
    api = pd.read_parquet(PROJECT / API_DIR / "oof_predictions.parquet")
    emb = pd.read_parquet(PROJECT / EMB_DIR / "oof_predictions.parquet")
    return api[["variant_id", "gene", "y", "p_baseline_xgb"]].merge(
        emb[["variant_id", "p_emb_base_xgb"]], on="variant_id", how="inner"
    )


def _local_pm_thr(score: np.ndarray, y: np.ndarray) -> float:
    k = int(y.sum())
    if k <= 0:
        return float("nan")
    return float(np.partition(np.asarray(score, dtype=float), -k)[-k])


def _bold_axis(ax: plt.Axes) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.10,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _plot_one(df: pd.DataFrame, title: str, out_stem: Path) -> None:
    y = df["y"].to_numpy(dtype=int)
    n_germ = int((y == 0).sum())
    n_som = int(y.sum())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    # Reserve bottom band for legend (figure coords 0–~0.28); axes sit above it.
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.30, top=0.76, wspace=0.38)
    ax_roc, ax_b, ax_e = axes

    roc_legend: list[str] = []
    for key in ("baseline_xgb", "emb_base_xgb"):
        score = df[f"p_{key}"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, score)
        auc = float(roc_auc_score(y, score))
        ax_roc.plot(fpr, tpr, color=COLORS[key], lw=2.2)
        roc_legend.append(f"{LABELS[key]} (AUROC {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], ls="--", color="#9A9A9A", lw=1.0)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_aspect("equal", adjustable="box")

    bins = np.linspace(0, 1, 41)
    pm_thr: dict[str, float] = {}
    for ax, key in ((ax_b, "baseline_xgb"), (ax_e, "emb_base_xgb")):
        score = df[f"p_{key}"].to_numpy(dtype=float)
        ax.hist(
            score[y == 0],
            bins=bins,
            density=True,
            alpha=0.55,
            color="#5B6B73",
            edgecolor="none",
        )
        ax.hist(
            score[y == 1],
            bins=bins,
            density=True,
            alpha=0.55,
            color=COLORS[key],
            edgecolor="none",
        )
        t = _local_pm_thr(score, y)
        pm_thr[key] = t
        ax.axvline(t, color="#222222", ls="--", lw=1.1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("OOF predicted P(somatic)")
        ax.set_ylabel("Density")
        ax.set_title(LABELS[key])

    for ax, letter in zip(axes, ("a)", "b)", "c)")):
        _panel_label(ax, letter)
        _bold_axis(ax)

    fig.suptitle(
        f"{title}\nARCH21 MANE noncoding · OOF · germline vs somatic",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    t_base = pm_thr["baseline_xgb"]
    t_emb = pm_thr["emb_base_xgb"]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["baseline_xgb"],
            lw=2.2,
            label=roc_legend[0],
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["emb_base_xgb"],
            lw=2.2,
            label=roc_legend[1],
        ),
        Patch(
            facecolor="#5B6B73",
            alpha=0.55,
            edgecolor="none",
            label=f"Germline (n={n_germ})",
        ),
        Patch(
            facecolor=COLORS["emb_base_xgb"],
            alpha=0.55,
            edgecolor="none",
            label=f"Somatic (n={n_som})",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            ls="--",
            lw=1.1,
            label=f"PM thr, baseline ({t_base:.2f})",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            ls="--",
            lw=1.1,
            label=f"PM thr, emb+baseline ({t_emb:.2f})",
        ),
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        prop={"weight": "bold"},
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    print(f"wrote {out_stem.with_suffix('.png')}")

    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        pdf_path = out_stem.with_suffix(".pdf")
        fig.savefig(pdf_path, format="pdf")
        print(f"wrote {pdf_path} (editable text)")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default="results/plots/arch21_nc_publish",
    )
    ap.add_argument(
        "--genes",
        default="ASXL1,PPM1D,TP53",
        help="Comma-separated genes (default: ASXL1,PPM1D,TP53)",
    )
    ap.add_argument(
        "--include-pooled",
        action="store_true",
        help="Also write pooled discrimination figure",
    )
    args = ap.parse_args()
    df = _load()
    out = PROJECT / args.out_dir
    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]

    if args.include_pooled:
        _plot_one(
            df,
            title=f"Pooled (n={len(df):,} · somatic={int(df.y.sum()):,})",
            out_stem=out / "discrimination_pooled_baseline_vs_emb_base_xgb",
        )

    for gene in genes:
        sub = df[df["gene"].astype(str).str.upper() == gene].copy()
        if sub.empty:
            raise SystemExit(f"No rows for gene {gene}")
        _plot_one(
            sub,
            title=f"{gene} (n={len(sub):,} · somatic={int(sub.y.sum()):,})",
            out_stem=out / f"discrimination_{gene.lower()}_baseline_vs_emb_base_xgb",
        )


if __name__ == "__main__":
    main()
