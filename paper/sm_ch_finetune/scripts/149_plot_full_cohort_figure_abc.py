#!/usr/bin/env python3
"""Full-cohort finetune figure panels a–c (bold + editable PDF).

  a) AUROC by backbone (covariates / finetune / ensemble)
  b) Ensemble AUROC by Func.refGene
  c) Spearman concordance heatmap

  python3 sm_ch_finetune/scripts/149_plot_full_cohort_figure_abc.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT = Path(__file__).resolve().parents[1]

BACKBONES = ["AlphaGenome", "NTv3", "Borzoi"]
SERIES_A = [
    ("Covariates only", [0.562, 0.562, 0.562], "#4C78A8"),
    ("Finetune only", [0.569, 0.579, 0.559], "#54A24B"),
    ("Ensemble (finetune + covariates)", [0.584, 0.588, 0.583], "#E45756"),
]

ROWS_B = [
    ("splicing", 111, 0.7669354839, 0.7423387097, 0.7548387097),
    ("ncRNA_exonic", 123, 0.6215659341, 0.6761675824, 0.6656936813),
    ("ncRNA_intronic", 144, 0.57343107, 0.5882201646, 0.5977366255),
    ("exonic", 12423, 0.5937446654, 0.5959121999, 0.5901834425),
    ("intronic", 4576, 0.560178328, 0.5668357572, 0.5605562486),
    ("UTR5", 683, 0.5516316396, 0.5596568236, 0.5749700718),
    ("UTR3", 415, 0.5065368879, 0.5292801432, 0.5074159315),
]
BACKBONES_B = [
    ("AlphaGenome", 2, "#4C78A8"),
    ("NTv3", 3, "#54A24B"),
    ("Borzoi", 4, "#E45756"),
]

MODELS = ("covariates", "alphagenome", "ntv3", "borzoi")
MODEL_LABELS = {
    "covariates": "Covariates",
    "alphagenome": "AlphaGenome",
    "ntv3": "NTv3",
    "borzoi": "Borzoi",
}


def _bold_axis(ax: plt.Axes) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _spearman_matrix(pairwise: dict[str, float]) -> pd.DataFrame:
    m = pd.DataFrame(np.eye(len(MODELS)), index=MODELS, columns=MODELS, dtype=float)
    for key, val in pairwise.items():
        a, b = key.split("__", 1)
        m.loc[a, b] = val
        m.loc[b, a] = val
    m.index = [MODEL_LABELS[i] for i in m.index]
    m.columns = [MODEL_LABELS[c] for c in m.columns]
    return m


def plot_panel_a(ax: plt.Axes, *, compact: bool = False) -> None:
    x = np.arange(len(BACKBONES))
    width = 0.24
    offsets = (np.arange(len(SERIES_A)) - (len(SERIES_A) - 1) / 2) * width
    ax.axhline(0.5, color="#888888", ls="--", lw=0.8, zorder=0)
    for off, (name, vals, color) in zip(offsets, SERIES_A):
        bars = ax.bar(
            x + off,
            vals,
            width=width * 0.92,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.0025,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5 if compact else 7.5,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(BACKBONES, fontweight="bold")
    ax.set_ylabel("AUROC (held-out test)")
    ax.set_ylim(0.52, 0.61)
    title = (
        "SM vs ARCH — overall test\n"
        "n = 18,620 · CH ≈ 22.8%"
        if compact
        else (
            "Full age-filtered cohort — SM vs ARCH\n"
            "train n = 335,156 · test n = 18,620 · CH ≈ 22.8%"
        )
    )
    ax.set_title(title, fontsize=10 if compact else 11, fontweight="bold", pad=8)
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=7 if compact else 8.5,
        prop={"weight": "bold"},
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E0E0E0", lw=0.6)
    _bold_axis(ax)


def plot_panel_b(ax: plt.Axes, *, compact: bool = False) -> None:
    rows = list(ROWS_B)
    labels = [f"{name}\n(n={n:,})" for name, n, *_ in rows]
    x = np.arange(len(rows))
    width = 0.26
    offsets = (np.arange(len(BACKBONES_B)) - (len(BACKBONES_B) - 1) / 2) * width
    ax.axhline(0.5, color="#888888", ls="--", lw=0.8, zorder=0)
    for off, (name, col_i, color) in zip(offsets, BACKBONES_B):
        vals = [r[col_i] for r in rows]
        bars = ax.bar(
            x + off,
            vals,
            width=width * 0.92,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.006,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5 if compact else 6.5,
                fontweight="bold",
                rotation=90,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7 if compact else 8.5, fontweight="bold")
    ax.set_ylabel("AUROC (held-out test)")
    ax.set_ylim(0.40, 0.84)
    ax.set_title(
        "Ensemble (covariates + finetune) by Func.refGene",
        fontsize=10 if compact else 11,
        fontweight="bold",
        pad=8,
    )
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=7 if compact else 8.5,
        prop={"weight": "bold"},
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E0E0E0", lw=0.6)
    _bold_axis(ax)


def plot_panel_c(ax: plt.Axes, mat: pd.DataFrame, *, compact: bool = False) -> None:
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="rocket_r",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.75},
        annot_kws={"fontsize": 8 if compact else 9, "weight": "bold"},
    )
    ax.set_title(
        "Spearman ρ of P(CH) ranks",
        fontsize=10 if compact else 11,
        fontweight="bold",
        pad=8,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight="bold")
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=8)
        for t in cbar.ax.get_yticklabels():
            t.set_fontweight("bold")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    print(f"Wrote {path.with_suffix('.png')}")
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(path.with_suffix(".pdf"), format="pdf")
    print(f"Wrote {path.with_suffix('.pdf')} (editable text)")


def save_standalone_panels(
    out_ft: Path,
    out_agree: Path,
    agreement_json: Path,
) -> None:
    d = json.loads(agreement_json.read_text())
    mat = _spearman_matrix(d["spearman"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plot_panel_a(ax, compact=False)
    fig.tight_layout()
    _save(fig, out_ft / "full_cohort_auroc_by_backbone")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    plot_panel_b(ax, compact=False)
    fig.tight_layout()
    _save(fig, out_ft / "full_cohort_ensemble_auroc_by_func_bar")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    plot_panel_c(ax, mat, compact=False)
    fig.tight_layout()
    _save(fig, out_agree / "concordance_spearman_heatmap")
    plt.close(fig)


def save_combined(out_ft: Path, agreement_json: Path) -> None:
    d = json.loads(agreement_json.read_text())
    mat = _spearman_matrix(d["spearman"])

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.12],
        width_ratios=[1.05, 0.95],
        hspace=0.42,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])

    plot_panel_a(ax_a, compact=True)
    plot_panel_b(ax_b, compact=True)
    plot_panel_c(ax_c, mat, compact=True)

    _panel_label(ax_a, "a)")
    _panel_label(ax_b, "b)")
    _panel_label(ax_c, "c)")

    _save(fig, out_ft / "full_cohort_finetune_performance_concordance_abc")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-ft",
        type=Path,
        default=PROJECT / "results/plots/full_cohort_finetune",
    )
    p.add_argument(
        "--out-agree",
        type=Path,
        default=PROJECT / "results/plots/model_agreement",
    )
    p.add_argument(
        "--agreement-json",
        type=Path,
        default=PROJECT / "results/full_model_agreement.json",
    )
    args = p.parse_args()
    save_standalone_panels(args.out_ft, args.out_agree, args.agreement_json)
    save_combined(args.out_ft, args.agreement_json)


if __name__ == "__main__":
    main()
