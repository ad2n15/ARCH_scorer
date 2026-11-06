#!/usr/bin/env python3
"""Per-gene AUROC heatmap for ARCH21 NC nine-model publish set.

Rows = 21 genes, columns = 9 models:
  baseline / baseline+score / baseline+embedding × LR/RF/XGB.

Genes ordered by embedding benefit (best emb+base − best baseline AUROC).

Usage:
  python sm_ch_finetune/scripts/118_plot_arch21_nc_per_gene_auroc_heatmap.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

MODEL_COLS = [
    ("baseline_lr", "Baseline\nLR"),
    ("baseline_rf", "Baseline\nRF"),
    ("baseline_xgb", "Baseline\nXGB"),
    ("api_baseline_lr", "Base+score\nLR"),
    ("api_baseline_rf", "Base+score\nRF"),
    ("api_baseline_xgb", "Base+score\nXGB"),
    ("emb_base_lr", "Base+emb\nLR"),
    ("emb_base_rf", "Base+emb\nRF"),
    ("emb_base_xgb", "Base+emb\nXGB"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--api-per-gene",
        default=(
            "results/pooled_arch21_cv5_stratify_gene_func_genefeat_onehot/"
            "per_gene_oof.csv"
        ),
    )
    p.add_argument(
        "--emb-per-gene",
        default=(
            "results/pooled_arch21_cv5_fold1_emb_baseline_emb_diff_pca64_"
            "stratify_gene_func_genefeat_onehot/per_gene_oof.csv"
        ),
    )
    p.add_argument(
        "--out-stem",
        default=(
            "results/plots/arch21_nc_publish/"
            "per_gene_auroc_heatmap_baseline_api_emb"
        ),
    )
    args = p.parse_args()

    api = pd.read_csv(PROJECT / args.api_per_gene)
    emb = pd.read_csv(PROJECT / args.emb_per_gene)
    keep_api = [m for m, _ in MODEL_COLS if not m.startswith("emb_")]
    keep_emb = [m for m, _ in MODEL_COLS if m.startswith("emb_")]
    api = api[api["model"].isin(keep_api)].copy()
    emb = emb[emb["model"].isin(keep_emb)].copy()
    both = pd.concat([api, emb], ignore_index=True)

    wide = both.pivot_table(
        index="gene", columns="model", values="auroc", aggfunc="first"
    )
    meta = (
        both.groupby("gene", as_index=True)
        .agg(n=("n", "first"), n_somatic=("n_somatic", "first"))
        .reindex(wide.index)
    )
    for m, _ in MODEL_COLS:
        if m not in wide.columns:
            raise SystemExit(f"Missing model column: {m}")
    wide = wide[[m for m, _ in MODEL_COLS]]

    base_cols = ["baseline_lr", "baseline_rf", "baseline_xgb"]
    emb_cols = ["emb_base_lr", "emb_base_rf", "emb_base_xgb"]
    wide["best_baseline"] = wide[base_cols].max(axis=1)
    wide["best_emb"] = wide[emb_cols].max(axis=1)
    wide["emb_benefit"] = wide["best_emb"] - wide["best_baseline"]
    wide = wide.join(meta)
    wide = wide.sort_values("emb_benefit", ascending=False)

    mat = wide[[m for m, _ in MODEL_COLS]].to_numpy(float)
    row_labels = [
        f"{g}  (n={int(n)}, som={int(s)})"
        for g, n, s in zip(wide.index, wide["n"], wide["n_somatic"])
    ]
    col_labels = [lab for _, lab in MODEL_COLS]

    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.35, vmax=0.95)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("AUROC")

    y_tick_kw = {"fontsize": 9, "fontweight": "bold"}
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(
        col_labels,
        fontsize=9,
        fontweight="bold",
        rotation=60,
        ha="left",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, **y_tick_kw)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")

    for x in (2.5, 5.5):
        ax.axvline(x, color="white", linewidth=2.0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            text_color = "black" if 0.45 < val < 0.78 else "white"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color=text_color,
            )

    ax.set_title(
        "ARCH21 MANE noncoding — per-gene OOF AUROC\n"
        "(stratify_gene_func + gene onehot; genes ordered by "
        "emb benefit = best base+emb − best baseline)",
        pad=14,
    )
    star_mask = wide["emb_base_xgb"].to_numpy(float) > wide["best_baseline"].to_numpy(
        float
    )
    for i, win in enumerate(star_mask):
        if win:
            ax.scatter(
                [-0.7],
                [i],
                marker="*",
                s=90,
                color="#2ca02c",
                clip_on=False,
                zorder=5,
            )
    star_legend = Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="#2ca02c",
        markersize=14,
        linestyle="None",
        label=(
            "Gene with higher per-gene OOF AUROC for Base+emb XGB than the "
            "best baseline model (LR, RF, or XGB)"
        ),
    )
    fig.legend(
        handles=[star_legend],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        prop={"size": 9, "weight": "bold"},
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    out = PROJECT / args.out_stem
    out.parent.mkdir(parents=True, exist_ok=True)
    png_path = out.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {png_path}")

    # TrueType text in PDF (fonttype 42) — editable in Illustrator/Inkscape.
    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        pdf_path = out.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
        print(f"Saved -> {pdf_path} (editable text)")
        svg_path = out.with_name(out.name + "_editable").with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", format="svg")
        print(f"Saved -> {svg_path} (vector text; best for layout edits)")
    plt.close(fig)

    ordered = wide.reset_index()[
        ["gene", "n", "n_somatic", "best_baseline", "best_emb", "emb_benefit"]
        + [m for m, _ in MODEL_COLS]
    ]
    ordered["star_emb_xgb_vs_best_baseline"] = star_mask
    csv_path = out.with_name("per_gene_auroc_ordered_by_emb_benefit.csv")
    ordered.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")
    print("Genes starred (Base+emb XGB > best baseline):")
    print(", ".join(ordered.loc[star_mask, "gene"].tolist()) or "(none)")


if __name__ == "__main__":
    main()
