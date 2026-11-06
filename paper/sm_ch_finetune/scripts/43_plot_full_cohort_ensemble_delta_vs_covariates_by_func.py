#!/usr/bin/env python3
"""Stratified Δ AUROC: ensemble (cov+finetune) vs covariates-only reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

STRATA = ["splicing", "ncRNA_exonic", "ncRNA_intronic", "exonic", "intronic", "UTR5", "UTR3"]
PATHS = {
    "AlphaGenome": "results/finetune_covariates_full_age_filtered.json",
    "NTv3": "results/ntv3_full_finetune_covariates.json",
    "Borzoi": "results/borzoi_full_finetune_covariates.json",
}
COLORS = ["#4C78A8", "#54A24B", "#E45756"]


def load_by_func(path: Path, model_prefix: str) -> dict:
    d = json.loads(path.read_text())
    keys = [k for k in d["by_func"] if str(k).startswith(model_prefix)]
    if not keys:
        raise SystemExit(f"No model starting with {model_prefix!r} in {path}")
    out = {}
    for r in d["by_func"][keys[0]]["rows"]:
        if r.get("skipped"):
            continue
        out[r["func"]] = {"n": int(r["n"]), "auroc": float(r["auroc"])}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/plots/full_cohort_finetune"))
    p.add_argument(
        "--cov-json",
        type=Path,
        default=Path("results/finetune_covariates_full_age_filtered.json"),
        help="JSON used for covariates-only reference (AG file by default)",
    )
    args = p.parse_args()

    cov = load_by_func(args.cov_json, "covariates_only")
    backbones = list(PATHS.keys())
    ens = {name: load_by_func(Path(path), "ensemble") for name, path in PATHS.items()}

    # Top of plot = first stratum (splicing)
    labels = [f"{s}  (n={cov[s]['n']:,})" for s in STRATA]
    y = np.arange(len(STRATA))[::-1]  # splicing at top
    height = 0.26
    offsets = (np.arange(len(backbones)) - (len(backbones) - 1) / 2) * height

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.axvline(0.0, color="#333333", lw=1.0, zorder=1)

    for off, name, color in zip(offsets, backbones, COLORS):
        deltas = [ens[name][s]["auroc"] - cov[s]["auroc"] for s in STRATA]
        bars = ax.barh(
            y + off,
            deltas,
            height=height * 0.92,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        for b, d in zip(bars, deltas):
            ax.text(
                d + (0.004 if d >= 0 else -0.004),
                b.get_y() + b.get_height() / 2,
                f"{d:+.2f}",
                ha="left" if d >= 0 else "right",
                va="center",
                fontsize=7,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δ AUROC vs covariates only")
    ax.set_title(
        "Full cohort — ensemble gain over covariates by functional class\n"
        "zero = covariates-only AUROC in each stratum (held-out test)",
        fontsize=11,
        pad=10,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E0E0E0", lw=0.6)

    ref = ", ".join(f"{s}={cov[s]['auroc']:.3f}" for s in STRATA)
    ax.text(
        0.0,
        -0.12,
        f"Covariates-only AUROC: {ref}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "full_cohort_ensemble_delta_vs_covariates_by_func.png"
    pdf = args.out_dir / "full_cohort_ensemble_delta_vs_covariates_by_func.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print("Reference covariates-only AUROC by stratum:")
    for s in STRATA:
        print(f"  {s}: {cov[s]['auroc']:.4f} (n={cov[s]['n']})")
        for name in backbones:
            d = ens[name][s]["auroc"] - cov[s]["auroc"]
            print(f"    {name}: ensemble={ens[name][s]['auroc']:.4f}  Δ={d:+.4f}")


if __name__ == "__main__":
    main()
