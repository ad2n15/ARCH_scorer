#!/usr/bin/env python3
"""Per-gene Sens/Spec for all 9 ARCH21 NC publish models (1 row per gene).

Models: baseline / baseline+API / baseline+emb × LR/RF/XGB.
Each model uses its own global prevalence-matched top-k threshold from OOF.
Also writes gene × function long/wide CSVs for reference.

Usage:
  python sm_ch_finetune/scripts/120_arch21_nc_gene_x_func_sens_spec.py
  python sm_ch_finetune/scripts/120_arch21_nc_gene_x_func_sens_spec.py --write-md-snippet
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]

FX_ORDER = [
    "intron",
    "UTR3",
    "UTR5",
    "promoter_200_upstream",
    "splicing",
]
FX_LAB = {
    "intron": "Intron",
    "UTR3": "UTR3",
    "UTR5": "UTR5",
    "promoter_200_upstream": "Promoter",
    "splicing": "Splicing",
}

MODELS = [
    ("baseline", "LR", "baseline_lr", "p_baseline_lr", "api"),
    ("baseline", "RF", "baseline_rf", "p_baseline_rf", "api"),
    ("baseline", "XGB", "baseline_xgb", "p_baseline_xgb", "api"),
    ("baseline + API", "LR", "api_baseline_lr", "p_api_baseline_lr", "api"),
    ("baseline + API", "RF", "api_baseline_rf", "p_api_baseline_rf", "api"),
    ("baseline + API", "XGB", "api_baseline_xgb", "p_api_baseline_xgb", "api"),
    ("baseline + embedding", "LR", "emb_base_lr", "p_emb_base_lr", "emb"),
    ("baseline + embedding", "RF", "emb_base_rf", "p_emb_base_rf", "emb"),
    ("baseline + embedding", "XGB", "emb_base_xgb", "p_emb_base_xgb", "emb"),
]

API_DIR = "results/pooled_arch21_cv5_stratify_gene_func_genefeat_onehot"
EMB_DIR = (
    "results/pooled_arch21_cv5_fold1_emb_baseline_emb_diff_pca64_"
    "stratify_gene_func_genefeat_onehot"
)


def _effects() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(
        "u88", PROJECT / "scripts" / "88_univariate_pooled_noncoding.py"
    )
    u88 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(u88)
    return u88._attach_annovar(u88._load_cohort())[
        ["variant_id", "annotated_effect"]
    ]


def _cell_auroc(y: np.ndarray, score: np.ndarray, min_n: int = 10, min_pos: int = 2) -> float:
    n = int(len(y))
    n_pos = int(y.sum())
    if n < min_n or n_pos < min_pos or n_pos == n:
        return float("nan")
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def _md_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    float_cols = float_cols or [
        c for c in df.columns if c not in ("Gene", "N", "somatic_n", "n", "n_som")
    ]
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|"
        + "|".join(
            ["------" if c in ("Gene",) else "------:" for c in cols]
        )
        + "|",
    ]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "Gene":
                cells.append(str(v))
            elif c in ("N", "somatic_n", "n", "n_som"):
                cells.append(str(int(v)))
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(f"{float(v):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-dir", default=API_DIR)
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument(
        "--gene-order-csv",
        default=(
            "results/plots/arch21_nc_publish/"
            "per_gene_auroc_ordered_by_emb_benefit.csv"
        ),
    )
    ap.add_argument("--out-dir", default="results/plots/arch21_nc_publish")
    ap.add_argument("--write-md-snippet", action="store_true")
    args = ap.parse_args()

    api_pred = pd.read_parquet(PROJECT / args.api_dir / "oof_predictions.parquet")
    emb_pred = pd.read_parquet(PROJECT / args.emb_dir / "oof_predictions.parquet")
    api_sum = pd.read_csv(PROJECT / args.api_dir / "oof_summary.csv")
    emb_sum = pd.read_csv(PROJECT / args.emb_dir / "oof_summary.csv")
    thr_map = {
        **dict(zip(api_sum["model"], api_sum["pm_threshold"])),
        **dict(zip(emb_sum["model"], emb_sum["pm_threshold"])),
    }

    keep_api = ["variant_id", "gene", "y"] + [
        c for _, _, _, c, src in MODELS if src == "api"
    ]
    keep_emb = ["variant_id"] + [
        c for _, _, _, c, src in MODELS if src == "emb"
    ]
    df = api_pred[keep_api].merge(emb_pred[keep_emb], on="variant_id", how="inner")
    effects = _effects()
    df = df.merge(effects, on="variant_id", how="left", validate="one_to_one")

    order_path = PROJECT / args.gene_order_csv
    genes = (
        pd.read_csv(order_path)["gene"].astype(str).tolist()
        if order_path.is_file()
        else sorted(df["gene"].unique())
    )

    # ---- 1 row per gene × 9 models (overall within gene) ----
    model_cols = [f"{feat} {clf}" for feat, clf, *_ in MODELS]
    sens_rows: list[dict] = []
    spec_rows: list[dict] = []
    for g in genes:
        sub = df[df["gene"] == g]
        y = sub["y"].to_numpy(dtype=int)
        srow: dict = {"Gene": g, "N": len(sub), "somatic_n": int(y.sum())}
        prow: dict = {"Gene": g, "N": len(sub), "somatic_n": int(y.sum())}
        for feat, clf, mkey, scol, _ in MODELS:
            col = f"{feat} {clf}"
            thr = float(thr_map[mkey])
            pp = sub[scol].to_numpy(dtype=float) >= thr
            tp = int(((y == 1) & pp).sum())
            fn = int(((y == 1) & ~pp).sum())
            tn = int(((y == 0) & ~pp).sum())
            fp = int(((y == 0) & pp).sum())
            srow[col] = (tp / (tp + fn)) if (tp + fn) else np.nan
            prow[col] = (tn / (tn + fp)) if (tn + fp) else np.nan
        sens_rows.append(srow)
        spec_rows.append(prow)

    sens = pd.DataFrame(sens_rows)
    spec = pd.DataFrame(spec_rows)
    out = PROJECT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sens_path = out / "per_gene_sensitivity_all9models.csv"
    spec_path = out / "per_gene_specificity_all9models.csv"
    sens.to_csv(sens_path, index=False)
    spec.to_csv(spec_path, index=False)
    print(f"Saved -> {sens_path}")
    print(f"Saved -> {spec_path}")

    # ---- gene × function long (all 9) for reference ----
    fx_rows: list[dict] = []
    for feat, clf, mkey, scol, _ in MODELS:
        thr = float(thr_map[mkey])
        for g in genes:
            for fx in FX_ORDER:
                sub = df[(df["gene"] == g) & (df["annotated_effect"] == fx)]
                if sub.empty:
                    fx_rows.append(
                        dict(
                            features=feat,
                            model=clf,
                            model_key=mkey,
                            gene=g,
                            effect=fx,
                            N=0,
                            somatic_n=0,
                            germline_n=0,
                            auroc=np.nan,
                            sens=np.nan,
                            spec=np.nan,
                            threshold=thr,
                        )
                    )
                    continue
                y = sub["y"].to_numpy(dtype=int)
                score = sub[scol].to_numpy(dtype=float)
                pp = score >= thr
                tp = int(((y == 1) & pp).sum())
                fn = int(((y == 1) & ~pp).sum())
                tn = int(((y == 0) & ~pp).sum())
                fp = int(((y == 0) & pp).sum())
                fx_rows.append(
                    dict(
                        features=feat,
                        model=clf,
                        model_key=mkey,
                        gene=g,
                        effect=fx,
                        N=len(sub),
                        somatic_n=int(y.sum()),
                        germline_n=int((y == 0).sum()),
                        auroc=_cell_auroc(y, score),
                        sens=(tp / (tp + fn)) if (tp + fn) else np.nan,
                        spec=(tn / (tn + fp)) if (tn + fp) else np.nan,
                        threshold=thr,
                    )
                )
    long = pd.DataFrame(fx_rows)
    long_path = out / "per_gene_x_func_sens_spec_all9models.csv"
    long.to_csv(long_path, index=False)
    print(f"Saved -> {long_path}")

    # emb_base_xgb gene × function wide (legacy)
    sub = long[long["model_key"] == "emb_base_xgb"]
    for metric, name in (("sens", "sensitivity"), ("spec", "specificity")):
        w = sub.pivot(index="gene", columns="effect", values=metric)
        w = w.reindex(index=genes, columns=FX_ORDER)
        w.columns = [FX_LAB[c] for c in w.columns]
        w.to_csv(out / f"per_gene_x_func_{name}_emb_base_xgb.csv")

    if args.write_md_snippet:
        snip = out / "per_gene_sens_spec_all9models.md"
        snip.write_text(
            "#### Sensitivity — all 9 models (1 row / gene)\n\n"
            + _md_table(sens)
            + "\n\n#### Specificity — all 9 models (1 row / gene)\n\n"
            + _md_table(spec)
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved -> {snip}")


if __name__ == "__main__":
    main()
