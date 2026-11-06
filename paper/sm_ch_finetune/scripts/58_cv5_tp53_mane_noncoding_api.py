#!/usr/bin/env python3
"""TP53 MANE noncoding 5-fold CV: gnomAD WGS germline vs COSMIC NCV somatic.

API tracks only (10 CD34+ AlphaGenome features). Cohort from
data/tp53_mane_noncoding/ + results/track_scores_tp53_mane_noncoding/.

Example:
  ./run_binemo_AMD.sh sm_ch_finetune/scripts/58_cv5_tp53_mane_noncoding_api.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from eval.classification_metrics import (
    attach_cosmic_haem_flags,
    haem_ge3_membership,
    prevalence_matched_table,
    ranking_metrics,
    source_sensitivity_pm,
)
from paths import project_subpath
from scoring.track_classifiers import make_track_classifier

META = {"variant_id", "label", "gene"}
SHORT = {
    "RNA_SEQ__CL_0001059_polyA_plus_RNA_seq__GeneMaskLFC": "RNA_SEQ",
    "SPLICE_SITE_USAGE__usage_CL_0001059_polyA_plus_RNA_seq__GeneMaskSplicing": "SPLICE_USAGE",
    "SPLICE_JUNCTIONS__junction_CL_0001059_polyA_plus_RNA_seq__SpliceJunction": "SPLICE_JUNC",
    "DNASE__CL_0001059_DNase_seq__CenterMask": "DNASE",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K27ac__CenterMask": "H3K27ac",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K27me3__CenterMask": "H3K27me3",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K36me3__CenterMask": "H3K36me3",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K4me1__CenterMask": "H3K4me1",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K4me3__CenterMask": "H3K4me3",
    "CHIP_HISTONE__CL_0001059_Histone_ChIP_seq_H3K9me3__CenterMask": "H3K9me3",
}


def _api_feats(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort",
        default="data/tp53_mane_noncoding/tp53_mane_noncoding_germline_plus_somatic.csv",
    )
    p.add_argument(
        "--scores",
        default="results/track_scores_tp53_mane_noncoding/test_features_wide.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="results/tp53_mane_noncoding_cv5_api_only",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classifiers", default="lr,rf,xgb")
    args = p.parse_args()

    classifiers = tuple(x.strip() for x in args.classifiers.split(",") if x.strip())
    cohort = pd.read_csv(project_subpath(PROJECT, args.cohort))
    cohort["variant_id"] = cohort["variant_id"].astype(str)
    sc = pd.read_parquet(project_subpath(PROJECT, args.scores))
    sc_feats = [c for c in sc.columns if c not in ("label", "gene")]
    feats = [c for c in sc_feats if c != "variant_id" and pd.api.types.is_numeric_dtype(sc[c])]
    if len(feats) != 10:
        print(f"WARNING: expected 10 API features, got {len(feats)}: {feats}")

    df = cohort.merge(sc[["variant_id"] + feats], on="variant_id", how="inner")
    df = df.drop_duplicates("variant_id")
    df["y"] = (df["label"].astype(str) == "somatic").astype(int)
    if "source" not in df.columns:
        df["source"] = np.where(df["y"] == 1, "cosmic_ncv", "gnomad_wgs")
    df = attach_cosmic_haem_flags(df)
    n_cos = int(df["source"].astype(str).str.contains("cosmic", case=False).sum())
    n_ge3 = int(((df["y"] == 1) & (df["haem_ge3"] == 1)).sum())
    print(f"COSMIC alleles={n_cos}  haem_ge3={n_ge3}")

    y = df["y"].astype(int).values
    print(
        f"CV cohort n={len(df)} germline={(y == 0).sum()} somatic={(y == 1).sum()}"
    )
    print(df["source"].value_counts().to_string())
    print("features:", [SHORT.get(f, f) for f in feats])

    n_splits = int(args.n_splits)
    if (y == 1).sum() < n_splits or (y == 0).sum() < n_splits:
        raise SystemExit("Too few positives/negatives for stratified CV")

    imp = SimpleImputer(strategy="median")
    X_api = imp.fit_transform(df[feats].replace([np.inf, -np.inf], np.nan).values)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    model_specs = [(f"api_{kind}", kind) for kind in classifiers]
    oof = {m: np.full(len(df), np.nan) for m, _ in model_specs}
    fold_metrics: dict[str, list] = {m: [] for m, _ in model_specs}
    fold_weights: dict[str, list] = {m: [] for m, _ in model_specs}

    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(df)), y), 1):
        y_tr, y_te = y[tr], y[te]
        X_tr, X_te = X_api[tr], X_api[te]
        print(
            f"fold {fold_i}/{n_splits} n_train={len(tr)} n_test={len(te)} "
            f"pos_te={int(y_te.sum())}"
        )
        for name, kind in model_specs:
            n_pos = int(y_tr.sum())
            n_neg = int(len(y_tr) - n_pos)
            spw = (n_neg / n_pos) if (kind == "xgb" and n_pos > 0) else None
            if kind == "lr":
                clf = LogisticRegression(
                    max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"
                )
            else:
                clf = make_track_classifier(
                    kind, random_state=args.seed + fold_i, scale_pos_weight=spw
                )
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            oof[name][te] = prob
            fold_metrics[name].append(ranking_metrics(y_te, prob))
            if kind == "lr":
                fold_weights[name].append(np.abs(clf.coef_.ravel()))
            else:
                fold_weights[name].append(
                    np.asarray(clf.feature_importances_, dtype=float)
                )

    out_dir = project_subpath(PROJECT, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short_names = [SHORT.get(c, c) for c in feats]

    weight_rows = []
    for name, kind in model_specs:
        mean_w = np.mean(fold_weights[name], axis=0)
        order = np.argsort(-mean_w)
        for rank_i, j in enumerate(order, 1):
            weight_rows.append(
                {
                    "model": name,
                    "rank": rank_i,
                    "feature": short_names[j],
                    "feature_full": feats[j],
                    "weight": float(mean_w[j]),
                    "weight_type": "abs_coef" if kind == "lr" else "feature_importance",
                }
            )
        print(f"\n{name} weights top5:", [short_names[j] for j in order[:5]])
    pd.DataFrame(weight_rows).to_csv(out_dir / "feature_weights_cv5.csv", index=False)

    # SHAP full suite (beeswarm / dependence / waterfall)
    shap_rows = []
    plot_dir = out_dir / "shap"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        for name, kind in model_specs:
            spw = (n_neg / n_pos) if (kind == "xgb" and n_pos > 0) else None
            if kind == "lr":
                clf = LogisticRegression(
                    max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"
                )
                clf.fit(X_api, y)
                masker = shap.maskers.Independent(X_api, max_samples=X_api.shape[0])
                explainer = shap.LinearExplainer(clf, masker)
                sv = np.asarray(explainer.shap_values(X_api))
                if isinstance(sv, list):
                    sv = np.asarray(sv[1] if len(sv) > 1 else sv[0])
            else:
                clf = make_track_classifier(
                    kind, random_state=args.seed, scale_pos_weight=spw
                )
                clf.fit(X_api, y)
                explainer = shap.TreeExplainer(clf)
                raw = explainer.shap_values(X_api)
                if isinstance(raw, list):
                    sv = np.asarray(raw[1] if len(raw) > 1 else raw[0])
                else:
                    sv = np.asarray(raw)
                    if sv.ndim == 3:
                        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
            mean_abs = np.abs(sv).mean(axis=0)
            mean_signed = sv.mean(axis=0)
            order = np.argsort(-mean_abs)
            for rank_i, j in enumerate(order, 1):
                shap_rows.append(
                    {
                        "model": name,
                        "rank": rank_i,
                        "feature": short_names[j],
                        "feature_full": feats[j],
                        "mean_abs_shap": float(mean_abs[j]),
                        "mean_shap": float(mean_signed[j]),
                    }
                )
            print(f"{name} SHAP top5:", [short_names[j] for j in order[:5]])
            pd.DataFrame(
                [
                    {
                        "rank": r + 1,
                        "feature": short_names[j],
                        "mean_abs_shap": float(mean_abs[j]),
                        "mean_shap": float(mean_signed[j]),
                    }
                    for r, j in enumerate(order)
                ]
            ).to_csv(plot_dir / f"shap_summary_{name}.csv", index=False)

            plt.figure()
            shap.summary_plot(
                sv, X_api, feature_names=short_names, max_display=10, show=False
            )
            plt.tight_layout()
            plt.savefig(
                plot_dir / f"shap_beeswarm_{name}.png", dpi=150, bbox_inches="tight"
            )
            plt.close("all")

            for rank, j in enumerate(order[:3], 1):
                plt.figure()
                shap.dependence_plot(j, sv, X_api, feature_names=short_names, show=False)
                plt.tight_layout()
                feat = short_names[j].replace("/", "_")
                plt.savefig(
                    plot_dir / f"shap_dependence_{name}_top{rank}_{feat}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close("all")

            proba = clf.predict_proba(X_api)[:, 1]
            examples = {
                "highest_somatic_prob": int(np.argmax(proba)),
                "lowest_somatic_prob": int(np.argmin(proba)),
            }
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            if len(pos):
                examples["true_somatic"] = int(pos[np.argmax(proba[pos])])
            if len(neg):
                examples["true_germline"] = int(neg[np.argmin(proba[neg])])
            expected = explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = float(np.asarray(expected).ravel()[-1])
            else:
                expected = float(expected)
            for label, i in examples.items():
                try:
                    exp = shap.Explanation(
                        values=sv[i],
                        base_values=expected,
                        data=X_api[i],
                        feature_names=short_names,
                    )
                    plt.figure()
                    shap.plots.waterfall(exp, max_display=12, show=False)
                    plt.tight_layout()
                    plt.savefig(
                        plot_dir / f"shap_waterfall_{name}_{label}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close("all")
                except Exception as e:
                    print(f"  waterfall {name}/{label} skipped: {e}")
    except Exception as e:
        print(f"SHAP skipped: {e}")

    if shap_rows:
        pd.DataFrame(shap_rows).to_csv(out_dir / "shap_ranking.csv", index=False)

    rows = []
    src_rows = []
    is_cosmic = df["source"].astype(str).str.contains("cosmic", case=False)
    for name, _ in model_specs:
        prob = oof[name]
        rank = ranking_metrics(y, prob)
        pm = prevalence_matched_table(y, prob)
        src_sens = source_sensitivity_pm(
            y,
            prob,
            df["source"].values,
            extra_membership=haem_ge3_membership(df),
        )
        for r in src_sens:
            src_rows.append({"model": name, **r})
        memb = {
            r["source"]: r["sensitivity"]
            for r in src_sens
            if r["scope"] == "source_membership"
        }
        fold_au = [m["auroc"] for m in fold_metrics[name]]
        row = {
            "model": name,
            "n": int(len(df)),
            "n_germline": int((y == 0).sum()),
            "n_somatic": int((y == 1).sum()),
            "n_cosmic": int(is_cosmic.sum()),
            "auroc_oof": rank["auroc"],
            "auprc_oof": rank["auprc"],
            "auroc_fold_mean": float(np.mean(fold_au)),
            "auroc_fold_sd": float(np.std(fold_au, ddof=1)) if len(fold_au) > 1 else 0.0,
            "mean_p_germline": float(np.nanmean(prob[y == 0])),
            "mean_p_somatic": float(np.nanmean(prob[y == 1])),
            "sens_pm": pm.get("sensitivity"),
            "spec_pm": pm.get("specificity"),
            "sens_cosmic_pm": memb.get("cosmic"),
            "sens_cosmic_haem_ge3_pm": memb.get("cosmic_haem_ge3"),
            "sens_gnomad_pm": memb.get("gnomad"),
        }
        rows.append(row)
        print(
            f"{name:12s} OOF AUROC={row['auroc_oof']:.3f}  "
            f"fold={row['auroc_fold_mean']:.3f}±{row['auroc_fold_sd']:.3f}  "
            f"sens@pm={row['sens_pm']:.3f}  spec@pm={row['spec_pm']:.3f}  "
            f"gnomad={row['sens_gnomad_pm']} cosmic={row['sens_cosmic_pm']} "
            f"haem≥3={row['sens_cosmic_haem_ge3_pm']}"
        )

    pd.DataFrame(rows).to_csv(out_dir / "cv5_metrics.csv", index=False)
    pd.DataFrame(src_rows).to_csv(out_dir / "source_sensitivity_pm.csv", index=False)
    oof_df = df[["variant_id", "y", "source", "n_haem", "haem_ge3"]].copy()
    for name, _ in model_specs:
        oof_df[f"prob_{name}"] = oof[name]
    oof_df.to_csv(out_dir / "oof_scores.csv", index=False)
    summary = {
        "cohort": "TP53 MANE noncoding (gnomAD WGS germline AF<0.01 vs COSMIC NCV somatic)",
        "n": int(len(df)),
        "n_germline": int((y == 0).sum()),
        "n_somatic": int((y == 1).sum()),
        "features": short_names,
        "n_splits": n_splits,
        "seed": args.seed,
        "metrics": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_dir / 'cv5_metrics.csv'}")
    print(f"Wrote {out_dir / 'source_sensitivity_pm.csv'}")
    print(f"Wrote {out_dir / 'feature_weights_cv5.csv'}")
    print(f"Wrote {out_dir / 'shap_ranking.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
