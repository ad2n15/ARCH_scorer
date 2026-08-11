#!/usr/bin/env python3
"""Smoke test: score ASXL1 20-32437607-T-G with exported artifact."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ARCH_scorer" / "src"))

from arch_scorer import ArchScorer  # noqa: E402

VID = "20-32437607-T-G"
ART = ROOT / "ARCH_scorer/artifacts/emb_baseline_xgb_gene_onehot.joblib"
META = ROOT / "sm_ch_finetune/results/fold1_embeddings_pooled_sm/ASXL1/variants_meta.parquet"
DIFF = ROOT / "sm_ch_finetune/results/fold1_embeddings_pooled_sm/ASXL1/emb_diff.npy"
OOF = (
    ROOT
    / "sm_ch_finetune/results/pooled_arch21_cv5_fold1_emb_baseline_emb_diff_pca64_stratify_gene_func_genefeat_onehot/oof_predictions.parquet"
)


def main() -> None:
    meta = pd.read_parquet(META)
    diff = np.load(DIFF)
    i = int(meta.index[meta.variant_id.astype(str) == VID][0])
    oof = pd.read_parquet(OOF)
    r0 = oof.loc[oof.variant_id.astype(str) == VID].iloc[0]
    scorer = ArchScorer(
        "emb_baseline", ART, fetch_cadd=False, fetch_annovar=False
    )
    res = scorer.score(
        "20",
        32437607,
        "T",
        "G",
        emb_diff=diff[i],
        gene="ASXL1",
        cadd_phred=float(r0.cadd_phred),
        func_refgene=str(r0["Func.refGene"]),
    )
    print(
        f"emb full_fit_P={res.p_somatic:.4f} oof_P={float(r0.p_emb_base_xgb):.4f} "
        f"PM={res.call_somatic_pm} thr={res.pm_threshold:.4f} "
        f"{res.gene} {res.mutation_type} {res.func_refgene}"
    )

    api_art = ROOT / "ARCH_scorer/artifacts/api_baseline_xgb_gene_onehot.joblib"
    if api_art.is_file():
        api_oof = (
            ROOT
            / "sm_ch_finetune/results/pooled_arch21_cv5_stratify_gene_func_genefeat_onehot/oof_predictions.parquet"
        )
        ao = pd.read_parquet(api_oof)
        ar = ao.loc[ao.variant_id.astype(str) == VID].iloc[0]
        sc_api = pd.read_parquet(
            ROOT
            / "sm_ch_finetune/results/track_scores_arch_mane_noncoding_pooled_sm_by_gene/ASXL1/test_features_wide.parquet"
        )
        sc_api["variant_id"] = sc_api["variant_id"].astype(str)
        row = sc_api.loc[sc_api.variant_id == VID].iloc[0]
        api_feats = [
            c
            for c in sc_api.columns
            if any(
                c.startswith(p)
                for p in (
                    "RNA_SEQ__",
                    "SPLICE_SITE_USAGE__",
                    "SPLICE_JUNCTIONS__",
                    "DNASE__",
                    "CHIP_HISTONE__",
                )
            )
        ]
        tracks = {c: float(row[c]) for c in api_feats}
        scorer2 = ArchScorer(
            "api_baseline", api_art, fetch_cadd=False, fetch_annovar=False
        )
        res2 = scorer2.score(
            "20",
            32437607,
            "T",
            "G",
            api_tracks=tracks,
            gene="ASXL1",
            cadd_phred=float(ar.cadd_phred) if "cadd_phred" in ar else float(r0.cadd_phred),
            func_refgene=str(ar["Func.refGene"]) if "Func.refGene" in ar else str(r0["Func.refGene"]),
            fetch_api=False,
        )
        oof_p = float(ar["p_api_baseline_xgb"]) if "p_api_baseline_xgb" in ar else float("nan")
        print(
            f"api full_fit_P={res2.p_somatic:.4f} oof_P={oof_p:.4f} "
            f"PM={res2.call_somatic_pm} thr={res2.pm_threshold:.4f}"
        )


if __name__ == "__main__":
    main()
