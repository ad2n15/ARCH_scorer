#!/usr/bin/env python
"""Supplementary tables built from the summary files in paper/sm_ch_finetune/results.

Writes paper/figures/output/supplementary_tables.xlsx with one sheet per table:
  S10  covariate-adjusted ORs (Fig. 2)            S11  scores vs embeddings (Fig. 3a)
  S12  pooled feature importance (Fig. 3b)        S15  genome-wide score concordance (Fig. 5c)
  S16  per-gene NTv3 held-out AUROC               plus the bootstrap AUROC tables for Figs 4 and 5.
"""
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

RES = Path(__file__).resolve().parents[1] / "sm_ch_finetune/results"
OUT = Path(__file__).resolve().parent / "output"
HEAD = PatternFill("solid", fgColor="DDE3E7")


def sheet(wb, name, title, note, df):
    ws = wb.create_sheet(name)
    ws.append([title])
    ws.append([note])
    ws.append([])
    ws["A1"].font = Font(bold=True)
    ws["A2"].font = Font(italic=True, color="5B6B76")
    ws.append(list(df.columns))
    for c in ws[ws.max_row]:
        c.font, c.fill = Font(bold=True), HEAD
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in row])
    for col in ws.columns:
        width = max(len(str(c.value or "")) for c in col[3:])
        ws.column_dimensions[col[0].column_letter].width = min(max(9, width + 2), 40)
        for c in col[3:]:
            c.alignment = Alignment(vertical="top")


def concordance() -> pd.DataFrame:
    d = json.loads((RES / "full_model_agreement.json").read_text())
    names = {"covariates": "Annotations (LR)", "alphagenome": "AlphaGenome", "ntv3": "NTv3", "borzoi": "Borzoi"}
    rows = []
    for k in sorted(d["spearman"], key=lambda k: (k.startswith("covariates"), -d["spearman"][k])):
        a, b = k.split("__")
        sc = d["shared_positive_calls"][k]
        rows.append({"Model A": names[a], "Model B": names[b], "Spearman rho": round(d["spearman"][k], 3),
                     "Jaccard": round(d["jaccard_prev_matched"][k], 3),
                     "Cohen's kappa": round(d["cohen_kappa_prev_matched"][k], 3),
                     "Shared ARCH calls (n)": sc["n_shared"], "Shared ARCH calls (% of k)": round(100 * sc["frac_of_k"], 1)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    t1 = RES / "arch_mane_table1_covariate_adjusted"
    sheet(wb, "S10", "Supplementary Table S10. AlphaGenome CD34+ CMP features and somatic status in 21 ARCH genes: "
          "covariate-adjusted odds ratios", "One logistic regression per feature, adjusted for CADD-PHRED, SBS6, "
          "functional class and gene; OR per 1 SD; BH q within stratum. Shown in Fig. 2.",
          pd.read_csv(t1 / "table1b_covariate_adjusted_wide.csv"))
    sheet(wb, "S11", "Supplementary Table S11. AlphaGenome scores and embeddings vs CADD-PHRED",
          "Logistic regression, 5-fold CV; AUROC fold mean and SD. Shown in Fig. 3a.",
          pd.read_csv(RES / "table2_cadd_api_emb_lr_train_x_eval.csv"))
    sheet(wb, "S12", "Supplementary Table S12. Pooled feature importance of the ten AlphaGenome CD34+ CMP features",
          "Random forest fitted on all 71,544 non-coding variants; impurity importance and mean |TreeSHAP|. "
          "Shown in Fig. 3b.", pd.read_csv(RES / "plots/arch21_nc_publish/pooled_api_rf_impurity_vs_shap.csv"))
    sheet(wb, "S15", "Supplementary Table S15. Concordance of held-out genome-wide scores",
          "ARCH vs other somatic, held-out test set (n = 18,620; 4,254 ARCH); prevalence-matched top-k calls. "
          "Shown in Fig. 5c.", concordance())
    sheet(wb, "S16", "Supplementary Table S16. Per-gene held-out AUROC, NTv3, non-coding germline vs somatic",
          "Annotation baseline, fine-tuned NTv3 and NTv3 + annotations; 95% CI from 2,000 bootstrap resamples.",
          pd.read_csv(RES / "per_gene_finetune_ntv3_nc.csv"))
    sheet(wb, "AUROC_21genes", "Held-out AUROC with 95% bootstrap CIs, 21 ARCH genes (Fig. 4)",
          "10,000 resamples within class.", pd.read_csv(RES / "arch21_finetune_table_s5_bootstrap.csv"))
    sheet(wb, "AUROC_genomewide", "Held-out AUROC with 95% bootstrap CIs, genome-wide ARCH vs other somatic (Fig. 5)",
          "10,000 resamples within class; overall and by functional class.",
          pd.read_csv(RES / "full_cohort_finetune_table_s7_bootstrap.csv"))
    out = OUT / "supplementary_tables.xlsx"
    wb.save(out)
    print(out)


if __name__ == "__main__":
    main()
