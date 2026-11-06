# Reproducing the paper

There are two levels of reproduction:

1. **Redraw every figure and table from the released summary files.** This takes minutes on a laptop and needs no individual-level data.
2. **Rerun the analyses from the raw inputs.** This needs the licensed and public source data, AlphaGenome API access and a GPU.

## Folder layout

`paper/` mirrors the working layout of the original project, so scripts can import one another by file name:

| Folder | Contents |
|---|---|
| `sm_ch_finetune/` | Cohort building, annotation, AlphaGenome scoring and embeddings, CV classifiers, AlphaGenome fine-tuning, and evaluation (`scripts/`, `src/`, `config/`) |
| `sm_ch_finetune_ntv3/`, `sm_ch_finetune_borzoi/` | NTv3 and Borzoi heads-only fine-tuning (`03_train.py`) and prediction export (`04_export_predictions.py`) |
| `figures/` | `make_figures.py` (all figures), `make_supp_tables.py`, `make_per_gene_finetune_table.py` |
| `sm_ch_finetune/results/` | The summary files the figures are drawn from: no variant identifiers, no individual-level data |

Run every script from its project folder, for example `cd paper/sm_ch_finetune && python3 scripts/136_….py`. Each script's docstring gives its exact command and options.

## 1. Figures and tables from the released summaries

```bash
pip install -e ".[paper]"
python paper/figures/make_figures.py        # -> paper/figures/output/Fig1..5, SuppFig3..4 (PDF, PNG, TIFF)
python paper/figures/make_supp_tables.py    # -> paper/figures/output/supplementary_tables.xlsx
```

| Display item | Summary file(s) in `sm_ch_finetune/results/` | Produced by |
|---|---|---|
| Fig. 1 cohort counts | `arch_mane_table1_covariate_adjusted/table1b_*` (n rows); `../data/processed_age_filtered/split_stats.json` | 119; `01e_build_age_filtered_splits.py` |
| Fig. 2, Table S10 | `arch_mane_table1_covariate_adjusted/table1b_*.csv`, `table1c_*.csv` | `119_table1_covariate_adjusted_or.py` |
| Fig. 3a, Table S11 | `table2_cadd_api_emb_lr_train_x_eval.csv` | 106, 102, 109, 116, 142 → `146_table2_cadd_api_emb_lr_fold_sd.py` |
| Fig. 3b, Table S12 | `plots/arch21_nc_publish/pooled_api_rf_impurity_vs_shap.csv` | `119b_pooled_api_rf_importance.py` |
| Fig. 3c, Supp. Fig. 3 | `plots/arch21_nc_publish/per_gene_auroc_ordered_by_emb_benefit.csv` | 106 and 109 (`--cv-mode stratify_gene_func --gene-feature onehot`) → `118_plot_arch21_nc_per_gene_auroc_heatmap.py` |
| Fig. 4 | `arch21_finetune_table_s5_bootstrap.csv`, `arch21_baseline_only_lr_summary.csv`, `arch21_nc_finetune_test_bootstrap/finetune_test_auroc_bootstrap.csv` | 115 → fine-tuning (below) → 137, 136, `144_bootstrap_arch21_finetune_table_s5.py`, 143 |
| Fig. 5a,b | `finetune_covariates_full_age_filtered.json`, `ntv3_full_finetune_covariates.json`, `borzoi_full_finetune_covariates.json` | `08_evaluate_finetune_covariates.py` with `config/{full_age_filtered,ntv3_full,borzoi_full}_covariates.yaml` |
| Fig. 5c, Table S15 | `full_model_agreement.json` | `26_full_model_agreement.py` with `config/full_age_filtered_multibackbone_ensembles.yaml` |
| Genome-wide bootstrap CIs | `full_cohort_finetune_table_s7_bootstrap.csv` | `145_bootstrap_full_cohort_table_s7.py` |
| Supp. Fig. 4 | `plots/arch21_nc_publish/asxl1_oof_scores_deidentified.csv` (ASXL1 out-of-fold scores, no IDs) | 106 and 109 (out-of-fold predictions) |
| Table S16 | `per_gene_finetune_ntv3_nc.csv` | `figures/make_per_gene_finetune_table.py` |

## 2. Full rerun from raw inputs

### External inputs

Download these into the project root, where the scripts expect them:

| Input | Used by | Access |
|---|---|---|
| gnomAD v4.1 genomes and exomes sites VCFs (GRCh38) | 65, 66, 75, 110 | public |
| gnomAD v2.1.1 age histograms (liftover) | 00 | public |
| Colin-Leitzinger et al. 2025 supplementary table (`40246_2025_847_MOESM2_ESM.csv`) | 00 → 01e | public (CC BY 4.0) |
| COSMIC v104 GRCh38: NonCodingVariants, MutantCensus, Classification | 61, 81, 110 | **licence required** (free for academic use) |
| IARC TP53 database r21 (tumour and germline downloads) | 01q, 82 | public |
| GENCODE v46 GTF; hg38 FASTA | 61, 89, AlphaGenome / NTv3 / Borzoi windows | public |
| CADD v1.7 GRCh38 whole-genome SNVs (or the CADD API) | 86 (or 50) | public, non-commercial |
| ANNOVAR with refGene hg38 | `sh/80` | registration |
| AlphaGenome API key; fold_1 weights (Kaggle) | 05, 101, AlphaGenome fine-tuning | free, terms apply |
| NTv3 650M post-trained; Borzoi human_rep0 | NTv3 and Borzoi fine-tuning | Hugging Face |

If you are not using COSMIC, `data/release/arch21_*_public.csv.gz` (Zenodo doi:10.5281/zenodo.24008687; `python scripts/fetch_data.py`) contains every germline variant and every somatic variant with a public source. With these tables the analyses run on a subset (1,088 of 1,914 non-coding somatic variants), so the results will differ from the paper.

### Order of steps (21 ARCH genes)

1. **Cohort.**
   - Build the non-coding lists with 61, 65 (also run with `--min-ab-n 2 --out-dir …_minABn2`), 66, 75, 81, 01q and 82.
   - Pool them with 76, then apply the Ellingford filter with 89 (n = 71,544).
   - Build the coding cohort with 110 (n = 6,037).
2. **Annotation.** Run ANNOVAR (`sh/80_annovar_arch_mane_pooled_sm_by_gene.sh`) and CADD (86, or `sh/68` → 50).
3. **AlphaGenome CD34⁺ CMP scores.**
   - Make the pilot files with 111.
   - Score each gene with `sh/67_score_arch_mane_pooled_sm_by_gene.sh`, which runs `05_score_variant_tracks.py` with `config/track_scoring_arch_mane_noncoding_pooled_sm.yaml`.
4. **Embeddings.** Run 101 on a GPU with fold_1.
5. **Association and CV.**
   - 119 (Table 1 / Fig. 2) and 119b (Table 3 / Fig. 3b).
   - 106, 102, 109, 112, 114, 116, 142, 146 and 118 (Fig. 3, Supp. Figs 3–4).
6. **Fine-tuning splits.** Run 115 (80/10/10, stratified by gene × Func.refGene × label).
7. **Heads-only fine-tuning, on a GPU.**
   - **AlphaGenome:** `03_train_head.py`, then `04_evaluate.py`, with `config/arch21_mane_{noncoding,cnc}_gvs_nocds_64kb_bs8_5ep.yaml`.
   - **NTv3 and Borzoi:** `sm_ch_finetune_{ntv3,borzoi}/scripts/03_train.py`, then `04_export_predictions.py`, with `config/arch21_mane_{noncoding,cnc}_gvs.yaml`.
8. **Evaluation.** Run 137, 136, 144, 143 and `figures/make_per_gene_finetune_table.py`.

### Order of steps (genome-wide)

1. **Splits.** Run 00, then 01e (random 90/5/5, seed 42).
2. **Fine-tuning.**
   - **AlphaGenome:** 03, then 04, with `config/full_age_filtered_nocds_64kb_bs8_5ep.yaml`.
   - **NTv3 and Borzoi:** `03_train.py`, then `04_export_predictions.py`, with `config/full_age_filtered.yaml`.
3. **Evaluation.** Run 08 (once per backbone config), 145 and 26.

### Hardware

- AlphaGenome scoring uses the hosted API.
- Embeddings and fine-tuning used one NVIDIA L40 (48 GB) per job. The genome-wide fine-tuning runs take many hours each (about 42,000 steps per epoch at batch size 8).
- Everything else runs on a CPU.

## Known differences from the submitted figures

- **Fig. 3a:** two labels were truncated rather than rounded in the submitted figure. They now read 0.736 (was 0.735) and 0.620 (was 0.619), as in `table2_cadd_api_emb_lr_train_x_eval.csv`.
- **Rebuilt scripts:** `119_table1_covariate_adjusted_or.py` and `119b_pooled_api_rf_importance.py` were rewritten in November 2026 because the originals were not kept. They reproduce the stored outputs to within 1e-8 (Table 1) and 1e-14 (Table 3).
