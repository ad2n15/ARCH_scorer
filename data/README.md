# Data

- **`arch21_mane_noncoding_intervals.json`:** MANE Select non-coding windows of the 21 ARCH genes, used by the AlphaGenome sklearn models.
- **`release/`:** variant-level tables, downloaded with `python scripts/fetch_data.py` from Zenodo, [doi:10.5281/zenodo.24008687](https://doi.org/10.5281/zenodo.24008687). These are not in git.

  | File | Rows | Content |
  |---|---|---|
  | `genomewide_age_filtered.csv.gz` | 372,396 (85,078 ARCH) | Colin-Leitzinger et al. 2025 age-filtered SNVs (CC BY 4.0). Includes label, 90/5/5 split, gene, Func/ExonicFunc, CADD, substitution type, and held-out scores of the three fine-tuned models |
  | `arch21_noncoding_public.csv.gz` | 70,718 (69,630 germline, 1,088 somatic) | 21 ARCH genes, MANE non-coding. Includes CADD-PHRED, ANNOVAR Func.refGene, the ten AlphaGenome CD34⁺ CMP scores, fine-tuning split and held-out fine-tuned scores |
  | `arch21_coding_public.csv.gz` | 4,724 (4,124 germline, 600 somatic) | 21 ARCH genes, MANE coding; same columns, without fine-tuned scores |

  Germline variants come from gnomAD v4.1. The somatic variants included here are those with a public source: gnomAD allele-balance and age evidence (`paper_mw_*`) or the IARC TP53 database.

  The paper also used 826 non-coding and 1,313 coding somatic variants that are supported only by COSMIC v104. Those are licensed and are not redistributed. To rebuild them, use your own COSMIC download with `paper/sm_ch_finetune/scripts/61`, `81` and `110` (see `paper/REPRODUCE.md`).
