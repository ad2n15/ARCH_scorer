# ARCH_scorer

Code, trained models and reproduction scripts for:

> Dawoud A. *et al.* Sequence-to-function models for age-related clonal haematopoiesis variants (2026).
> *Citation and DOI to be added on publication.*

The repository does two things:

1. **Score your own variants** (GRCh38 SNVs) for two tasks:
   - **`arch21`**: somatic vs germline, for variants in 21 ARCH genes (ASXL1, ASXL2, CBL, DNMT3A, ETNK1, GNAS, GNB1, IDH1, IDH2, JAK2, KIT, KRAS, MPL, NRAS, PPM1D, RUNX1, SETBP1, SF3B1, SRSF2, TET2, TP53).
   - **`genomewide`**: ARCH (age-skewed somatic) vs other somatic, for variants anywhere in the genome.
2. **Reproduce the paper**: every figure and table, from the released summary files or from raw inputs. See [`paper/REPRODUCE.md`](paper/REPRODUCE.md).

**Research use only.** These models are not validated for clinical use.

## Models

### Fine-tuned sequence models

Each model is a frozen backbone with a heads-only classifier, applied to a 64 kb window centred on the ALT allele (`arch-score-ft`). Every model gives a fine-tuned score and a fused score, which adds CADD-PHRED, functional class, substitution type and gene through a logistic regression.

| Task | Backbone | Held-out AUROC, fine-tuned | Held-out AUROC, + annotations | Paper |
|---|---|---|---|---|
| `arch21_noncoding`, trained on non-coding variants (test n = 7,155; 189 somatic) | **NTv3** | **0.789** | **0.818** | Fig. 4b |
| | AlphaGenome | 0.761 | 0.789 | |
| | Borzoi | 0.746 | 0.791 | |
| `arch21_cnc`, trained on coding + non-coding variants (test n = 7,759; 382 somatic) | NTv3 | 0.869 | 0.871 | Fig. 4c |
| | AlphaGenome | 0.862 | 0.862 | |
| | Borzoi | 0.857 | 0.865 | |
| `genomewide` (test n = 18,620; 4,254 ARCH) | **NTv3** | **0.579** | **0.587** | Fig. 5 |
| | AlphaGenome | 0.569 | 0.584 | |
| | Borzoi | 0.559 | 0.583 | |

- **Fused scores for the 21-gene tasks** are slightly higher than in the paper (for example NTv3 0.818 vs 0.815). The released fusion models use CADD-PHRED for every training variant, as the tool does when scoring. The paper's evaluation median-imputed CADD where its CADD file lacked a score.
- **Reproducibility of live scores.** On 20 held-out variants per task, live inference reproduces the stored test scores to within 6e-7 for NTv3 and 1.4e-7 for Borzoi. For AlphaGenome the agreement is within 3e-3 (median 3e-4). AlphaGenome runs in bfloat16, so its scores vary at this level between hardware (GPU or CPU).
- **The genome-wide task is hard.** Use those scores to rank variants, not as a classifier.

### AlphaGenome track-score models (21 ARCH genes, non-coding)

These are gradient-boosted and random-forest models on AlphaGenome CD34⁺ CMP track scores or fold_1 embeddings, plus annotations, fitted on all 71,544 non-coding variants (`arch-score`; details below).

| Model | Features | 5-fold CV AUROC |
|---|---|---|
| `emb_baseline` | fold_1 ALT−REF embedding (64 PCs) + CADD + SBS6 + Func.refGene + gene | 0.741 |
| `api_baseline` | 10 AlphaGenome API track scores + the same annotations | 0.730 |
| `api_only` | 10 AlphaGenome API track scores; use with `--explain` to rank which tracks drive a call | 0.647 |

## Install

```bash
git clone https://github.com/ad2n15/ARCH_scorer.git
cd ARCH_scorer
pip install -e ".[ntv3]"          # fine-tuned NTv3 (recommended); add ,borzoi or ,alphagenome as needed
pip install -e ".[explain]"       # AlphaGenome track-score models with SHAP
pip install -e ".[paper]"         # redraw the paper's figures
```

- **NTv3** is a gated model on Hugging Face, so before first use you need to:
  1. accept the licence at https://huggingface.co/InstaDeepAI/NTv3_650M_post and https://huggingface.co/InstaDeepAI/ntv3_base_model;
  2. create a read token at https://huggingface.co/settings/tokens;
  3. run `hf auth login`, or `export HF_TOKEN=...`.

  The weights (about 2.6 GB) are then downloaded on first use. To use a local copy instead, set `ARCH_NTV3_PATH` or pass `--model-path`.
- **Borzoi** downloads from Hugging Face without a login.
- **Borzoi** needs gReLU (`pip install -e ".[borzoi]"`).
- **AlphaGenome** needs `alphagenome_research`, JAX, and fold_1 weights from Kaggle (`python scripts/install_fold1.py`; you must accept the AlphaGenome model terms).
- **CPU:** the models run on a CPU, but slowly. A GPU is recommended for more than a few variants.

## Score variants with the fine-tuned models

```bash
printf "20-32437597-T-G\n20-32437607-T-G\n" > variants.txt   # chrom-pos-ref-alt, or a VCF
arch-score-ft variants.txt --backbone ntv3 -o scores.csv
```

```python
from arch_scorer.finetuned import Scorer
Scorer("ntv3").score(["20-32437597-T-G"])
```

The terminal shows a short summary, with one row per variant and one column per task, and the CSV has every column. Add `--html report.html` for a readable report: a sortable summary, then one card per variant with percentile bars, scores and notes, then a "how to read this" section. `--full` prints every column instead of the summary, and `--quiet` prints only where the files went.

A summary cell such as `0.412 p97 HIGH` gives:
- the score;
- its percentile among held-out test variants (mostly germline for the 21-gene tasks, and mostly other somatic for genome-wide);
- `HIGH` when the score is above the prevalence-matched threshold used in the paper. This is the test-set quantile equal to 1 − the share of positives, for example p95 for `arch21_cnc`.

`n/a` means the task does not apply, for example a variant outside the 21 genes.

Output columns, for each task:
- `<task>_<backbone>`: the fine-tuned score;
- `<task>_<backbone>_plus_annotations`: the fused score;
- `<task>_percentile`: the percentile of `<task>_score` among held-out test variants;
- `<task>_score`, `<task>_score_type`: the headline score, which is the fused score where all annotations are available and otherwise the fine-tuned score;
- `<task>_flag`: `high` above the prevalence-matched threshold, `-` below it, and empty when the task does not apply;
- `notes`: for example, outside the 21 genes, REF mismatch, or coding variant scored by the non-coding model.

Annotations are fetched from free public APIs, with no keys needed:
- **reference sequence:** UCSC (Ensembl as fallback);
- **functional class:** computed from UCSC RefSeq transcripts in the ANNOVAR style;
- **CADD-PHRED and coding consequence:** CADD v1.7 API.

For offline or batch use, set `ARCH_FASTA=/path/to/hg38.fa` to read sequence windows locally.

## AlphaGenome track-score models

```bash
export ALPHA_GENOME_API_KEY=...                     # or put it in .env (see .env.example)
arch-score --model api_baseline --chrom 20 --pos 32437607 --ref T --alt G --json
arch-score --model api_only --explain --chrom 20 --pos 32437607 --ref T --alt G --json
arch-score --model emb_baseline --extract-emb --chrom 20 --pos 32437607 --ref T --alt G --json   # GPU + fold_1
```

| Model | Needs |
|---|---|
| `api_baseline`, `api_only` | AlphaGenome API key |
| `emb_baseline` | fold_1 weights and a local FASTA (`ARCH_FASTA`; `bash scripts/install_reference.sh ag`), or a precomputed `--emb` file |

The AlphaGenome track scores come from the hosted API at run time. They can differ slightly from the scores used to train the models; for one test variant, nine of ten tracks agreed within about 0.006, and splice junction differed by 0.02 (about 0.1 SD). Scores from `api_*` models may therefore vary a little from the paper's values. The fine-tuned models (`arch-score-ft`) run frozen weights locally and reproduce stored scores exactly.

CADD and functional class are fetched live, or you can pass them with `--cadd-phred` / `--func`. ANNOVAR is optional (`scripts/install_annovar.sh`). Without it, the ANNOVAR-style functional class is computed from UCSC RefSeq transcripts.

## Data

The variant-level tables derived from public sources are on Zenodo, [doi:10.5281/zenodo.24008687](https://doi.org/10.5281/zenodo.24008687) (`python scripts/fetch_data.py`; see [`data/README.md`](data/README.md)). They cover:
- the genome-wide cohort;
- the 21-gene cohort, without the somatic variants supported only by COSMIC, which is licensed.

## Repository layout

```
src/arch_scorer/            arch-score (AlphaGenome track-score models)
src/arch_scorer/finetuned/  arch-score-ft (fine-tuned AlphaGenome / NTv3 / Borzoi heads)
models/finetuned/           9 heads with fusion models, test metrics and score quantiles
artifacts/                  AlphaGenome track-score models (joblib)
paper/                      reproduction: analysis scripts, summary results, figure code (paper/REPRODUCE.md)
scripts/                    install helpers; maintainer export and verification scripts
```

## Maintainers

The model files are rebuilt from the training projects with these scripts:

```bash
python scripts/export_finetuned_models.py                      # models/finetuned/ (needs ALPHAGENOME_ROOT)
ARCH_FASTA=hg38.fa python scripts/verify_finetuned.py --backbone ntv3 -n 20   # live vs stored test scores
python scripts/export_models_from_sm_ch_finetune.py           # artifacts/*.joblib
python scripts/export_public_data.py                          # release_data/ for Zenodo
```

## Licence

- **Code:** Apache-2.0.
- **Model files:** they carry the terms of the backbones they derive from, namely the NTv3 InstaDeep Open Model Licence (non-commercial), the AlphaGenome model terms and MIT for Borzoi. See [`MODEL-LICENSE.md`](MODEL-LICENSE.md).
