# ARCH_scorer

AlphaGenome **companion** for ARCH21 MANE **noncoding** alleles → somatic vs germline score.

## Models (full-cohort fit)

| Model | Features | When to use | OOF (5-fold) |
|-------|----------|-------------|--------------|
| **`emb_baseline`** (default) | fold_1 ALT−REF DIFF PCA64 + CADD + SBS6 + Func.refGene + **gene one-hot** | Best accuracy; needs emb | AUROC **0.741±0.013**, PM Sens **0.231**, Spec **0.979** |
| **`api_baseline`** | 10 CD34+ AG **API** tracks + CADD + SBS6 + Func + gene one-hot | No GPU / no fold_1 weights | AUROC **0.730**, PM Sens **0.197** |
| **`api_only`** | 10 CD34+ AG API tracks only (RF) | **Interpretability** (`--explain` SHAP ranks) | AUROC **0.647**, PM Sens **0.118** |

Artifacts are fit on **all** ARCH21 NC alleles; the binary threshold is the published OOF prevalence-matched cut.

`api_only` is **not** a production classifier — use it to rank which of the 10 API tracks drive the call for a variant.

The hosted AlphaGenome API does **not** return trunk embeddings. For `emb_baseline`, use a precomputed `--emb` file **or** extract locally with **pretrained fold_1** (not a fine-tuned model).

## Install

```bash
git clone https://github.com/ad2n15/ARCH_scorer.git
cd ARCH_scorer
pip install -e .
pip install -e ".[explain]"   # api_only --explain (shap)
pip install -e ".[emb]"       # local fold_1 DIFF extract

# Research model stack (fold_1 forward pass)
pip install git+https://github.com/google-deepmind/alphagenome_research.git
# Install a JAX GPU build matching your CUDA (see jax docs)

# ANNOVAR (registration download required)
export ANNOVAR_TGZ=/path/to/annovar.latest.tar.gz
bash scripts/install_annovar.sh
export ANNOVAR_DIR=$PWD/annovar_install/annovar
```

### API keys (`.env` — set once)

Copy the template and edit (`.env` is gitignored):

```bash
cp .env.example .env
```

| Key | Needed for | How to get it |
|-----|------------|---------------|
| `ALPHA_GENOME_API_KEY` | `api_baseline`, `api_only` | Create/register at [AlphaGenome](https://deepmind.google.com/science/alphagenome) (see also [client install](https://github.com/google-deepmind/alphagenome#installation)) |
| `KAGGLE_USERNAME` + `KAGGLE_KEY` | `python scripts/install_fold1.py` / `--extract-emb` only | [Kaggle settings → API → Create New Token](https://www.kaggle.com/settings); also **accept** [AlphaGenome model terms](https://www.kaggle.com/models/google/alphagenome) |

Example `.env`:

```bash
ALPHA_GENOME_API_KEY=your_ag_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_token
```

`arch-score` and `install_fold1.py` auto-load `.env` from the repo root (or cwd). You can still `export` the same variables in your shell instead.

- **API-only users** need only `ALPHA_GENOME_API_KEY`.
- **Precomputed `--emb` only** need neither key.
- Kaggle alternatively: place `~/.kaggle/kaggle.json` from the token download.

### AlphaGenome API client

- Installation: https://github.com/google-deepmind/alphagenome#installation  
- Key via `.env` as above, or `--api-key`

### fold_1 weights (for `--extract-emb`)

After Kaggle credentials are in `.env` (and model terms accepted):

```bash
python scripts/install_fold1.py
export ARCH_FOLD1_CKPT=$(cat artifacts/fold1_checkpoint_path.txt)
```

### Reference genome (hg38 / GRCh38)

AlphaGenome was trained on **GRCh38.p13** ([FAQ](https://www.alphagenomedocs.com/faqs.html)). Prefer that FASTA:

```bash
# Recommended — AlphaGenome GRCh38.p13
bash scripts/install_reference.sh ag
export ARCH_FASTA=$PWD/reference/GRCh38.p13.genome.fa

# Alternative — Illumina iGenomes NCBI GRCh38 (also hg38-compatible)
bash scripts/install_reference.sh igenomes
export ARCH_FASTA=...   # path printed by the script
```

iGenomes tarball: https://s3.amazonaws.com/igenomes.illumina.com/Homo_sapiens/NCBI/GRCh38/Homo_sapiens_NCBI_GRCh38.tar.gz  

AG default FASTA URL (used by `install_reference.sh ag`):  
https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa

Chromosome names may differ (`chr1` vs `1`); the extractor resolves both.

Maintainers — rebuild full-fit joblibs from `sm_ch_finetune`:

```bash
./run_binemo_AMD.sh ARCH_scorer/scripts/export_models_from_sm_ch_finetune.py
# → artifacts/emb_baseline_xgb_gene_onehot.joblib
# → artifacts/api_baseline_xgb_gene_onehot.joblib
# → artifacts/api_only_rf.joblib
```

## Usage

### A) emb + baseline (best)

```bash
# Precomputed 1536-d ALT−REF DIFF
arch-score --model emb_baseline \
  --chrom 20 --pos 32437607 --ref T --alt G \
  --emb emb_diff.npy --json

# Extract with pretrained fold_1 (GPU), then score
export ARCH_FOLD1_CKPT=... ARCH_FASTA=...
arch-score --model emb_baseline --extract-emb \
  --chrom 20 --pos 32437607 --ref T --alt G --json

# Standalone extract (single or batch)
python -m arch_scorer.extract_emb --chrom 20 --pos 32437607 --ref T --alt G --out emb_diff.npy
python -m arch_scorer.extract_emb --batch variants.tsv --out emb_diff.npy
```

### B) api + baseline (no local emb)

```bash
export ALPHA_GENOME_API_KEY=...
arch-score --model api_baseline \
  --chrom 20 --pos 32437607 --ref T --alt G --json
```

### C) api only — interpretability (top-10 SHAP ranks)

```bash
export ALPHA_GENOME_API_KEY=...
arch-score --model api_only --explain \
  --chrom 20 --pos 32437607 --ref T --alt G --json

arch-score --model api_only --explain \
  --batch variants.tsv --out ranks.jsonl
```

Skip live CADD/ANNOVAR on production models with `--cadd-phred` / `--func`.

## Scope

- 21 ARCH genes; windows in `data/arch21_mane_noncoding_intervals.json`
- hg38 SNVs
- Hosted AG API: track scores only (`api_baseline` / `api_only`)
- Embeddings: pretrained **fold_1** local extract (or user-supplied `--emb`)

## License

Apache-2.0. AlphaGenome / ANNOVAR / CADD have separate terms.
