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

## Install

```bash
git clone https://github.com/ad2n15/ARCH_scorer.git
cd ARCH_scorer
pip install -e .
pip install -e ".[explain]"   # needed for --model api_only --explain

# ANNOVAR (registration download required)
export ANNOVAR_TGZ=/path/to/annovar.latest.tar.gz
bash scripts/install_annovar.sh
export ANNOVAR_DIR=$PWD/annovar_install/annovar
```

### AlphaGenome (for `api_baseline` / `api_only`)

These models call the **hosted AlphaGenome API** (CD34+ track scores). Install/setup the official client and API key as documented by DeepMind:

- Installation: https://github.com/google-deepmind/alphagenome#installation  
- Then set `export ALPHA_GENOME_API_KEY=...` (or pass `--api-key`).

`pip install -e .` already depends on the `alphagenome` Python package; follow the link above if you need a specific install path, extras, or key registration.

**Not required** if you only run `emb_baseline` with a precomputed `--emb` file.  
`--extract-emb` is a separate local fold_1 / research-stack path (not the hosted API).

Maintainers — rebuild full-fit joblibs from `sm_ch_finetune`:

```bash
./run_binemo_AMD.sh ARCH_scorer/scripts/export_models_from_sm_ch_finetune.py
# → artifacts/emb_baseline_xgb_gene_onehot.joblib
# → artifacts/api_baseline_xgb_gene_onehot.joblib
# → artifacts/api_only_rf.joblib

# api_only only:
./run_binemo_AMD.sh ARCH_scorer/scripts/export_models_from_sm_ch_finetune.py --only api_only
```

## Usage

### A) emb + baseline (best)

```bash
arch-score --model emb_baseline \
  --chrom 20 --pos 32437607 --ref T --alt G \
  --emb emb_diff.npy --json

export ALPHAGENOME_ROOT=/path/to/alphagenome
arch-score --model emb_baseline --extract-emb \
  --chrom 20 --pos 32437607 --ref T --alt G --json
```

### B) api + baseline (no local emb)

```bash
export ALPHA_GENOME_API_KEY=...
arch-score --model api_baseline \
  --chrom 20 --pos 32437607 --ref T --alt G --json
```

### C) api only — interpretability (top-10 SHAP ranks)

Single variant:

```bash
export ALPHA_GENOME_API_KEY=...
arch-score --model api_only --explain \
  --chrom 20 --pos 32437607 --ref T --alt G --json
```

Batch (TSV/CSV with `chrom,pos,ref,alt[,gene]`):

```bash
arch-score --model api_only --explain \
  --batch variants.tsv --out ranks.jsonl
```

Output `feature_ranks`: `rank`, `feature_short`, `shap_value` (signed toward somatic), `feature`.

Skip live CADD/ANNOVAR on production models with `--cadd-phred` / `--func`.

## Scope

- 21 ARCH genes; windows in `data/arch21_mane_noncoding_intervals.json`
- hg38 SNVs
- Hosted AG API does **not** return embeddings — only track scores for `api_baseline` / `api_only`

## License

Apache-2.0. AlphaGenome / ANNOVAR / CADD have separate terms.
