# ARCH_scorer

AlphaGenome **companion** for ARCH21 MANE **noncoding** alleles → somatic vs germline score.

## Production models (full-cohort fit)

| Model | Features | When to use | OOF (5-fold) |
|-------|----------|-------------|--------------|
| **`emb_baseline`** (default) | fold_1 ALT−REF DIFF PCA64 + CADD + SBS6 + Func.refGene + **gene one-hot** | Best accuracy; needs emb | AUROC **0.741±0.013**, PM Sens **0.231**, Spec **0.979** |
| **`api_baseline`** | 10 CD34+ AG **API** tracks + CADD + SBS6 + Func + gene one-hot | No GPU / no fold_1 weights | AUROC **0.730**, PM Sens **0.197** |

Artifacts are fit on **all** ARCH21 NC alleles; the binary threshold is the published OOF prevalence-matched cut.

API RF track models remain for **interpretability only** (not shipped as scorers).

## Install

```bash
git clone https://github.com/ad2n15/ARCH_scorer.git
cd ARCH_scorer
pip install -e .

# ANNOVAR (registration download required)
export ANNOVAR_TGZ=/path/to/annovar.latest.tar.gz
bash scripts/install_annovar.sh
export ANNOVAR_DIR=$PWD/annovar_install/annovar
```

Maintainers — rebuild full-fit joblibs from `sm_ch_finetune`:

```bash
./run_binemo_AMD.sh ARCH_scorer/scripts/export_models_from_sm_ch_finetune.py
# → artifacts/emb_baseline_xgb_gene_onehot.joblib
# → artifacts/api_baseline_xgb_gene_onehot.joblib
```

## Usage

### A) emb + baseline (best)

Provide a precomputed 1536-d DIFF vector **or** extract locally:

```bash
# Option 1: precomputed emb
arch-score --model emb_baseline \
  --chrom 20 --pos 32437607 --ref T --alt G \
  --emb emb_diff.npy --json

# Option 2: extract then score (GPU + fold_1 research stack + sibling sm_ch_finetune)
export ALPHAGENOME_ROOT=/path/to/alphagenome
arch-score --model emb_baseline --extract-emb \
  --chrom 20 --pos 32437607 --ref T --alt G --json
```

Standalone extract:

```bash
python -m arch_scorer.extract_emb --chrom 20 --pos 32437607 --ref T --alt G --out emb_diff.npy
```

### B) api + baseline (no local emb)

```bash
export ALPHA_GENOME_API_KEY=...
arch-score --model api_baseline \
  --chrom 20 --pos 32437607 --ref T --alt G --json
```

Skip live CADD/ANNOVAR with `--cadd-phred` / `--func`.

## Scope

- 21 ARCH genes; windows in `data/arch21_mane_noncoding_intervals.json`
- hg38 SNVs
- Hosted AG API does **not** return embeddings — only track scores for `api_baseline`

## License

Apache-2.0. AlphaGenome / ANNOVAR / CADD have separate terms.
