#!/usr/bin/env bash
# Score AlphaGenome API tracks for ARCH MANE pooled-SM cohort, one gene at a time.
#
# Usage (from alphagenome/):
#   export ALPHA_GENOME_API_KEY=...
#   GENE=TP53 bash sm_ch_finetune/scripts/sh/67_score_arch_mane_pooled_sm_by_gene.sh
#   bash sm_ch_finetune/scripts/sh/67_score_arch_mane_pooled_sm_by_gene.sh --genes TP53,DNMT3A
#   bash sm_ch_finetune/scripts/sh/67_score_arch_mane_pooled_sm_by_gene.sh --all

set -euo pipefail

ROOT="${ALPHAGENOME_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PILOTS="${PILOTS:-${ROOT}/sm_ch_finetune/data/pilot_arch_mane_noncoding_pooled_sm_by_gene}"
SCORES_ROOT="${SCORES_ROOT:-${ROOT}/sm_ch_finetune/results/track_scores_arch_mane_noncoding_pooled_sm_by_gene}"
PILOT_REL="${PILOT_REL:-data/pilot_arch_mane_noncoding_pooled_sm_by_gene}"
SCORES_REL="${SCORES_REL:-results/track_scores_arch_mane_noncoding_pooled_sm_by_gene}"
RUNNER="${RUNNER:-./run_binemo_AMD.sh}"

ALL=0
GENES_LIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL=1; shift ;;
    --genes)
      shift
      GENES_LIST="${1:?--genes needs a comma-separated gene list}"
      shift
      ;;
    --genes=*)
      GENES_LIST="${1#--genes=}"
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: GENE=TP53 $0 | $0 --genes G1,G2 | $0 --all" >&2
      exit 1
      ;;
  esac
done

score_one() {
  local gene="$1"
  local pilot="${PILOTS}/${gene}"
  if [[ ! -f "${pilot}/test.parquet" ]]; then
    echo "SKIP ${gene}: no pilot at ${pilot}" >&2
    return 1
  fi
  mkdir -p "${SCORES_ROOT}/${gene}"
  local cfg_tmp="${ROOT}/sm_ch_finetune/config/_tmp_track_scoring_pooled_sm_${gene}.yaml"
  cat > "${cfg_tmp}" <<EOF
project_dir: .
data:
  age_filtered_dir: data/processed_age_filtered
  gtf_path: data/reference/gencode.v46.annotation.gtf
  fasta_path: data/reference/hg38.fa
  pilot_dir: ${PILOT_REL}/${gene}
  use_cds_filter: false
  use_age_filter: true
  seed: 42
api:
  key_env: ALPHA_GENOME_API_KEY
  model_version: null
  organism: HOMO_SAPIENS
  sequence_length: 1048576
scoring:
  ontology_terms:
    - CL:0001059
  scorers:
    - RNA_SEQ
    - DNASE
    - ATAC
    - CHIP_HISTONE
    - CHIP_TF
    - CAGE
    - PROCAP
    - SPLICE_SITES
    - SPLICE_SITE_USAGE
    - SPLICE_JUNCTIONS
  include_polyadenylation: false
  match_gene_strand: true
  raw_only: true
  track_model: rf
  splits: ["test"]
  max_variants: null
  resume: true
results_dir: results
scores_dir: ${SCORES_REL}/${gene}
EOF
  echo "=== API score ${gene} (n from ${PILOT_REL}/${gene}/test.parquet) ==="
  cd "${ROOT}"
  ${RUNNER} sm_ch_finetune/scripts/05_score_variant_tracks.py \
    --config "sm_ch_finetune/config/_tmp_track_scoring_pooled_sm_${gene}.yaml"
}

if [[ -n "${GENES_LIST}" ]]; then
  IFS=',' read -ra _genes <<< "${GENES_LIST}"
  for gene in "${_genes[@]}"; do
    gene="$(echo "${gene}" | tr -d '[:space:]')"
    [[ -z "${gene}" ]] && continue
    score_one "${gene}"
  done
elif [[ "${ALL}" -eq 1 ]]; then
  for d in "${PILOTS}"/*; do
    [[ -d "$d" ]] || continue
    score_one "$(basename "$d")"
  done
else
  GENE="${GENE:?Set GENE=TP53 or pass --all / --genes}"
  score_one "${GENE}"
fi
