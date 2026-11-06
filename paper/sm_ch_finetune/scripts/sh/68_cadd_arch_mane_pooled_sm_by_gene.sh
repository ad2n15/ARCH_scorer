#!/usr/bin/env bash
# Fetch CADD PHRED for ARCH MANE pooled-SM cohort, one gene at a time.
#
# Usage (from alphagenome/):
#   GENE=TP53 bash sm_ch_finetune/scripts/sh/68_cadd_arch_mane_pooled_sm_by_gene.sh
#   bash sm_ch_finetune/scripts/sh/68_cadd_arch_mane_pooled_sm_by_gene.sh --genes TP53,DNMT3A
#   bash sm_ch_finetune/scripts/sh/68_cadd_arch_mane_pooled_sm_by_gene.sh --all

set -euo pipefail

ROOT="${ALPHAGENOME_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
COHORT_DIR="${COHORT_DIR:-${ROOT}/sm_ch_finetune/data/arch_mane_noncoding_pooled_sm/by_gene}"
OUT_DIR="${OUT_DIR:-${ROOT}/sm_ch_finetune/data/arch_mane_noncoding_pooled_sm/cadd_by_gene}"
COHORT_REL="${COHORT_REL:-sm_ch_finetune/data/arch_mane_noncoding_pooled_sm/by_gene}"
OUT_REL="${OUT_REL:-sm_ch_finetune/data/arch_mane_noncoding_pooled_sm/cadd_by_gene}"

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

cadd_one() {
  local gene="$1"
  local cohort="${COHORT_DIR}/${gene}.csv"
  if [[ ! -f "${cohort}" ]]; then
    echo "SKIP ${gene}: missing ${cohort}" >&2
    return 1
  fi
  mkdir -p "${OUT_DIR}"
  echo "=== CADD ${gene} ==="
  cd "${ROOT}"
  python3 sm_ch_finetune/scripts/50_fetch_tp53_cadd_api.py \
    --cohort "${COHORT_REL}/${gene}.csv" \
    --out "${OUT_REL}/${gene}_cadd_phred_api.csv"
}

if [[ -n "${GENES_LIST}" ]]; then
  IFS=',' read -ra _genes <<< "${GENES_LIST}"
  for gene in "${_genes[@]}"; do
    gene="$(echo "${gene}" | tr -d '[:space:]')"
    [[ -z "${gene}" ]] && continue
    cadd_one "${gene}"
  done
elif [[ "${ALL}" -eq 1 ]]; then
  for f in "${COHORT_DIR}"/*.csv; do
    [[ -f "$f" ]] || continue
    cadd_one "$(basename "$f" .csv)"
  done
else
  GENE="${GENE:?Set GENE=TP53 or pass --all / --genes}"
  cadd_one "${GENE}"
fi
