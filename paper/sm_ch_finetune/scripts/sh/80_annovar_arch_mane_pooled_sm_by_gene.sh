#!/usr/bin/env bash
# ANNOVAR Func.refGene for ARCH MANE pooled-SM cohort, one gene at a time.
#
# Usage (from alphagenome/):
#   GENE=TP53 bash sm_ch_finetune/scripts/sh/80_annovar_arch_mane_pooled_sm_by_gene.sh
#   bash sm_ch_finetune/scripts/sh/80_annovar_arch_mane_pooled_sm_by_gene.sh --genes TP53,DNMT3A
#   bash sm_ch_finetune/scripts/sh/80_annovar_arch_mane_pooled_sm_by_gene.sh --all

set -euo pipefail

ROOT="${ALPHAGENOME_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
COHORT_DIR="${COHORT_DIR:-${ROOT}/sm_ch_finetune/data/arch_mane_noncoding_pooled_sm/by_gene}"
OUT_DIR="${OUT_DIR:-${ROOT}/sm_ch_finetune/data/arch_mane_noncoding_pooled_sm/annovar_by_gene}"
ANNOVAR_PL="${ANNOVAR_PL:-${ROOT}/annovar/table_annovar.pl}"
HUMANDB="${HUMANDB:-${ROOT}/annovar/humandb/}"

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

annovar_one() {
  local gene="$1"
  local cohort="${COHORT_DIR}/${gene}.csv"
  if [[ ! -f "${cohort}" ]]; then
    echo "SKIP ${gene}: missing ${cohort}" >&2
    return 1
  fi
  mkdir -p "${OUT_DIR}"
  local avi="${OUT_DIR}/${gene}.avinput"
  local out_prefix="${OUT_DIR}/${gene}_annovar"
  python3 - <<PY
import pandas as pd
df = pd.read_csv("${cohort}")
with open("${avi}", "w") as f:
    for r in df.itertuples(index=False):
        c = str(r.chrom).replace("chr", "")
        f.write(f"{c}\t{int(r.pos)}\t{int(r.pos)}\t{r.ref}\t{r.alt}\n")
print("${gene}: wrote", len(df), "lines -> ${avi}")
PY
  echo "=== ANNOVAR ${gene} ==="
  cd "${ROOT}"
  perl "${ANNOVAR_PL}" \
    "${avi}" \
    "${HUMANDB}" \
    -buildver hg38 \
    -out "${out_prefix}" \
    -remove -protocol refGene -operation g -nastring . -csvout
  echo "Wrote ${out_prefix}.hg38_multianno.csv"
}

if [[ -n "${GENES_LIST}" ]]; then
  IFS=',' read -ra _genes <<< "${GENES_LIST}"
  for gene in "${_genes[@]}"; do
    gene="$(echo "${gene}" | tr -d '[:space:]')"
    [[ -z "${gene}" ]] && continue
    annovar_one "${gene}"
  done
elif [[ "${ALL}" -eq 1 ]]; then
  for f in "${COHORT_DIR}"/*.csv; do
    [[ -f "$f" ]] || continue
    annovar_one "$(basename "$f" .csv)"
  done
else
  GENE="${GENE:?Set GENE=TP53 or pass --all / --genes}"
  annovar_one "${GENE}"
fi
