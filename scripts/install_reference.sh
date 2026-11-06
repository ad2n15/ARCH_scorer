#!/usr/bin/env bash
# Download an hg38 / GRCh38 FASTA for fold_1 embedding extraction.
#
# AlphaGenome was trained on GRCh38.p13 (hg38). Preferred (matches AG docs):
#   https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa
# FAQ: https://www.alphagenomedocs.com/faqs.html
#
# Illumina iGenomes NCBI GRCh38 (also usable for hg38 SNVs; chromosome naming
# may differ — extract code handles chr / no-chr):
#   https://s3.amazonaws.com/igenomes.illumina.com/Homo_sapiens/NCBI/GRCh38/Homo_sapiens_NCBI_GRCh38.tar.gz
#
# Usage:
#   bash scripts/install_reference.sh              # AG GRCh38.p13 (default)
#   bash scripts/install_reference.sh igenomes     # Illumina iGenomes NCBI GRCh38
#   bash scripts/install_reference.sh /out/dir ag  # custom prefix + source
#
# After install:
#   export ARCH_FASTA=/path/to/....fa

set -euo pipefail

SRC="${1:-ag}"
PREFIX="${2:-}"
if [[ "${SRC}" == /* || "${SRC}" == .* ]]; then
  # allow: install_reference.sh /out/dir [ag|igenomes]
  PREFIX="${SRC}"
  SRC="${2:-ag}"
fi
PREFIX="${PREFIX:-${PWD}/reference}"
mkdir -p "${PREFIX}"
PREFIX="$(cd "${PREFIX}" && pwd)"

need_faidx() {
  local fa="$1"
  if [[ -f "${fa}.fai" ]]; then
    return 0
  fi
  if command -v samtools >/dev/null 2>&1; then
    echo "Indexing with samtools faidx …"
    samtools faidx "${fa}"
  else
    echo "No samtools; indexing with pyfaidx …"
    python3 - <<PY
from pyfaidx import Fasta
Fasta("${fa}")
print("Wrote ${fa}.fai")
PY
  fi
}

case "${SRC}" in
  ag|alphagenome|grch38.p13|p13)
    FA_URL="https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa"
    FA="${PREFIX}/GRCh38.p13.genome.fa"
    echo "Downloading AlphaGenome reference (GRCh38.p13) …"
    echo "  ${FA_URL}"
    if [[ ! -f "${FA}" ]]; then
      # large file; resume-friendly
      curl -L --fail --retry 3 -C - -o "${FA}.partial" "${FA_URL}"
      mv "${FA}.partial" "${FA}"
    else
      echo "Already present: ${FA}"
    fi
    need_faidx "${FA}"
    echo
    echo "Done. Add to your environment:"
    echo "  export ARCH_FASTA=${FA}"
    ;;
  igenomes|illumina|ncbi)
    TGZ_URL="https://s3.amazonaws.com/igenomes.illumina.com/Homo_sapiens/NCBI/GRCh38/Homo_sapiens_NCBI_GRCh38.tar.gz"
    TGZ="${PREFIX}/Homo_sapiens_NCBI_GRCh38.tar.gz"
    echo "Downloading Illumina iGenomes NCBI GRCh38 …"
    echo "  ${TGZ_URL}"
    if [[ ! -f "${TGZ}" ]]; then
      curl -L --fail --retry 3 -C - -o "${TGZ}.partial" "${TGZ_URL}"
      mv "${TGZ}.partial" "${TGZ}"
    else
      echo "Already present: ${TGZ}"
    fi
    echo "Extracting …"
    tar -xzf "${TGZ}" -C "${PREFIX}"
    # typical layout: .../Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/genome.fa
    FA="$(find "${PREFIX}" -type f \( -name 'genome.fa' -o -name 'GenomeFasta.fa' -o -name '*.fa' \) \
      | grep -E 'WholeGenomeFasta|Sequence' | head -1 || true)"
    if [[ -z "${FA}" ]]; then
      FA="$(find "${PREFIX}" -type f -name 'genome.fa' | head -1 || true)"
    fi
    if [[ -z "${FA}" || ! -f "${FA}" ]]; then
      echo "Could not locate genome.fa after extract under ${PREFIX}" >&2
      echo "Locate the FASTA manually and: export ARCH_FASTA=/path/to/genome.fa" >&2
      exit 1
    fi
    need_faidx "${FA}"
    echo
    echo "Done. Add to your environment:"
    echo "  export ARCH_FASTA=${FA}"
    echo
    echo "Note: AlphaGenome docs recommend GRCh38.p13 specifically"
    echo "  (bash scripts/install_reference.sh ag)."
    echo "  iGenomes NCBI GRCh38 is also hg38-compatible for SNV windows."
    ;;
  *)
    echo "Unknown source '${SRC}'. Use: ag | igenomes" >&2
    exit 1
    ;;
esac
