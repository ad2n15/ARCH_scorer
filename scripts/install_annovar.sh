#!/usr/bin/env bash
# Install ANNOVAR + hg38 refGene humandb for ARCH_scorer.
#
# ANNOVAR requires a free registration download URL from:
#   https://annovar.openbioinformatics.org/en/latest/user-guide/download/
# Set ANNOVAR_TGZ to the path of the downloaded annovar.latest.tar.gz
#
# Usage:
#   export ANNOVAR_TGZ=/path/to/annovar.latest.tar.gz
#   bash scripts/install_annovar.sh [/install/prefix]
#
# After install:
#   export ANNOVAR_DIR=/install/prefix/annovar

set -euo pipefail

PREFIX="${1:-${PWD}/annovar_install}"
mkdir -p "${PREFIX}"
PREFIX="$(cd "${PREFIX}" && pwd)"

if [[ -z "${ANNOVAR_TGZ:-}" || ! -f "${ANNOVAR_TGZ}" ]]; then
  echo "Set ANNOVAR_TGZ to your downloaded annovar.latest.tar.gz" >&2
  echo "Register/download: https://annovar.openbioinformatics.org/en/latest/user-guide/download/" >&2
  exit 1
fi

echo "Extracting ${ANNOVAR_TGZ} → ${PREFIX}"
tar -xzf "${ANNOVAR_TGZ}" -C "${PREFIX}"
# tarball usually creates PREFIX/annovar/
ANNOVAR_DIR="${PREFIX}/annovar"
if [[ ! -f "${ANNOVAR_DIR}/table_annovar.pl" ]]; then
  # flat extract
  if [[ -f "${PREFIX}/table_annovar.pl" ]]; then
    ANNOVAR_DIR="${PREFIX}"
  else
    echo "Could not find table_annovar.pl under ${PREFIX}" >&2
    exit 1
  fi
fi

mkdir -p "${ANNOVAR_DIR}/humandb"
cd "${ANNOVAR_DIR}"
echo "Downloading hg38 refGene …"
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar refGene humandb/

echo
echo "Done. Add to your environment:"
echo "  export ANNOVAR_DIR=${ANNOVAR_DIR}"
echo "  export ANNOVAR_HUMANDB=${ANNOVAR_DIR}/humandb"
echo "  export ANNOVAR_PL=${ANNOVAR_DIR}/table_annovar.pl"
