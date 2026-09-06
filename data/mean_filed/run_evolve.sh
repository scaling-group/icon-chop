#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/generated/evolve_mfc}"
PREFIX="test_ood_operator_mfc_gparam_forward22"
MERGED_PATH="${OUTPUT_DIR}/${PREFIX}_merged_1.h5"

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${MERGED_PATH}" ]]; then
  echo "Refusing to overwrite ${MERGED_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_mfc.py" \
  --mode gparam --ptype forward22 \
  --eqns 100 --quests 1 --num 100 --file_split 100 \
  --name test_ood_operator --ood_target operator --ood_l_scale 0.5 \
  --seed 301 --device "${DEVICE}" --dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_mfc.py" \
  --input-dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}" \
  --output "${MERGED_PATH}" \
  --expected-groups 100 \
  --delete-sources

