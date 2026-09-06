#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/generated/eval_mfc}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_mfc.py" \
  --mode both --eqns 100 --quests 1 --num 100 --file_split 100 \
  --name ood_operator_l01 --ood_target operator --ood_l_scale 0.1 \
  --seed 777 --device "${DEVICE}" --dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_mfc.py" \
  --mode gparam --eqns 100 --quests 1 --num 100 --file_split 100 \
  --name ood_operator_l03 --ood_target operator --ood_l_scale 0.3 \
  --seed 777 --device "${DEVICE}" --dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_mfc.py" \
  --mode rhoparam --eqns 100 --quests 1 --num 100 --file_split 100 \
  --name ood_operator_l03 --ood_target operator --ood_l_scale 0.3 \
  --seed 777 --device "${DEVICE}" --dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_mfc.py" \
  --mode both --eqns 100 --quests 1 --num 100 --file_split 100 \
  --name ood_operator_l05 --ood_target operator --ood_l_scale 0.5 \
  --seed 777 --device "${DEVICE}" --dir "${OUTPUT_DIR}"

