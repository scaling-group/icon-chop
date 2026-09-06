#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/generated/eval_conservation}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_conservation.py" eval \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"

