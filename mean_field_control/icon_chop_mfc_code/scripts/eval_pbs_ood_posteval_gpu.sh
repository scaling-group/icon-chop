#!/usr/bin/env bash
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -N icon-chop-chain2-ood
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed

set -euo pipefail

: "${BUNDLE_ROOT:?Set BUNDLE_ROOT to the ICON-CHOP bundle directory}"
: "${REPO_ROOT:?Set REPO_ROOT to the repository root}"
OUTPUT_DIR="${REPO_ROOT}/post_eval/icon_agent_mfc/results/icon-chop-operator-chain2-nonnegcenter-ood-500pt-gpu"

cd "${BUNDLE_ROOT}"
source scripts/eval_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running best_program2 ICON-CHOP nonnegative-center 500-point OOD post-eval over all datasets"

"${PYTHON_BIN:-python}" \
  "${REPO_ROOT}/post_eval/icon_agent_mfc/eval_icon_agent_mfc.py" \
  --program "${BUNDLE_ROOT}/seed/initial_program.py" \
  --entrypoint run_icon_chop \
  --ckpt "${ICON_CHOP_CKPT_PATH}" \
  --data-dir "${ICON_CHOP_DATA_DIR}" \
  --pattern '*.h5' \
  --device cuda \
  --batch-size 1 \
  --num-samples-per-dataset 1000000 \
  --max-len 100 \
  --time-subsample 10 \
  --space-subsample 50 \
  --output-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_DIR}/summary.json"

"${PYTHON_BIN:-python}" \
  "${BUNDLE_ROOT}/scripts/validate_ood_posteval_outputs.py" \
  "${OUTPUT_DIR}"
