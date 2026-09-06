#!/usr/bin/env bash
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -N icon-chop-gres-rho
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -k oed

set -euo pipefail

: "${BUNDLE_ROOT:?Set BUNDLE_ROOT to the ICON-CHOP bundle directory}"
: "${REPO_ROOT:?Set REPO_ROOT to the repository root}"
OUTPUT_DIR="${REPO_ROOT}/post_eval/icon_agent_mfc/results/icon-chop-operator-chain2-gres-only-rhoparam-500pt-gpu"

cd "${BUNDLE_ROOT}"
source scripts/eval_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running best_program2 G_res-only rhoparam 500-point OOD post-eval"

"${PYTHON_BIN:-python}" \
  "${REPO_ROOT}/post_eval/icon_agent_mfc/eval_icon_agent_mfc.py" \
  --program "${BUNDLE_ROOT}/seed/initial_program_g_res_only.py" \
  --entrypoint run_icon_chop \
  --ckpt "${ICON_CHOP_CKPT_PATH}" \
  --data-dir "${ICON_CHOP_DATA_DIR}" \
  --pattern '*rhoparam*.h5' \
  --device cuda \
  --batch-size 1 \
  --num-samples-per-dataset 1000000 \
  --max-len 100 \
  --time-subsample 10 \
  --space-subsample 50 \
  --output-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_DIR}/summary.json"

"${PYTHON_BIN:-python}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
summaries = summary.get("summaries", [])
if summary.get("dataset_count") != 6 or len(summaries) != 6:
    raise SystemExit(f"Expected 6 rhoparam datasets, got dataset_count={summary.get('dataset_count')} summaries={len(summaries)}")
failed = [row["dataset_name"] for row in summaries if float(row.get("success_rate", 0.0)) <= 0.0]
if failed:
    raise SystemExit(f"G_res-only rhoparam eval had failed datasets: {failed}")
for row in summaries:
    metrics_path = output_dir / "metrics" / f"{row['dataset_name']}.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = payload.get("best_sample_artifacts", [])
    if not records:
        raise SystemExit(f"{row['dataset_name']} has no best_sample_artifacts")
print(f"Validated G_res-only rhoparam outputs in {output_dir}")
PY
