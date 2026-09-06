#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_weno_eval.sh [options]

Options:
  --data-dir PATH
  --checkpoint PATH
  --dataset FILE              Repeat to override the three canonical datasets.
  --num-samples N             Default: 500
  --rollout-steps N           Default: 10
  --batch-size N              Default: 10
  --device DEVICE             Default: auto
  --run-name NAME
  -h, --help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../.." && pwd)"

data_dir=""
checkpoint=""
datasets=(test_seq_sin.pt test_seq_tanh.pt test_seq_u2.pt)
datasets_overridden=0
num_samples=500
rollout_steps=10
batch_size=10
device=auto
run_name="weno_eval_$(date +%Y%m%d_%H%M%S)"

while (($#)); do
  case "$1" in
    --data-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; data_dir=$2; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; checkpoint=$2; shift 2 ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      if ((datasets_overridden == 0)); then datasets=(); datasets_overridden=1; fi
      datasets+=("$2")
      shift 2
      ;;
    --num-samples) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; num_samples=$2; shift 2 ;;
    --rollout-steps) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; rollout_steps=$2; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; batch_size=$2; shift 2 ;;
    --device) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; device=$2; shift 2 ;;
    --run-name) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; run_name=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

args=(
  --program "$workspace_root/conservation_law/icon_chop_weno_code/initial_program.py"
  --num-samples "$num_samples"
  --rollout-steps "$rollout_steps"
  --batch-size "$batch_size"
  --device "$device"
  --run-name "$run_name"
  --include-manual
)
for dataset in "${datasets[@]}"; do args+=(--dataset "$dataset"); done
[[ -z $data_dir ]] || args+=(--data-dir "$data_dir")
[[ -z $checkpoint ]] || args+=(--checkpoint "$checkpoint")

exec bash "$script_dir/run_multistep.sh" "${args[@]}"
