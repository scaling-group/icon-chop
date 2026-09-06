#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_multistep.sh [options]

Options:
  --data-dir PATH
  --checkpoint PATH
  --program PATH
  --dataset FILE              Repeat to select multiple datasets.
  --num-samples N             Default: 500
  --rollout-steps N           Default: 10
  --batch-size N              Default: 10
  --device DEVICE             Default: auto
  --run-name NAME
  --include-manual
  -h, --help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../.." && pwd)"
cloud_root="$(dirname -- "$workspace_root")"

data_dir="$cloud_root/chop_data/eval/eval_conservation"
checkpoint="$workspace_root/ckpt/conservation_law/conservation.ckpt"
program="$workspace_root/conservation_law/icon_chop_weno_code/initial_program.py"
datasets=(test_seq_sin.pt test_seq_tanh.pt test_seq_u2.pt)
datasets_overridden=0
num_samples=500
rollout_steps=10
batch_size=10
device=auto
run_name=""
include_manual=0

while (($#)); do
  case "$1" in
    --data-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; data_dir=$2; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; checkpoint=$2; shift 2 ;;
    --program) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; program=$2; shift 2 ;;
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
    --include-manual) include_manual=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null 2>&1 || {
  echo "uv is required; install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 127
}

export KMP_DUPLICATE_LIB_OK=TRUE
args=(
  uv run --project "$workspace_root" --no-dev python "$script_dir/multistep_eval.py"
  --data-dir "$data_dir"
  --ckpt "$checkpoint"
  --icon-chop-program "$program"
  --num-samples "$num_samples"
  --rollout-steps "$rollout_steps"
  --batch-size "$batch_size"
  --seed 0
  --device "$device"
)
if ((${#datasets[@]})); then args+=(--datasets "${datasets[@]}"); fi
[[ -z $run_name ]] || args+=(--run-name "$run_name")
((include_manual == 0)) || args+=(--include-manual)

exec "${args[@]}"
