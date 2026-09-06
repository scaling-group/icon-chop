#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_eval.sh [options]

Options:
  --data-dir PATH
  --checkpoint PATH
  --program PATH
  --pattern GLOB             Default: *.h5
  --batch-size N             Default: 1
  --num-samples-per-dataset N
  --device DEVICE            Default: auto
  --output-dir PATH
  -h, --help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../.." && pwd)"
cloud_root="$(dirname -- "$workspace_root")"

data_dir="$cloud_root/chop_data/eval/eval_mfc"
checkpoint="$workspace_root/ckpt/mean_field/mfc.ckpt"
program=""
pattern='*.h5'
batch_size=1
num_samples_per_dataset=0
device=auto
output_dir=""

while (($#)); do
  case "$1" in
    --data-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; data_dir=$2; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; checkpoint=$2; shift 2 ;;
    --program) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; program=$2; shift 2 ;;
    --pattern) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; pattern=$2; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; batch_size=$2; shift 2 ;;
    --num-samples-per-dataset) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; num_samples_per_dataset=$2; shift 2 ;;
    --device) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; device=$2; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; output_dir=$2; shift 2 ;;
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
  uv run --project "$workspace_root" --no-dev python "$script_dir/evaluate.py"
  --data-dir "$data_dir"
  --ckpt "$checkpoint"
  --pattern "$pattern"
  --batch-size "$batch_size"
  --seed 0
  --device "$device"
)
[[ -z $program ]] || args+=(--program "$program")
((num_samples_per_dataset <= 0)) || args+=(--num-samples-per-dataset "$num_samples_per_dataset")
[[ -z $output_dir ]] || args+=(--output-dir "$output_dir")

exec "${args[@]}"
