#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_g_res.sh [options]

Options:
  --data-dir PATH
  --checkpoint PATH
  --batch-size N             Default: 1
  --num-samples-per-dataset N
  --device DEVICE            Default: auto
  --output-dir PATH
  -h, --help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../.." && pwd)"

data_dir=""
checkpoint=""
batch_size=1
num_samples_per_dataset=0
device=auto
output_dir=""

while (($#)); do
  case "$1" in
    --data-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; data_dir=$2; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; checkpoint=$2; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; batch_size=$2; shift 2 ;;
    --num-samples-per-dataset) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; num_samples_per_dataset=$2; shift 2 ;;
    --device) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; device=$2; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; output_dir=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

args=(
  --program "$workspace_root/mean_field_control/icon_chop_mfc_code/seed/initial_program_g_res_only.py"
  --pattern '*rhoparam*.h5'
  --batch-size "$batch_size"
  --num-samples-per-dataset "$num_samples_per_dataset"
  --device "$device"
)
[[ -z $data_dir ]] || args+=(--data-dir "$data_dir")
[[ -z $checkpoint ]] || args+=(--checkpoint "$checkpoint")
[[ -z $output_dir ]] || args+=(--output-dir "$output_dir")

exec bash "$script_dir/run_eval.sh" "${args[@]}"
