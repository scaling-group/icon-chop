#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_eval.sh [options]

Options:
  --data-dir PATH
  --checkpoint PATH
  --model-region REGION       Default: auto
  --eval-region REGION        Default: auto
  --delta HOURS               Repeat to override 1,12,24,36,48,72.
  --strategy NAME             Repeat to override random,retrieval.
  --device DEVICE             Default: cpu
  --output-dir PATH
  --force
  -h, --help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../.." && pwd)"
cloud_root="$(dirname -- "$workspace_root")"

data_dir="$cloud_root/chop_data/eval/eval_air"
checkpoint="$workspace_root/ckpt/air_quality/bthsa/multiple_demo_num5_best/checkpoints/step_90000.ckpt"
model_region=auto
eval_region=auto
deltas=(1 12 24 36 48 72)
deltas_overridden=0
strategies=(random retrieval)
strategies_overridden=0
device=cpu
output_dir=""
force=0

while (($#)); do
  case "$1" in
    --data-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; data_dir=$2; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; checkpoint=$2; shift 2 ;;
    --model-region) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; model_region=$2; shift 2 ;;
    --eval-region) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; eval_region=$2; shift 2 ;;
    --delta)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      if ((deltas_overridden == 0)); then deltas=(); deltas_overridden=1; fi
      deltas+=("$2")
      shift 2
      ;;
    --strategy)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      if ((strategies_overridden == 0)); then strategies=(); strategies_overridden=1; fi
      strategies+=("$2")
      shift 2
      ;;
    --device) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; device=$2; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; output_dir=$2; shift 2 ;;
    --force) force=1; shift ;;
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
  --model-region "$model_region"
  --eval-region "$eval_region"
  --device "$device"
  --deltas "${deltas[@]}"
  --strategies "${strategies[@]}"
)
[[ -z $output_dir ]] || args+=(--output-dir "$output_dir")
((force == 0)) || args+=(--force)

exec "${args[@]}"
