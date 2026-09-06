#!/usr/bin/env bash
#PBS -q auto
#PBS -l select=1:ncpus=1
#PBS -N icon-chop-chain-cpu
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -k oed

set -euo pipefail

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR must be set by qsub}"

source scripts/eval_env.sh

export ICON_CHOP_DEVICE=cpu
export ICON_CHOP_OUTPUT_JSON="${PBS_O_WORKDIR}/results/eval/full_cpu.json"

"${PYTHON_BIN:-python}" evaluate.py --program seed/initial_program.py
