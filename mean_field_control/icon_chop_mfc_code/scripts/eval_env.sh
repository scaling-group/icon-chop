#!/usr/bin/env bash

# Source this file before running evaluation, or copy these exports into your job script.
#
# Recommended model:
# 1. create one shared uv-managed environment outside the git checkout
# 2. sync dependencies into that environment once
# 3. activate that environment before launching Escher Loop
#
# Example:
#   uv sync
#   source .venv/bin/activate
#
# `uv sync` installs the cu126 PyTorch wheel configured in pyproject.toml.

: "${ICON_CHOP_CKPT_PATH:?Set ICON_CHOP_CKPT_PATH before sourcing this file}"
: "${ICON_CHOP_DATA_DIR:?Set ICON_CHOP_DATA_DIR before sourcing this file}"
export ICON_CHOP_CKPT_PATH
export ICON_CHOP_DATA_DIR
export ICON_CHOP_DATA_GLOB='test_ood_operator_mfc_gparam_forward22_merged_1.h5'

# Evaluator controls:
# - ICON_CHOP_BATCH_SIZE controls batch collation in eval/eval_core.py
# - ICON_CHOP_NUM_SAMPLES limits how many operator instances are evaluated
# - ICON_CHOP_SEED controls dataset permutation and demo/query splitting
export ICON_CHOP_DEVICE=cuda
export ICON_CHOP_BATCH_SIZE=10
export ICON_CHOP_NUM_SAMPLES=100
export ICON_CHOP_SEED=0
export ICON_CHOP_DEMO_NUM=5
export ICON_CHOP_MAX_LEN=100
export ICON_CHOP_K_DIM=2
export ICON_CHOP_MODEL_IN_FEATURES=3

# Optional: ask the repo-native evaluator to persist the full payload somewhere specific.
# export ICON_CHOP_OUTPUT_JSON=/abs/path/to/output/summary.json
