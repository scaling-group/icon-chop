# Evaluation entry points

The reproducible evaluation scripts are grouped by task:

- `conservation_law/run_multistep.sh` — 500-sample, 10-step rollout;
- `conservation_law/run_weno_eval.sh` — canonical WENO evaluation;
- `conservation_law/run_mfc_weno.sh` — transfer the unchanged MFC agent to WENO data/model;
- `mean_field_control/run_eval.sh` — 15 OOD MFC datasets, 100 operators each;
- `mean_field_control/run_g_res.sh` — six-dataset residual-only ablation;
- `graph_in_context/run_eval.sh` — 2023 full-year cross-region GICON sweep.

On Linux, install `uv`, then run `uv sync --no-dev` once from the workspace
root. Every wrapper uses `uv run --project --no-dev` and therefore works from
any current directory while sharing the root `.venv` and lockfile. Use plain
`uv sync` only when the optional lint and test tools are needed.

All entry points resolve the evolved program and fixed protocol from this
workspace. The only task-specific runtime inputs are the evaluation data path
and checkpoint path; both also have local defaults in the Bash wrappers.

Numerical result files always include the effective protocol and per-sample
identifiers and metrics. Every entry point runs from the current workspace,
input data, and checkpoint without reading any previously generated results.
