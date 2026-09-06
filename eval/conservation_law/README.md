# Conservation-law multi-step evaluation

This directory contains a self-contained evaluation entry point for the 1D
conservation-law checkpoint. It evaluates two autoregressive rollouts:

- `raw_icon`: the checkpoint prediction is fed directly into the next step.
- `icon_chop`: `conservation_law/icon_chop_weno_code/initial_program.py` is
  applied at every step and its prediction is fed into the next step.

The first five sequence pairs are fixed demonstrations. Frame 5 is the initial
query state, and frames 6 through 15 are the default 10-step targets.

## Run all 500 samples

From the workspace root:

```bash
bash ./eval/conservation_law/run_multistep.sh
```

The default batch size is 10. Reduce it with `--batch-size 1` when GPU memory is
limited. Batch size does not change the sample selection or ordering.

Each invocation creates a new directory under `eval/conservation_law/logs/`.
For each of the three datasets it writes:

- `rollout_per_sample_metrics.csv`
- `rollout_horizon_summary.csv`
- `rollout_method_summary.csv`
- `rollouts.pt` containing targets, raw ICON predictions, ICON-Chop predictions,
  sample IDs, and source indices

The run root also contains `summary.json` and `run.log`.

## Seed

The seed is fixed to 0 by the Bash entry point. Sample selection uses
`numpy.random.RandomState(0).permutation(500)`.


## WENO evaluation

The following entry runs raw ICON, manual prompt min-max scaling, and the
WENO-native evolved program on the three canonical datasets:

```bash
bash ./eval/conservation_law/run_weno_eval.sh
```

## MFC-to-WENO transfer

The current MFC operator-chain is loaded directly from
`mean_field_control/icon_chop_mfc_code/seed/initial_program.py`. Run it against
the frozen WENO checkpoint and the same canonical data protocol with:

```bash
bash ./eval/conservation_law/run_mfc_weno.sh
```

This entry evaluates raw ICON, manual prompt min-max scaling, and the current
MFC operator-chain for 500 samples and a 10-step rollout.
