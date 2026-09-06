# Conservation-law maximum-rollout results

This directory contains saved evaluation results for three conservation-law
datasets:

- `test_seq_sin_150step_legacy20aligned`
- `test_seq_tanh_150step_legacy20aligned`
- `test_seq_u2_150step_legacy20aligned`

The stored results compare two method labels, `raw_icon` and `icon_chop`.
Model weights and the input datasets are not included in this directory.

## Relationship to `icon-chop-weno-result`

This directory is the long-horizon extension of the stored 10-step rollout
results in `../icon-chop-weno-result/ms10_test_seq_{sin,tanh,u2}`. The
relationship is:

| `icon-chop-weno-result` | This directory |
| --- | --- |
| `test_seq_{flux}.pt` | `test_seq_{flux}_150step_legacy20aligned.pt` |
| 20-frame sequence | 150-frame sequence |
| 10 rollout steps | 144 rollout steps |
| `baseline` | `raw_icon` |
| `final_best_program` | `icon_chop` |
| 500 samples, seed 0 | 500 samples, seed 0 |
| Batch size 10 | Batch size 20 |
| Includes `manual_linear_prompt_scale` | Does not include that method |

The dataset generator in `data/conservation_law/generate_conservation.py`
creates each 20-frame `test_seq_{flux}.pt` dataset from frames 0 through 19 of
the corresponding 150-frame `legacy20aligned` dataset. Consequently, the two
result sets share the same demonstrations and the same first 10 rollout target
frames.

The stored CSV files confirm the result correspondence. After mapping
`baseline` to `raw_icon` and `final_best_program` to `icon_chop`, all 30,000
common method/sample/horizon rows are present with the same sample IDs for the
three fluxes and horizons 1 through 10. The maximum absolute differences in
the primary metrics are:

| Metric | Maximum absolute difference |
| --- | ---: |
| MSE | `1.758337e-6` |
| Relative L2 | `5.602837e-6` |
| L1 | `1.035631e-6` |

Thus the shared 10-step interval agrees numerically, but it is not bit-for-bit
identical. The two result sets record different historical absolute paths and
do not contain the referenced checkpoint, input data, or program files, so
those external files cannot be compared directly from the result directories.

## Directory layout

```text
max_rollout/
|-- README.md
|-- summary.json
|-- test_seq_sin_150step_legacy20aligned/
|   |-- rollouts.pt
|   |-- rollout_per_sample_metrics.csv
|   |-- rollout_horizon_summary.csv
|   `-- rollout_method_summary.csv
|-- test_seq_tanh_150step_legacy20aligned/
|   |-- rollouts.pt
|   |-- rollout_per_sample_metrics.csv
|   |-- rollout_horizon_summary.csv
|   `-- rollout_method_summary.csv
`-- test_seq_u2_150step_legacy20aligned/
    |-- rollouts.pt
    |-- rollout_per_sample_metrics.csv
    |-- rollout_horizon_summary.csv
    `-- rollout_method_summary.csv
```

## Recorded run configuration

The following values are recorded in `summary.json`:

| Field | Value |
| --- | --- |
| Datasets | `sin`, `tanh`, and `u2` result directories listed above |
| Full sequence shape | `[500, 150, 100, 1]` for each dataset |
| Evaluated samples | 500 per dataset |
| Demonstrations | 5 |
| Maximum rollout steps | 144 |
| Evaluated rollout steps | 144 |
| Sample-ordering seed | 0 |
| Batch size | 20 |
| Device | `cuda` |
| `max_rollout_cap` | `null` |
| Saved predictions | `true` |
| `disable_mha_fastpath` | `false` |

Each saved prediction tensor therefore has dimensions
`[sample, rollout_step, spatial_point, channel]`.

## `summary.json`

This file stores the run metadata and per-dataset summaries. Its top-level
keys are:

| Key | Type | Contents |
| --- | --- | --- |
| `output_dir` | string | Output directory recorded when the results were generated |
| `checkpoint` | string | Checkpoint path recorded for the run |
| `icon_chop_program` | string | ICON-Chop program path recorded for the run |
| `data_dir` | string | Input-data directory recorded for the run |
| `datasets` | list of objects | Per-dataset metadata, summary metrics, and output paths |
| `num_samples` | integer | Requested samples per dataset |
| `seed` | integer | Sample-ordering seed |
| `batch_size` | integer | Inference batch size |
| `demo_num` | integer | Number of demonstrations |
| `max_rollout_cap` | integer or null | Recorded rollout cap |
| `device` | string | Recorded inference device |
| `save_predictions` | boolean | Whether rollout tensors were saved |
| `disable_mha_fastpath` | boolean | Recorded MHA fast-path setting |
| `summary_json` | string | Output path recorded for this JSON file |

Each object in `datasets` contains:

| Key | Type | Contents |
| --- | --- | --- |
| `dataset` | string | Dataset name |
| `data_path` | string | Recorded input dataset path |
| `num_samples` | integer | Number of evaluated samples |
| `sequence_shape` | list of integers | Full input shape `[N, T, P, C]` |
| `method_summary` | list of objects | Full-rollout metrics for both methods |
| `per_sample_csv` | string | Recorded path to the per-sample CSV |
| `horizon_summary_csv` | string | Recorded path to the horizon-summary CSV |
| `rollouts_pt` | string | Recorded path to the rollout tensor file |
| `maximum_rollout_steps` | integer | Maximum recorded rollout length |
| `evaluated_rollout_steps` | integer | Evaluated rollout length |

The paths stored in `summary.json` are absolute paths from the machine that
generated the results. Use the files in this directory after moving or copying
the result directory.

## Dataset-level files

All three dataset directories have the same four files and schemas.

### `rollouts.pt`

This is a PyTorch-serialized dictionary with the following keys:

| Key | Type or shape | Contents |
| --- | --- | --- |
| `dataset` | string | Dataset name |
| `data_path` | string | Recorded absolute input path |
| `sample_ids` | list of 500 strings | Sample labels in saved tensor order |
| `source_indices` | int64 tensor `[500]` | Original sample indices in saved tensor order |
| `seed` | integer | `0` |
| `demo_num` | integer | `5` |
| `rollout_steps` | integer | `144` |
| `targets` | float32 tensor `[500, 144, 100, 1]` | Saved target trajectory |
| `raw_icon` | float32 tensor `[500, 144, 100, 1]` | Saved `raw_icon` trajectory |
| `icon_chop` | float32 tensor `[500, 144, 100, 1]` | Saved `icon_chop` trajectory |

No `manual_linear_prompt_scale` key is present in these files.

### `rollout_per_sample_metrics.csv`

Each file has 144,000 data rows:
`2 methods x 500 samples x 144 rollout steps`.

| Column | Contents |
| --- | --- |
| `method` | `raw_icon` or `icon_chop` |
| `dataset` | Dataset name |
| `sample_id` | Sample label |
| `source_index` | Original sample index |
| `horizon` | One-based rollout step, from 1 through 144 |
| `raw_icon_mse` | `raw_icon` MSE for the same sample and horizon |
| `raw_icon_rel_l2` | `raw_icon` relative L2 error for the same sample and horizon |
| `raw_icon_l1` | `raw_icon` mean absolute error for the same sample and horizon |
| `success` | Success flag; every row is `True` in the stored files |
| `error` | Error text; empty in the stored files |
| `mse` | MSE for the row's method |
| `rel_l2` | Relative L2 error for the row's method |
| `l1` | Mean absolute error for the row's method |
| `improvement_ratio` | `raw_icon_rel_l2 / rel_l2` |
| `relative_improve` | `(raw_icon_rel_l2 - rel_l2) / raw_icon_rel_l2` |

### `rollout_horizon_summary.csv`

Each file has 288 data rows: `2 methods x 144 rollout steps`.

| Column | Contents |
| --- | --- |
| `dataset` | Dataset name |
| `method` | `raw_icon` or `icon_chop` |
| `horizon` | One-based rollout step |
| `num_samples` | Number of samples aggregated at that horizon |
| `combined_score` | `mean_raw_icon_rel_l2 / mean_rel_l2` |
| `mean_rel_l2` | Mean relative L2 error for the method |
| `mean_raw_icon_rel_l2` | Mean `raw_icon` relative L2 baseline |
| `mean_mse` | Mean MSE for the method |
| `mean_raw_icon_mse` | Mean `raw_icon` MSE baseline |
| `mean_l1` | Mean absolute error for the method |
| `mean_raw_icon_l1` | Mean `raw_icon` absolute-error baseline |
| `win_rate_vs_raw_icon` | Fraction of rows with `rel_l2 <= raw_icon_rel_l2` |
| `mean_relative_improve_vs_raw_icon` | Mean `relative_improve` at that horizon |

### `rollout_method_summary.csv`

Each file has two data rows, one for each method, aggregated over all 500
samples and all 144 rollout steps.

| Column | Contents |
| --- | --- |
| `dataset` | Dataset name |
| `method` | `raw_icon` or `icon_chop` |
| `num_predictions` | Number of sample-step predictions; 72,000 per method |
| `combined_score` | `mean_raw_icon_rel_l2 / mean_rel_l2` |
| `mean_rel_l2` | Mean relative L2 error |
| `mean_raw_icon_rel_l2` | Mean `raw_icon` relative L2 baseline |
| `mean_mse` | Mean MSE |
| `mean_l1` | Mean absolute error |
| `win_rate_vs_raw_icon` | Fraction of predictions with `rel_l2 <= raw_icon_rel_l2` |
| `mean_relative_improve_vs_raw_icon` | Mean per-prediction `relative_improve` |

## Result snapshot

These values are copied from the three `rollout_method_summary.csv` files:

| Dataset | Method | Mean relative L2 | Mean MSE | Win rate vs Raw ICON | Combined score |
| --- | --- | ---: | ---: | ---: | ---: |
| `sin` | `raw_icon` | 1.066876 | 0.134028 | 1.000000 | 1.000000 |
| `sin` | `icon_chop` | 0.174383 | 0.007887 | 0.903750 | 6.118000 |
| `tanh` | `raw_icon` | 0.566457 | 0.098015 | 1.000000 | 1.000000 |
| `tanh` | `icon_chop` | 0.374945 | 0.041197 | 0.814708 | 1.510772 |
| `u2` | `raw_icon` | 1.140375 | 0.119754 | 1.000000 | 1.000000 |
| `u2` | `icon_chop` | 0.314665 | 0.041307 | 0.864667 | 3.624093 |

The `raw_icon` win rate is 1 because those rows are compared with their own
baseline. The `icon_chop` rows contain the cross-method win rates.
