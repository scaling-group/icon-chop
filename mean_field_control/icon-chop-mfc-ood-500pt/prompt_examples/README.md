# `prompt_examples` Dataset

This directory contains the complete five-shot prompts associated with the selected prediction files in the adjacent `artifacts` directory:

```text
mean_field_control/icon-chop-mfc-ood-500pt/prompt_examples
mean_field_control/icon-chop-mfc-ood-500pt/artifacts
```

The two directories describe the same 75 selected evaluation cases from two different views:

- `prompt_examples` stores the complete model prompt: five demonstration pairs, one query, the query ground truth, and both predictions.
- `artifacts` stores the same query, ground truth, and predictions, together with error metrics and plot-ready arrays.

The purpose of `prompt_examples` is therefore to add the demonstration context that is not present in `artifacts`.

## Contents of This Directory

The directory contains 15 task folders and 75 `.full_prompt.npz` files:

| Item | Count |
| --- | ---: |
| Task folders | 15 |
| Full-prompt files per task | 5 |
| Full-prompt files in total | 75 |
| Demonstration pairs per file | 5 |
| Query pairs per file | 1 |

Each task originally evaluated 100 prompts. The five retained files are the five samples selected for the corresponding task in `artifacts` and `metrics`.

Example layout:

```text
prompt_examples/
└── ood_operator_l01_mfc_gparam_forward11_1/
    ├── ood_operator_l01_mfc_gparam_forward11_1.full_prompt.npz
    ├── ood_operator_l01_mfc_gparam_forward11_1__rank02.full_prompt.npz
    ├── ood_operator_l01_mfc_gparam_forward11_1__rank03.full_prompt.npz
    ├── ood_operator_l01_mfc_gparam_forward11_1__rank04.full_prompt.npz
    └── ood_operator_l01_mfc_gparam_forward11_1__rank05.full_prompt.npz
```

The file without a rank suffix is rank 1. Files ending in `__rank02` through `__rank05` are ranks 2 through 5. Ranking is based primarily on the improvement of the CHOP(MFC) prediction over the Raw-ICON prediction.

## One-to-One Match with `artifacts`

Every full-prompt file has exactly one artifact with the same base filename:

```text
prompt_examples/<dataset_name>/<base_name>.full_prompt.npz
                                  <->
artifacts/<base_name>.npz
```

For example:

```text
prompt_examples/ood_operator_l01_mfc_gparam_forward11_1/
  ood_operator_l01_mfc_gparam_forward11_1__rank02.full_prompt.npz

matches

artifacts/
  ood_operator_l01_mfc_gparam_forward11_1__rank02.npz
```

All 75 pairs have been checked. Their query inputs, query coordinates, target coordinates, ground truth, Raw-ICON predictions, and CHOP(MFC) predictions are element-for-element identical.

## What One `.full_prompt.npz` File Stores

One file contains one complete five-shot inference case:

```text
demonstration 1: demo input  -> demo target
demonstration 2: demo input  -> demo target
demonstration 3: demo input  -> demo target
demonstration 4: demo input  -> demo target
demonstration 5: demo input  -> demo target

query:           query input -> ground truth
                              -> Raw-ICON prediction
                              -> CHOP(MFC) prediction
```

The five demonstrations and the query belong to the same source equation/group and therefore represent the same hidden operator. The model uses the five demonstration pairs to infer that operator and predict the query output.

The `groundtruth` array is stored for evaluation. It is the desired query output and is not part of the input shown to the model during prediction.

## NPZ Key Reference

### Sample identity and source indexes

| Key | Shape | Meaning |
| --- | --- | --- |
| `sample_id` | `()` | Selected sample identifier, formatted as `<dataset_name>:eid_i_sid_j` |
| `dataset_name` | `()` | Name of the MFC evaluation task |
| `source_index` | `()` | Position of the source equation/group in evaluator loading order |
| `demo_indices` | `(5,)` | Indexes of the five source pairs used as demonstrations |
| `query_index` | `()` | Index of the source pair used as the query |

`source_index` identifies the equation/group in evaluator order. It does not necessarily equal the numeric value in `eid_i`. The values in `demo_indices` and `query_index` identify different input-output pairs within that equation/group.

### Five demonstration pairs

| Key | Shape | Meaning |
| --- | --- | --- |
| `demo_input_coords` | `(5, N_in, 2)` | Coordinates of the five demonstration inputs |
| `demo_input_vals` | `(5, N_in, 1)` | Values of the five demonstration input functions or fields |
| `demo_target_coords` | `(5, N_out, 2)` | Coordinates of the five demonstration targets |
| `demo_target_vals` | `(5, N_out, 1)` | Ground-truth values of the five demonstration targets |

The first dimension is the demonstration number. For example, `demo_input_vals[2]` and `demo_target_vals[2]` form the third demonstration pair.

### Query, ground truth, and predictions

| Key | Shape | Meaning |
| --- | --- | --- |
| `query_input_coords` | `(N_in, 2)` | Coordinates of the query input |
| `query_input_vals` | `(N_in, 1)` | Values of the query input function or field |
| `query_target_coords` | `(N_out, 2)` | Coordinates where the query output is evaluated |
| `groundtruth` | `(N_out, 1)` | Ground-truth query output |
| `baseline_prediction` | `(N_out, 1)` | Query output predicted by Raw ICON |
| `agent_prediction` | `(N_out, 1)` | Query output predicted by CHOP(MFC) |

`groundtruth`, `baseline_prediction`, and `agent_prediction` all correspond to the same `query_target_coords` and can therefore be compared directly.

All coordinate, value, target, and prediction arrays use `float32`. Index fields use `int64`. Identifier fields are NumPy Unicode scalars. The files do not contain an additional batch dimension.

## Meaning of `N_in` and `N_out`

The input and output sizes depend on the task mode:

| Mode | Meaning | `N_in` | `N_out` |
| --- | --- | ---: | ---: |
| `forward11` | One-dimensional input to one-dimensional output | 100 | 100 |
| `forward12` | One-dimensional input to two-dimensional output | 100 | 500 |
| `forward22` | Two-dimensional input to two-dimensional output | 500 | 500 |

A one-dimensional field contains 100 spatial points. A two-dimensional field contains 500 time-space points obtained from 10 time locations and 50 spatial locations.

Coordinates are stored in `[t, x]` order:

- initial-density curves use `[0, x]`;
- terminal-density curves use `[1, x]`;
- terminal-cost inputs `g(x)` use `[0, x]`, where the zero is padding rather than a physical time dependence;
- two-dimensional fields use their actual `[t, x]` coordinates;
- flattened two-dimensional values are time-major, with space varying fastest, and reshape to `(10, 50)` after removing the final singleton channel.

## Meaning of the Task Names

Dataset names follow:

```text
ood_operator_<length>_mfc_<family>_<mode>_1
```

| Token | Meaning |
| --- | --- |
| `l05` | Gaussian-process length scale `ell = 0.5` |
| `l03` | Gaussian-process length scale `ell = 0.3` |
| `l01` | Gaussian-process length scale `ell = 0.1` |
| `gparam` | The operator has a fixed terminal cost `g(x)` and varies the initial density |
| `rhoparam` | The operator has a fixed initial density `rho(0, x)` and varies the terminal cost |
| `forward11` | One-dimensional input to one-dimensional output |
| `forward12` | One-dimensional input to two-dimensional output |
| `forward22` | Two-dimensional input to two-dimensional output |

The physical fields are:

- `rho(t, x)`: the density field;
- `rho(0, x)`: the initial density;
- `rho(1, x)`: the terminal density;
- `g(x)`: the terminal cost function.

The five mappings represented by the files are:

| Family | Mode | Input stored in the prompt | Target stored in the prompt |
| --- | --- | --- | --- |
| `gparam` | `forward11` | Initial density `rho(0, x)` | Terminal density `rho(1, x)` |
| `gparam` | `forward12` | Initial density `rho(0, x)` | Second-half density field `rho(t, x)` |
| `gparam` | `forward22` | First-half density field `rho(t, x)` | Second-half density field `rho(t, x)` |
| `rhoparam` | `forward11` | Terminal cost `g(x)` | Terminal density `rho(1, x)` |
| `rhoparam` | `forward12` | Terminal cost `g(x)` | Second-half density field `rho(t, x)` |

There are three `gparam` modes and two `rhoparam` modes at each of the three length scales, giving 15 task folders. This collection does not contain `rhoparam/forward22`.

## Fields Shared with and Added to `artifacts`

The two file types retain different parts of the same selected case:

| Contents | `prompt_examples/*.full_prompt.npz` | `artifacts/*.npz` |
| --- | --- | --- |
| `sample_id`, `dataset_name` | Yes | Yes |
| Query input and coordinates | Yes | Yes |
| Query target coordinates and ground truth | Yes | Yes |
| Raw-ICON and CHOP(MFC) predictions | Yes | Yes |
| Five complete demonstration pairs | Yes | No |
| `source_index`, `demo_indices`, `query_index` | Yes | No |
| Relative errors and improvement score | No | Yes |
| Pointwise absolute-error arrays | No | Yes |
| Plot-ready line axes and heatmap grids | No | Yes |

In short, `artifacts` is the query-result and plotting view, while `prompt_examples` is the full-prompt view of the same selected sample.

## Historical `all` Token in `l05` Artifacts

The 25 `l05` artifact files use the historical token `all` inside their `dataset_name` and `sample_id` fields. Their filenames, metric JSON files, and matching full prompts use `l05`.

```text
artifact filename:         ood_operator_l05_mfc_gparam_forward11_1.npz
artifact dataset_name:     ood_operator_all_mfc_gparam_forward11_1
full-prompt dataset_name:  ood_operator_l05_mfc_gparam_forward11_1
```

Here, `all` and `l05` refer to the same `ell = 0.5` task. The `eid/sid` suffixes and all shared numeric arrays still match. Files should be joined by their local base filename and rank, not only by the artifact's internal `dataset_name`.

## Scope of the Saved Data

Each full-prompt NPZ contains only the five demonstration pairs and one query used for the selected evaluation case. It does not contain:

- the original HDF5 file;
- the source HDF5 `equation` text;
- every candidate pair in the source equation/group;
- the other non-selected evaluation prompts;
- aggregate task metrics.

The complete per-sample and aggregate metrics remain in `../metrics` and `../summary.json`. The purpose of this directory is specifically to preserve the full demonstration context for the 75 query results stored in `../artifacts`.
