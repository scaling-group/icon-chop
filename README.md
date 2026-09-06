# Chain of Operators: An Inference-Time Harness for In-Context Operator Learning

Paper: [arXiv:2606.12318](https://arxiv.org/abs/2606.12318)

This repository contains the evaluation code, data-generation utilities,
checkpoints, and saved results used for ICON-CHOP experiments.

- conservation laws;
- mean-field control (MFC); and
- graph-in-context air-quality forecasting (GICON).


## Repository layout

| Directory | Contents |
| --- | --- |
| `ckpt/` | Checkpoints used by the evaluation entry points |
| `data/` | Data-generation utilities and dataset instructions |
| `eval/` | Reproducible evaluation entry points |
| `conservation_law/` | Conservation-law ICON-CHOP code and saved results |
| `mean_field_control/` | MFC ICON-CHOP code, prompt examples, and saved results |
| `graph_in_context/` | GICON-CHOP code and saved results |
| `cross_transfer/` | Cross-task MFC-to-WENO evaluation results |

The READMEs inside these directories document task-specific protocols, file
formats, and result layouts.

## Environment setup

The project requires Python 3.12 and uses `uv` to manage the environment and
install the dependencies recorded in `uv.lock`. From the repository root, run:

```bash
uv sync
```

The evaluation entry points should be run from the repository root. They use
the shared environment created by `uv`.

## Data

Conservation-law and MFC generation scripts are included under `data/`:

- [`data/conservation_law/README.md`](data/conservation_law/README.md)
- [`data/mean_filed/README.md`](data/mean_filed/README.md)

The `mean_filed` directory name is retained for compatibility. Air-quality
data are distributed separately; see [`data/air/README.md`](data/air/README.md)
for the download location.

## Evaluation

Evaluation entry points are organized by task under `eval/`:

- `eval/conservation_law/` for autoregressive conservation-law evaluation and
  cross-task transfer;
- `eval/mean_field_control/` for MFC out-of-distribution evaluation and
  ablations; and
- `eval/graph_in_context/` for cross-region air-quality evaluation.

The Bash wrappers expose task-specific data, checkpoint, and evaluation
options. See [`eval/README.md`](eval/README.md) and the README in each task
directory for commands and protocol details.
