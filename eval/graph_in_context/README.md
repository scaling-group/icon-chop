# Graph-in-context evaluation

This entry point runs the final GICON-CHOP post-evaluation protocol:

- 2023-01-01 00:00 through 2023-12-31 23:00 query year;
- 2016-2022 context years;
- direct `+delta_t` prediction for `delta_t = 1, 12, 24, 36, 48, 72` hours;
- random and cosine-retrieval demonstration contexts;
- seed 42, five demonstrations, 24-frame windows, batch size 16;
- full per-sample metrics and the original aggregate RMSE-ratio score.

The model region is inferred from the checkpoint path (`bthsa` or `yrd`), and
the default evaluation region is the other region. Thus a normal run needs
only the data directory and checkpoint:

```bash
bash ./eval/graph_in_context/run_eval.sh \
  --data-dir /data/eval_air \
  --checkpoint /checkpoints/bthsa/step_90000.ckpt
```

The local data and BTHSA checkpoint are defaults, so this also works here:

```bash
bash ./eval/graph_in_context/run_eval.sh
```

Run the second cross-region pairing with the YRD checkpoint:

```bash
bash ./eval/graph_in_context/run_eval.sh \
  --checkpoint ./ckpt/air_quality/yrd/multiple_demo_num5_best/checkpoints/step_90000.ckpt
```

The default is CPU for deterministic evaluation. Use `--device cuda` for a
faster run when small device-level floating-point differences are acceptable.

Retrieval indices are deterministic and cached under `cache/retrieval_indices`.
If they are missing, the script builds them with the local cosine-retrieval
implementation in `retrieval.py`; this preprocessing is RAM- and CPU-intensive
for a full seven-year context. Re-running with the same cache resumes directly
from evaluation. The full-year metric and batching protocol is local in
`direct_year.py`.

Build the BTHSA 2023 indices independently for selected horizons with:

```bash
uv run --no-dev python ./eval/graph_in_context/retrieval.py \
  --data-dir /data/eval_air \
  --eval-region bthsa \
  --delta-t 1 2 3 4 5 6 12 24 \
  --backend numpy
```

The generated `.npz` and metadata `.json` files use the same names and cache
directory expected by `evaluate.py`.

On Windows, use an ASCII-only data path if `netCDF4` cannot open a path that
contains non-ASCII characters.

Each leaf writes `raw_evaluator.json` (all hourly samples) and `score.yaml`.
The run root also contains `summary.csv`, `summary.json`, `summary.md`, and the
exact protocol in `run_config.json`.

For a quick smoke test:

```bash
bash ./eval/graph_in_context/run_eval.sh \
  --delta 24 --strategy random --device cuda
```

Install and synchronize the evaluation environment once with:

```bash
uv sync --no-dev
```
