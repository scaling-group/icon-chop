# Mean-field-control evaluation

This entry runs the fixed MFC OOD evaluation protocol:

- all 15 OOD `.h5` datasets (100 operators in each file);
- `numpy.random.RandomState(0)` operator order;
- five demonstrations and one query selected with `torch.Generator(seed=source_index)`;
- deterministic `10 x 50 = 500` point grids;
- batch size 1.

From the workspace root, the local paths are already the defaults:

```bash
bash ./eval/mean_field_control/run_eval.sh
```

With explicit paths:

```bash
bash ./eval/mean_field_control/run_eval.sh \
  --data-dir /data/eval_mfc \
  --checkpoint /checkpoints/mfc.ckpt
```

Each run writes `summary.json`, one full per-dataset JSON under `metrics/`,
and per-sample CSV files.

For a one-sample smoke test, pass `--num-samples-per-dataset 1` and narrow
`--pattern` to one file.

The residual-only ablation has a dedicated entry point selecting the six
`rhoparam` datasets:

```bash
bash ./eval/mean_field_control/run_g_res.sh
```

Use `--num-samples-per-dataset 1` for a reduced smoke test.
