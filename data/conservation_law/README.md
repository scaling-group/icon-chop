# Conservation-law data generation

This directory is a standalone conservation-law data-generation package.

Generate the nine evaluation files:

```bash
PYTHON_BIN=python DEVICE=cuda ./run_eval.sh /path/to/eval_conservation
```

Generate `test_seq.pt` for evolution:

```bash
PYTHON_BIN=python DEVICE=cuda ./run_evolve.sh /path/to/evolve/conservation
```


