# Mean-field data generation

This directory is a standalone mean-field-control data-generation package. The existing directory name, `mean_filed`, is retained for compatibility.

Generate the 15 MFC evaluation files:

```bash
PYTHON_BIN=python DEVICE=cuda ./run_eval.sh /path/to/eval_mfc
```

Generate the merged evolution file:

```bash
PYTHON_BIN=python DEVICE=cuda ./run_evolve.sh /path/to/evolve/mfc
```

