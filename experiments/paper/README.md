# Paper Reproduction

This directory contains only the figure-building entry points. The original
notebooks, portable saved results, and generated figures remain in the stable
top-level paths referenced by the paper:

```text
notebooks/{mean_field,mlp,sweeps,transformer}/
results/{mlp,sweeps,transformer}/
figures/paper/
```

Rebuild every checked-in figure without rerunning training:

```bash
python experiments/paper/make_figures.py --all
python experiments/paper/make_regularization_reach_figure.py
```

The saved MLP and sweep runs use JSON manifests plus NPZ arrays and load with
`allow_pickle=False`.
