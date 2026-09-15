# Paper Reproduction

This directory contains figure and table builders and an optional result exporter. The original
notebooks, portable saved results, and generated figures remain in the stable
top-level paths referenced by the paper:

```text
notebooks/{mean_field,mlp,sweeps,transformer}/
results/{mlp,sweeps,transformer}/
figures/paper/
```

Rebuild the original figures without rerunning training:

```bash
python experiments/paper/make_figures.py --all
python experiments/paper/make_regularization_reach_figure.py
```

The saved MLP and sweep runs use JSON manifests plus NPZ arrays and load with
`allow_pickle=False`.

## V2 benchmark appendix

Run `make figures` to rebuild the 21 benchmark figures, geometry-pilot figure,
main-body tables, and experimental appendix from the saved run export. This
works offline. See [the results index](../../paper/README.md) for comparisons,
coverage limits, and how to copy the insertions into the separate manuscript.

`export_benchmarks.py` refreshes the W&B snapshot when new results are available;
it requires project access and checks that every training epoch is present.
