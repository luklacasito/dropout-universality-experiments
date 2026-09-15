# Benchmark evidence used in the paper

`confirmation.json.gz` contains 618 completed confirmation runs exported from
[the W&B project](https://wandb.ai/llama2cmu/dropout-universality-benchmarks) on
2026-09-15. It stores individual runs, not values transcribed from a chart.

Each run includes its ID and source URL, scientific configuration, split hash,
recorded summary metrics, and every training/validation epoch. Export checks
require full epoch coverage. Machine-specific Slurm fields and W&B internal
metadata are omitted. Some original runs lack a source-provenance hash; the
export preserves that absence instead of inventing an identity.

## Rebuild offline

```bash
python experiments/paper/make_appendix.py
```

To deliberately refresh the snapshot from W&B (authentication required):

```bash
python experiments/paper/export_benchmarks.py
```

## Comparison rules

- Keep each cohort, dataset, architecture, split, and training horizon separate.
- Primary benchmark test metrics use the minimum-validation-loss checkpoint.
  Fixed-final-epoch test metrics, where recorded, are separate appendix columns.
- `winner` means the nonuniform profile with the lowest observed mean test loss
  among candidates with the complete expected seed set. This is a descriptive
  retrospective comparison, not an independent test of a selected schedule.
- Uniform and the winning profile must have identical paired seed sets. SEMs
  summarize seed variation; they do not correct for choosing a winner.
- Every available arm appears in the appendix, including no dropout, losses to
  uniform, and incomplete arms. Amazon 20k MLP has only four runs for each linear
  profile; these remain visible but cannot win the main comparison.
- The two linear follow-ups have no recovered matching uniform baseline in this
  export. Their endpoints are reported without a uniform improvement claim.
- Hyperparameters, including mean dropout, were tuned separately by profile.
  These are tuned-recipe comparisons, not all fixed-budget allocation tests.

The original CIFAR experiments use their original saved arrays and final-epoch
endpoints and are summarized separately. The validation-only geometry pilot has
no test endpoint and is not given a fabricated test-result row.
