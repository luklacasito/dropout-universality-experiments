# Code map

The repository has two main layers: experiment commands in `experiments/`, and
reusable Python code in `src/dropout_mft/`. Install with `pip install -e .` so
scripts and notebooks import the same source.

## Follow one benchmark trial

1. **Choose a study.** [`__main__.py`](../src/dropout_mft/experiments/benchmark/__main__.py)
   routes `STUDY COMMAND` to the study module or the common command handler.
2. **Plan the comparison.** A study module creates
   [`BenchmarkTrialSpec`](../src/dropout_mft/experiments/benchmark/protocol.py)
   objects. Each describes one dataset, architecture, profile, learning rate,
   dropout budget, and seed. Its contents determine the trial ID.
3. **Save the plan.** [`workflow.py`](../src/dropout_mft/experiments/benchmark/workflow.py)
   writes an immutable manifest and records the source identity. Later stages
   read the previous stage's validation selections.
4. **Load the data.** [`datasets.py`](../src/dropout_mft/experiments/benchmark/datasets.py)
   creates a `DatasetBundle` with training, validation, and test splits. Different
   architecture views retain the same split identity.
5. **Train.** [`cli.py`](../src/dropout_mft/experiments/benchmark/cli.py) iterates
   over assigned trials. `run_benchmark_trial` in `protocol.py` builds the model,
   sets random streams, and calls [`train_model`](../src/dropout_mft/training.py).
6. **Save or resume.** [`results.py`](../src/dropout_mft/results.py) writes JSON
   metadata and NumPy arrays into an NPZ file. Resume checks the trial, data,
   randomness, source identity, and any requested checkpoint before reusing it.
7. **Select and summarize.** `workflow.py` collects results. Study handlers select
   hyperparameters on validation loss and compare confirmation runs paired by seed.

```text
study definition → trial manifest → dataset + model → training → result file
                          ↑                              ↓
                   next-stage plan ← validation selection
```

## Where changes belong

| You want to change… | Start with… |
|---|---|
| A benchmark's rates, seeds, epochs, or comparison arms | Its module in [`experiments/benchmark`](../src/dropout_mft/experiments/benchmark/) |
| Dataset loading, preprocessing, or split rules | [`benchmark/datasets.py`](../src/dropout_mft/experiments/benchmark/datasets.py) |
| Common benchmark execution or CLI arguments | [`benchmark/cli.py`](../src/dropout_mft/experiments/benchmark/cli.py) |
| Manifest collection and shared selection | [`benchmark/workflow.py`](../src/dropout_mft/experiments/benchmark/workflow.py) |
| Dropout profiles and budget constraints | [`schedules.py`](../src/dropout_mft/schedules.py) and [`fields.py`](../src/dropout_mft/fields.py) |
| MLP parameterization and forward passes | [`models.py`](../src/dropout_mft/models.py) |
| Benchmark-specific model construction | [`benchmark/protocol.py`](../src/dropout_mft/experiments/benchmark/protocol.py) |
| Shared training and evaluation | [`training.py`](../src/dropout_mft/training.py) |
| Deterministic random streams | [`randomization.py`](../src/dropout_mft/randomization.py) |
| Paired tests and confidence intervals | [`statistics.py`](../src/dropout_mft/statistics.py) |
| Source snapshots and reproducibility checks | [`provenance.py`](../src/dropout_mft/provenance.py) |
| Optional W&B tracking | [`wandb_tracking.py`](../src/dropout_mft/wandb_tracking.py) |
| Paper figures | [`experiments/paper`](../experiments/paper/) |

The names under `src/dropout_mft/experiments/` identify scientific studies, not
separate frameworks. Add a function or a study parameter when that is enough.
Share code when the behavior really is the same across studies.

## Why some code stays separate

- **Legacy training** preserves the historical initialization and random-number
  order. A similar-looking loop can describe a different experiment.
- **Scale transfer** has its own trial definition and phase-specific selection
  rules. Its analysis includes standard and μP parameterizations.
- **Optiver** has its own feature pipeline and time-based splits.
- **RNN pilots** are standalone source snapshots with their own test suites and
  source hashes. See the [RNN guide](rnn-studies.md).
- **Profile sampling conventions** differ: endpoint samples and cell-center
  samples can have the same mean but different probabilities at each layer.
  Their distinction is part of the protocol.

## Files produced by benchmark runs

A typical `runs/example/` contains:

- `manifests/`: fixed trial lists and provenance metadata;
- `selections/`: validation-selected hyperparameters for the next stage;
- `trials/`: completed result files grouped by stage;
- `checkpoints/`: optional best-validation model checkpoints;
- `summary/`: aggregate comparisons;
- `wandb/`: optional tracking files.

Treat trial IDs and source hashes as experiment identities. A refactored source
tree has a new identity even when deterministic tests show equivalent behavior.
Use the original snapshot to resume an existing frozen run.
