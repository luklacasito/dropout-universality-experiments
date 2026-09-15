# Dropout Universality Experiments

Code and saved results for **Dropout Universality: Scaling Laws and Optimal
Scheduling at the Edge-of-Chaos** ([paper](https://arxiv.org/abs/2605.21648)).

The experiments ask how the amount and placement of dropout across network
layers affect training and generalization. This repo contains the paper
reproductions and follow-up studies on MLPs, transformers, and recurrent models.

## Get started

Use **Python 3.11**. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Check model shapes and dropout budgets on CPU, without downloading data.
python -m dropout_mft.experiments.benchmark benchmark smoke
```

On Windows, activate with `.venv\Scripts\activate`. The package pins the original
scientific dependencies; GPU runs need a PyTorch build compatible with the cluster.
For dataset preparation in the multi-dataset benchmark, also install:

```bash
python -m pip install -e '.[benchmarks]'
```

## Choose an experiment

| Experiment | What it does | Start here |
|---|---|---|
| [Paper](experiments/paper/) | Rebuild figures from saved results; no training needed | `python experiments/paper/make_figures.py --all` |
| [Scale transfer](experiments/scale_transfer/) | Compare standard and μP parameterizations across widths | `python experiments/scale_transfer/run.py smoke` |
| [Benchmark](experiments/benchmark/) | Compare dropout profiles across five datasets and two architectures | `python -m dropout_mft.experiments.benchmark --help` |
| [Legacy](experiments/legacy/) | Reproduce the historical CIFAR-10 MLP comparison | `python experiments/legacy/run.py smoke` |
| [Optiver](experiments/optiver/) | Run volatility-prediction studies with audited data splits | `python experiments/optiver/run_canary.py --help` |
| [RNN studies](docs/rnn-studies.md) | Synthetic recall, Penn Treebank, and a linear-profile follow-up | Read the guide for setup and individual commands |

Each experiment README describes its scientific protocol, data requirements,
outputs, and Slurm launch commands. **A smoke check verifies the implementation;
it is not a scientific result.**

## Run a benchmark study

All benchmark studies use the same command structure:

```text
python -m dropout_mft.experiments.benchmark STUDY COMMAND [OPTIONS]
```

| Study | Comparison |
|---|---|
| `benchmark` | Original five-dataset, depth-six suite |
| `zero_decay` | Depth-twelve dropout sweep with zero weight decay |
| `vision_zero_decay` | Tiny ImageNet preset of the zero-decay study |
| `data_regimes` | Data-size and class-balance comparisons |
| `vision` | Tiny ImageNet pilot |
| `sidecar` | Transformer linear-profile and tuned-control follow-up |
| `jannis_100` | 100-epoch Jannis transformer confirmation |

For example, inspect the cost and create a learning-rate search plan:

```bash
python -m dropout_mft.experiments.benchmark zero_decay cost
python -m dropout_mft.experiments.benchmark zero_decay plan \
  --run-dir runs/zero --stage lr_search
```

Once the data is prepared, `run` executes that manifest and `select` chooses
hyperparameters using validation loss. Repeat for the remaining stages in the
[benchmark protocol](experiments/benchmark/README.md). Use the same study name
and run directory throughout. `STUDY COMMAND --help` shows that command's options.

A **manifest** is the fixed list of trials to run. A **profile** is the list of
dropout probabilities across layers. A **cohort** is a group of trials sharing a
scientific protocol. Test data is reserved for the confirmation stage.

## Find your way around

```text
experiments/          Commands, experiment guides, and cluster launchers
src/dropout_mft/      Shared models, training, schedules, results, and analysis
  experiments/       Study definitions and study-specific analysis
tests/              Tests for the package and experiment commands
notebooks/           Published notebook reproductions
results/             Saved paper results
figures/paper/       Paper figures
docs/               Code map, contributor guide, and verification notes
```

Read the [code map](docs/code-map.md) for the path from a command to a model and
saved result. Read [contributing](CONTRIBUTING.md) before changing a protocol or
running the checks. New runs belong in an ignored directory such as `runs/`;
keep the checked-in paper results intact.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
make check
```

This runs lint, formatting, shell syntax checks, the package tests, and all three
standalone RNN test suites. Tests use small CPU cases and synthetic data; they do
not submit cluster jobs or require W&B. See [verification notes](docs/refactor-verification.md)
for the deterministic comparisons used to validate the consolidation.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the paper. Code is distributed under the
[MIT license](LICENSE).
