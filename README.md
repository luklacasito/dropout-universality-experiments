# Dropout Universality Experiments

Reproduction code and saved results for **Dropout Universality: Scaling Laws
and Optimal Scheduling at the Edge-of-Chaos** ([arXiv:2605.21648](https://arxiv.org/abs/2605.21648)).

The repository is organized by experiment. Each experiment directory contains
its runnable drivers, documentation, and cluster launchers; shared numerical,
model, training, result, and provenance code lives once in `src/dropout_mft`.

## Start here

| Experiment | Purpose | Entry point |
|---|---|---|
| [`paper`](experiments/paper/) | Rebuild the camera-ready figures from checked-in results | `python experiments/paper/make_figures.py --all` |
| [`scale_transfer`](experiments/scale_transfer/) | Run the SP/μP profile and width-transfer study | `python experiments/scale_transfer/run.py smoke` |
| [`benchmark`](experiments/benchmark/) | Run the shared multi-dataset benchmark engine and its registered cohorts | `python experiments/benchmark/run.py smoke` |
| [`legacy`](experiments/legacy/) | Reproduce the exact historical CIFAR-10 MLP comparison | `python experiments/legacy/run.py smoke` |
| [`optiver`](experiments/optiver/) | Run the leakage-audited Optiver studies | `python experiments/optiver/run_canary.py --help` |

Detailed protocols and Bridges-2 commands live in each experiment's README.

## Layout

```text
experiments/                  Drivers, docs, and Slurm launchers by experiment
src/dropout_mft/              Shared schedules, models, training, I/O, and plots
src/dropout_mft/experiments/  Protocol modules and dataset-specific adapters
tests/                        Tests grouped to mirror the experiment structure
notebooks/                    Published notebook reproductions
results/                      Checked-in paper results; never modified by extensions
figures/paper/                Reproduced paper figures
```

The benchmark engine is dataset-agnostic: dataset adapters return the common
`DatasetBundle`, while cohort modules generate immutable `BenchmarkTrialSpec`
rows. Shared planning, selection paths, atomic writes, provenance capture, and
paired bootstrap intervals are implemented once in
`dropout_mft.experiments.benchmark.workflow`.

## Setup

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the optional modality dependencies only for the multi-dataset suite:

```bash
pip install -e '.[benchmarks]'
```

## Verification

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers deterministic manifests and random streams, exact dropout
budgets, locked data splits, validation-only selection, final-only test access,
content-addressed resume behavior, CPU smoke training, and figure generation.
The experiment reorganization intentionally changes paths only; protocol
constants, trial specifications, RNG semantics, training loops, and checked-in
numerical results remain unchanged.
