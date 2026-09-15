# Consolidation and cleanup verification

The benchmark now uses `python -m dropout_mft.experiments.benchmark STUDY COMMAND`.
Seven old `experiments/benchmark/run*.py` entry points and the redundant
`vision_zero_decay.py` module were removed. Slurm submissions and documentation
use the unified command. Tiny ImageNet zero-decay remains a CLI preset implemented
in `zero_decay.py`; Python callers import its definitions from that module.

The original implementation was captured before refactoring from commit
`b77ae162b0edf5dc4b008927ae0b2bbf32c845c6`. Verification on 2026-09-14 used the
same Python 3.11/PyTorch environment for both implementations.

| Check | Result |
|---|---|
| Complete pytest suite | 385 passed |
| Cohort characterization | All 99 ordered cases / 3,340 trials identical: full specs, IDs, RNG streams, per-layer dropout |
| Real CPU training and resume | All 69 cases identical |
| Aggregate summaries | Benchmark, sidecar, and Jannis outputs identical on controlled trial records |
| CLI cost commands | All seven studies match the original output |
| Shell syntax | All 20 experiment shell/Slurm scripts pass `bash -n` |
| Saved paper artifacts | No changes to tracked results, figures, or notebooks |

The 69 training cases cover all five benchmark modalities with MLP and transformer
architectures, all benchmark schedules on Jannis, sealed-test execution on every
modality/architecture, SP and μP scale-transfer profiles and field budgets, the
ViT bridge, and all legacy profiles. Comparisons hash every scientific result
field, including curves and selected/final test metrics. For benchmark and
scale-transfer cases they also hash model tensors; benchmark confirmation cases
include the saved best-checkpoint payload. Every case verifies completed-result
resume without rewriting its result file.

Timing, runtime/source provenance, checkpoint paths, and serialized checkpoint
container hashes are excluded from cross-implementation equality. Checkpoint
tensors and metadata are compared separately. These exclusions are expected to
vary with execution location and time. Refactored source has a new provenance
identity; existing frozen runs must continue to use their original source.

These checks establish exact agreement for the tested deterministic CPU cases.
They do not claim that full production GPU cohorts were rerun or that different
GPU/software environments produce bitwise-identical results.

## Repeat the differential checks

From this repository, using the same environment for both source trees:

```bash
python tests/regression/training_fingerprints.py /path/to/original /tmp/training-before.json
python tests/regression/training_fingerprints.py "$PWD" /tmp/training-after.json
diff -u /tmp/training-before.json /tmp/training-after.json
python tests/regression/analysis_fingerprints.py /path/to/original /tmp/analysis-before.json
python tests/regression/analysis_fingerprints.py "$PWD" /tmp/analysis-after.json
diff -u /tmp/analysis-before.json /tmp/analysis-after.json
```

The workspace's `../archive/pre-refactor-implementation.tar.gz` preserves the
original tree. `../archive/verification.json` records the matching training and
analysis fingerprints. The in-repo cohort fixture remains part of normal pytest.

## Repository cleanup verification (2026-09-15)

The human-facing cleanup adds a quick start, experiment index, code map,
contributor guide, portable modality-preparation launcher, Ruff configuration,
and `make check` with an optional GitHub Actions template. Shared helpers now use clearer names and
types; the runner retains one dataset bundle at a time, and failed JSON writes
remove temporary files while preserving the previous selection.

Checks run locally with Python 3.11 and the existing pinned PyTorch environment:

| Check | Result |
|---|---|
| Package tests | 386 passed |
| Standalone RNN tests | 16 passed across three separate processes |
| Ruff lint and formatting | Passed for all 91 maintained Python files |
| Shell and Slurm syntax | Passed |
| Training and resume fingerprints | All 69 cases identical to both the starting working tree and commit `b77ae16` |
| Aggregate fingerprints | All three summaries identical to both the starting working tree and commit `b77ae16` |
| Frozen RNN bundles | Every entry in all three `SHA256SUMS` files matches |
| Saved paper inputs | No changes to tracked results, figures, or notebooks |

`make check` is the routine verification entry point. The differential scripts
above provide the deeper training/analysis checks. To reconstruct the original
committed source without local workspace archives:

```bash
mkdir -p /tmp/dropout-original
git archive b77ae162b0edf5dc4b008927ae0b2bbf32c845c6 | tar -x -C /tmp/dropout-original
```

The historical archives mentioned above are local recovery records, not required
files in a fresh GitHub checkout. At that point the RNN bundles were excluded from formatting because their
source bytes identified existing runs. They are now archived outside the repository.


## V2 scope and paper verification (2026-09-15)

The maintained tree now contains original-paper reproduction, scale transfer,
and multi-dataset benchmarks. RNN, legacy-rerun, and Optiver code, tests, and
visualizations were moved to a local archive outside Git. All 72 moved files
were verified by SHA-256; a complete source snapshot was also retained. These
files remain accessible in historical commits; no history was rewritten.

| Check | Result |
|---|---|
| Maintained pytest suite | 366 passed |
| Ruff lint/format and shell syntax | Passed |
| Retained training/resume fingerprints | All 63 identical to the pre-v2 tree |
| Benchmark export | 618 completed confirmation runs; every epoch recovered |
| Benchmark comparisons | 21 cohorts; paired seeds and split identity checked |
| New appendix figures | 21 benchmark figures and one 30-run geometry pilot |
| Original result table | Six final-epoch comparisons recomputed from saved arrays |

New evidence checks reject duplicate seeds and changed data splits, exclude
incomplete candidate arms from winner selection, preserve negative results,
and never substitute a baseline from another cohort. The archived experiments
are no longer part of `make check` or the training fingerprint script.

The figure/table build is offline. Its full manuscript integration was compiled
with LaTeX and visually checked. Two linear follow-ups lack recovered uniform
baselines, two Amazon 20k MLP candidate arms have only four seeds, and a complete
width-transfer grid is unavailable. These limits are stated in the results index
and manuscript rather than inferred away.
