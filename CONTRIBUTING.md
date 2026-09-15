# Working on this repository

## Setup and checks

Use Python 3.11, activate a virtual environment, then install:

```bash
python -m pip install -e . -r requirements-dev.txt
make check
```

`make check` runs Ruff, checks shell/Slurm syntax, and runs the package tests. Benchmark dataset extras are only needed when preparing real datasets.

Useful individual commands:

```bash
make format                         # Sort imports and format maintained Python code
make lint                           # Check without editing
make test                           # Package and experiment tests
python -m pytest tests/benchmark -q  # Just the benchmark package tests
make check-shell
```

Override the interpreter with `make check PYTHON=/path/to/python` if needed.
Without Make, run `python -m ruff check .`, `python -m ruff format --check .`,
and `python -m pytest`.

Ruff formats maintained `.py` files and leaves notebook outputs alone. The
optional pre-commit hook strips notebook output; enable it with `pre-commit install`.

## Optional GitHub Actions checks

[docs/github-actions.yml](docs/github-actions.yml) is a ready-to-use workflow
that installs CPU PyTorch and runs `make check`. To activate it, copy it to
`.github/workflows/checks.yml` using an account or token with workflow-writing
permission. It is kept as a template because the publishing connections do not
have that permission. Local checks work without GitHub Actions.

## Keep changes easy to read

- Use descriptive functions, straightforward control flow, and plain data structures.
- Put study-specific choices beside the study definition. Extract a shared helper
  when multiple callers actually need the same behavior.
- Comment on scientific assumptions and non-obvious constraints. Avoid comments
  that repeat the code.
- Keep command wrappers small; use the editable package for shared imports.
- Add tests for changed behavior or a reproduced bug. Tiny CPU cases should be
  enough for most correctness checks.

See the [code map](docs/code-map.md) to find the right module.

## Preserve the experiment being measured

Changes to splits, random streams, initialization, profile sampling, learning-rate
schedules, or checkpoint selection can change the comparison. Give a changed
scientific protocol a new cohort identity and document the difference.

Keep validation-only tuning, test-set separation, paired seeds, immutable
manifests, and source checks intact. Keep historical results and notebooks as
reproduction inputs. Write new outputs to `runs/` or an external run directory.

The fixture in `tests/benchmark/fixtures/cohort_fingerprints.json` records the
original ordered trial specifications, IDs, random streams, and layer profiles.
Do not regenerate it just to make an unexpected test failure pass. Investigate
whether the protocol changed first.

For shared training or analysis refactors, compare a before/after checkout using
[the differential verification commands](docs/refactor-verification.md).
Archived experiments are outside the working repository; restore their original
snapshot separately if historical reproduction is needed.

## Pull requests

Explain the concrete behavior changed, the relevant checks, and any effect on
existing runs. Include new setup or invocation steps in the experiment README.
