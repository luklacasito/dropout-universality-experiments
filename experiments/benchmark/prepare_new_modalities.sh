#!/usr/bin/env bash
# Import only the two new modalities through the shared benchmark preparer.
set -euo pipefail

ROOT="${1:-/Users/mac/Documents/icml/dropout-benchmark-data}"
PYTHON="${PYTHON:-/Users/mac/Documents/icml/.venv-benchmark-suite/bin/python}"

"${PYTHON}" experiments/benchmark/prepare_data.py amazon_reviews --root "${ROOT}" --raw "${ROOT}/raw"
"${PYTHON}" experiments/benchmark/prepare_data.py speech_commands --root "${ROOT}" --raw "${ROOT}/raw"
