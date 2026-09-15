#!/usr/bin/env bash
# Import only the two new modalities through the shared benchmark preparer.
set -euo pipefail

ROOT="${1:-data}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON}" "${SCRIPT_DIR}/prepare_data.py" amazon_reviews --root "${ROOT}" --raw "${ROOT}/raw"
"${PYTHON}" "${SCRIPT_DIR}/prepare_data.py" speech_commands --root "${ROOT}" --raw "${ROOT}/raw"
