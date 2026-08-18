#!/usr/bin/env bash
# Upload completed offline benchmark runs from Ocean to W&B.

set -euo pipefail

: "${RUN_DIR:?Set RUN_DIR to the benchmark or pilot results directory}"
: "${VENV_DIR:?Set VENV_DIR to the benchmark Python environment}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT to the destination W&B project}"

source "${VENV_DIR}/bin/activate"

wandb_root="${RUN_DIR}/wandb"
if [[ ! -d "${wandb_root}" ]]; then
  echo "No W&B spool found at ${wandb_root}" >&2
  exit 1
fi

# Only markers written after run.finish() are considered complete.  Reading
# their recorded paths avoids syncing a trial that is still training.
mapfile -t offline_runs < <(
  python -c '
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "tracked"
paths = set()
for marker in root.glob("*.json"):
    record = json.loads(marker.read_text())
    path = record.get("offline_sync_path")
    if record.get("status") == "complete" and path:
        paths.add(path)
print("\n".join(sorted(paths)))
' "${wandb_root}"
)

if (( ${#offline_runs[@]} == 0 )); then
  echo "No completed offline W&B runs found under ${wandb_root}"
  exit 0
fi

echo "Syncing ${#offline_runs[@]} completed runs to ${WANDB_PROJECT}"
wandb sync "${offline_runs[@]}"
