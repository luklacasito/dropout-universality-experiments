#!/usr/bin/env bash
# Validation-only selection for one stage, then plan the next stage's manifest.
# CPU only: this reads finished trial files and writes JSON.

#SBATCH --job-name=dropout-bench-select
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the transferred source checkout}"
: "${RUN_DIR:?Set RUN_DIR to a shared Ocean results directory}"
: "${VENV_DIR:?Set VENV_DIR to the Python 3.11 virtual environment}"
: "${STAGE:?Set STAGE to the stage whose results are being selected}"
: "${NEXT_STAGE:?Set NEXT_STAGE to the stage to plan next}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

python scripts/run_benchmark_suite.py status --run-dir "${RUN_DIR}"
python scripts/run_benchmark_suite.py select --run-dir "${RUN_DIR}" --stage "${STAGE}"
python scripts/run_benchmark_suite.py plan --run-dir "${RUN_DIR}" --stage "${NEXT_STAGE}"
