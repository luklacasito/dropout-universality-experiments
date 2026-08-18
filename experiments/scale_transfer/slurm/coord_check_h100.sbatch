#!/usr/bin/env bash
# Strict multi-seed muP coordinate check on one Bridges-2 H100.

#SBATCH --job-name=dropout-coord
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the transferred source checkout}"
: "${RUN_DIR:?Set RUN_DIR to a shared Ocean results directory}"
: "${VENV_DIR:?Set VENV_DIR to the Python 3.11 virtual environment}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

task_tmp="${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER}/dropout-${SLURM_JOB_ID}}}"
mkdir -p "${task_tmp}/matplotlib"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun python experiments/scale_transfer/run.py coord-check \
  --run-dir "${RUN_DIR}" \
  --widths 64 128 256 512 1024 2048 \
  --steps 2 \
  --seeds 0 1 2 \
  --learning-rate 3e-3 \
  --device cuda \
  --strict \
  --output "${RUN_DIR}/coord_check.npz"
