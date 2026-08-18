#!/usr/bin/env bash
# Build the five benchmark caches once, on CPU, before any GPU work.
# No GPU is needed and none is requested.  Submit as:
#   sbatch -A PROJECT --export=ALL,PROJECT_DIR=...,DATA_ROOT=...,VENV_DIR=... this_file

#SBATCH --job-name=dropout-bench-prep
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the transferred source checkout}"
: "${DATA_ROOT:?Set DATA_ROOT to a shared Ocean data directory}"
: "${VENV_DIR:?Set VENV_DIR to the Python 3.11 virtual environment}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

task_tmp="${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER}/dropout-prep-${SLURM_JOB_ID}}}"
mkdir -p "${task_tmp}/hf" "${task_tmp}/matplotlib"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Tiny-ImageNet and FI-2010 need their archives staged under ${DATA_ROOT}/raw
# beforehand; the script prints the exact commands if they are missing.
srun python experiments/benchmark/prepare_data.py "${TASK:-all}" \
  --root "${DATA_ROOT}" \
  --raw "${DATA_ROOT}/raw"

echo "prepared caches:"
ls -lh "${DATA_ROOT}/benchmarks"
