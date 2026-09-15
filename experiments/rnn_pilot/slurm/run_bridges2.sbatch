#!/usr/bin/env bash
# One paired seed (all six arms) per array element.
#SBATCH --job-name=rnn-dropout-pilot
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the transferred source checkout}"
: "${RUN_DIR:?Set RUN_DIR to the Ocean results directory}"
: "${VENV_DIR:?Set VENV_DIR to the existing PyTorch virtual environment}"
: "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-2%1}"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
args=(python experiments/rnn_pilot/run.py --out "${RUN_DIR}"
      --seed "${SLURM_ARRAY_TASK_ID}" --device cuda --threads "${OMP_NUM_THREADS}")
if [[ "${CANARY:-0}" == "1" ]]; then
    args+=(--arms none --epochs 2)
fi
srun "${args[@]}"
