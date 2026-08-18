#!/usr/bin/env bash
# One independent Optiver winning-feature MLP trial on one Bridges-2 H100.

#SBATCH --job-name=dropout-optiver
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${OPTIVER_CACHE:?Set OPTIVER_CACHE}"
: "${OPTIVER_STAGE:?Set OPTIVER_STAGE to selection, confirmation, or epoch_probe}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
export WANDB_DIR="${RUN_DIR}/wandb"

task_tmp="${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER}/dropout-optiver-${SLURM_JOB_ID}}}"
mkdir -p "${task_tmp}/matplotlib" "${task_tmp}/torch"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${task_tmp}/torch"

manifest="${RUN_DIR}/manifests/${OPTIVER_STAGE}.jsonl"
output_dir="${RUN_DIR}/${OPTIVER_STAGE}"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} stage=${OPTIVER_STAGE}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun python experiments/optiver/run_winning_mlp.py run \
  --manifest "${manifest}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --cache "${OPTIVER_CACHE}" \
  --output-dir "${output_dir}" \
  --checkpoint-dir "${RUN_DIR}/checkpoints" \
  --device cuda
