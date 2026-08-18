#!/usr/bin/env bash
# One independent benchmark-suite shard on one Bridges-2 H100.
#
# Submit one array per stage, after that stage has been planned:
#   sbatch -A PROJECT --array=0-15%8 \
#     --export=ALL,STAGE=lr_search,NUM_SHARDS=16,PROJECT_DIR=...,RUN_DIR=...,\
#VENV_DIR=...,DATA_ROOT=... experiments/benchmark/slurm/run_h100.sbatch
#
# Shards are disjoint and content-addressed, so a requeued or resubmitted task
# re-verifies finished trials and recomputes nothing.

#SBATCH --job-name=dropout-bench
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the transferred source checkout}"
: "${RUN_DIR:?Set RUN_DIR to a shared Ocean results directory}"
: "${VENV_DIR:?Set VENV_DIR to the Python 3.11 virtual environment}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory holding benchmarks/*.npz}"
: "${STAGE:?Set STAGE to lr_search, budget_search, or confirm}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the submitted array size}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT to the W&B project receiving these trials}"

# Spool locally on Ocean by default.  This avoids making expensive training
# depend on outbound connectivity from a compute node; sync from a login node
# after the array finishes.  Set WANDB_MODE=online only after verifying access.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

task_tmp="${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER}/dropout-bench-${SLURM_JOB_ID}}}"
mkdir -p "${task_tmp}/matplotlib" "${task_tmp}/torch"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${task_tmp}/torch"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

args=(
  python experiments/benchmark/run.py run
  --run-dir "${RUN_DIR}"
  --stage "${STAGE}"
  --shard-index "${SLURM_ARRAY_TASK_ID}"
  --num-shards "${NUM_SHARDS}"
  --device cuda
  --data-root "${DATA_ROOT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-run-group "${WANDB_RUN_GROUP}"
)

[[ -z "${WANDB_ENTITY:-}" ]] || args+=(--wandb-entity "${WANDB_ENTITY}")

# Optional filters, useful for timing a single cell before committing the array.
[[ -z "${FILTER_DATASET:-}" ]] || args+=(--dataset "${FILTER_DATASET}")
[[ -z "${FILTER_MODEL_KIND:-}" ]] || args+=(--model-kind "${FILTER_MODEL_KIND}")
[[ -z "${FILTER_PROFILE:-}" ]] || args+=(--profile "${FILTER_PROFILE}")
[[ "${SAVE_BEST_CHECKPOINT:-0}" != "1" ]] || args+=(--save-best-checkpoint)

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} stage=${STAGE}"
echo "wandb_mode=${WANDB_MODE} wandb_project=${WANDB_PROJECT} wandb_group=${WANDB_RUN_GROUP}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} bf16={torch.cuda.is_bf16_supported()}")'
srun "${args[@]}"
