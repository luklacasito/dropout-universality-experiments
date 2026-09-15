#!/usr/bin/env bash
#SBATCH --job-name=rnn-ptb
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
: "${PROJECT_DIR:?Set frozen source root}"
: "${RUN_DIR:?Set output directory}"
: "${VENV_DIR:?Set Python environment}"
: "${DATA_ROOT:?Set canonical PTB data directory}"
: "${STAGE:?Set canary, calibrate, or confirm}"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
sha256sum -c experiments/rnn_ptb/SHA256SUMS
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
args=(python experiments/rnn_ptb/run.py --stage "$STAGE" --data-root "$DATA_ROOT"
      --out "$RUN_DIR" --device cuda --threads "$OMP_NUM_THREADS"
      --epochs "${EPOCHS:-20}")
if [[ "$STAGE" == "canary" ]]; then args+=(--epochs 1); fi
if [[ "$STAGE" == "calibrate" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    args+=(--trial-index "$SLURM_ARRAY_TASK_ID")
fi
if [[ "$STAGE" == "confirm" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?Confirmation requires --array=0-17%1}"
    args+=(--seed "$((SLURM_ARRAY_TASK_ID / 6 + 1))" --trial-index "$((SLURM_ARRAY_TASK_ID % 6))")
fi
srun "${args[@]}"
