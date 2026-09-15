#!/usr/bin/env bash
#SBATCH --job-name=ptb-linear-up
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
: "${PROJECT_DIR:?}"
: "${BASE_RUN_DIR:?}"
: "${RUN_DIR:?}"
: "${DATA_ROOT:?}"
: "${VENV_DIR:?}"
: "${SLURM_ARRAY_TASK_ID:?Use array 1-3%1}"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
sha256sum -c experiments/rnn_ptb_linear/SHA256SUMS
srun python experiments/rnn_ptb_linear/extend.py --base-out "$BASE_RUN_DIR" \
  --out "$RUN_DIR" --data-root "$DATA_ROOT" --seed "$SLURM_ARRAY_TASK_ID"
python experiments/rnn_ptb_linear/compare.py --base-out "$BASE_RUN_DIR" --out "$RUN_DIR" --if-ready
