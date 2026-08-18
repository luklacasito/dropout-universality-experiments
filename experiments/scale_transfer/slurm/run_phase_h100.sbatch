#!/usr/bin/env bash
# One independent scale-transfer shard on one Bridges-2 H100.
# Supply the allocation and array bounds at submission time, for example:
#   sbatch -A PROJECT --array=0-39%4 --export=ALL,PHASE=width_transfer,NUM_SHARDS=40,... this_file

#SBATCH --job-name=dropout-profile
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
: "${DATA_ROOT:?Set DATA_ROOT to the shared CIFAR data directory}"
: "${PHASE:?Set PHASE to the locked experiment phase}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the submitted array size}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

task_tmp="${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER}/dropout-${SLURM_JOB_ID}}}"
mkdir -p "${task_tmp}/matplotlib" "${task_tmp}/torch"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${task_tmp}/torch"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

args=(
  python experiments/scale_transfer/run.py run
  --run-dir "${RUN_DIR}"
  --phase "${PHASE}"
  --shard-index "${SLURM_ARRAY_TASK_ID}"
  --num-shards "${NUM_SHARDS}"
  --device cuda
  --data-root "${DATA_ROOT}"
)

# These filters select a real immutable-manifest trial for timing.  The full
# array later resumes it rather than creating a separate benchmark treatment.
[[ -z "${FILTER_WIDTH:-}" ]] || args+=(--width "${FILTER_WIDTH}")
[[ -z "${FILTER_PARAMETERIZATION:-}" ]] || \
  args+=(--parameterization "${FILTER_PARAMETERIZATION}")
[[ -z "${FILTER_PROFILE:-}" ]] || args+=(--profile "${FILTER_PROFILE}")
[[ -z "${FILTER_SEED:-}" ]] || args+=(--seed "${FILTER_SEED}")

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} phase=${PHASE}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} bf16={torch.cuda.is_bf16_supported()}")'
srun "${args[@]}"
