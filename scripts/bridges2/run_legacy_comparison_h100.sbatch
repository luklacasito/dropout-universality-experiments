#!/usr/bin/env bash
# Exact-paper MLP comparison: one deterministic shard on one Bridges-2 H100.
# Recommended submission: --array=0-3%4 with NUM_SHARDS=4.

#SBATCH --job-name=dropout-legacy
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the exact frozen source checkout}"
: "${RUN_DIR:?Set RUN_DIR to the separate legacy comparison run directory}"
: "${VENV_DIR:?Set VENV_DIR to the frozen Python environment}"
: "${DATA_ROOT:?Set DATA_ROOT to the shared CIFAR data directory}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the submitted array size}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

task_tmp="${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER}/dropout-legacy-${SLURM_JOB_ID}}}"
mkdir -p "${task_tmp}/matplotlib" "${task_tmp}/torch"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${task_tmp}/torch"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

args=(
  python scripts/run_legacy_comparison.py run
  --run-dir "${RUN_DIR}"
  --data-root "${DATA_ROOT}"
  --device cuda
  --require-h100
  --num-shards "${NUM_SHARDS}"
  --shard-index "${SLURM_ARRAY_TASK_ID}"
)

# Optional canary/debug filters select rows from, but never alter, the frozen
# 125-trial manifest. Completed files are content-checked and resumed.
[[ -z "${FILTER_PROFILE:-}" ]] || args+=(--profile "${FILTER_PROFILE}")
[[ -z "${FILTER_SEED:-}" ]] || args+=(--seed "${FILTER_SEED}")

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} phase=legacy_apples_to_apples"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} bf16={torch.cuda.is_bf16_supported()}")'
python -c 'import pathlib,dropout_mft; expected=(pathlib.Path.cwd()/"src"/"dropout_mft").resolve(); actual=pathlib.Path(dropout_mft.__file__).resolve().parent; assert actual==expected, f"wrong dropout_mft checkout: {actual} != {expected}"; print(f"dropout_mft={actual}")'
srun "${args[@]}"
