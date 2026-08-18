#!/usr/bin/env bash
# 100-epoch, ten-seed Jannis Transformer confirmation with saved checkpoints.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new follow-up directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_SHARDS="${NUM_SHARDS:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${RUN_DIR}/manifests/confirm.jsonl" ]]; then
  echo "Refusing to reuse follow-up run directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DATA_ROOT}/benchmarks/openml_jannis.npz" ]]; then
  echo "Missing cache: ${DATA_ROOT}/benchmarks/openml_jannis.npz" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs"
python experiments/benchmark/run_jannis_100.py plan --run-dir "${RUN_DIR}"

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},DATA_ROOT=${DATA_ROOT},STAGE=confirm,NUM_SHARDS=${NUM_SHARDS}"
common="${common},SAVE_BEST_CHECKPOINT=1"

confirm_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --array="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT}" \
  --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
  --export="ALL,${common}" \
  "${HERE}/run_h100.sbatch")
echo "jannis 100epoch confirm ${confirm_job}"

aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run_jannis_100.py aggregate \
--run-dir ${RUN_DIR}"

aggregate_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${confirm_job}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-jannis100-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${aggregate_wrap}")
echo "aggregate               ${aggregate_job}"

echo
python experiments/benchmark/run_jannis_100.py cost
echo "Track: python experiments/benchmark/run.py status --run-dir ${RUN_DIR}"
echo "Checkpoints: ${RUN_DIR}/checkpoints/openml_jannis/transformer/confirm"
echo "W&B spool: ${RUN_DIR}/wandb"
