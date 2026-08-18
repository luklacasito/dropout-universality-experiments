#!/usr/bin/env bash
# Fixed-budget depth-12 pilot on FI-2010 and Jannis, for MLP and transformer.
# Sixty LR-search trials feed eighty fresh-seed confirmation trials.  This is a
# separate run directory and never mixes with the depth-6 benchmark manifests.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a dedicated depth-12 pilot directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT to the W&B project receiving the pilot}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR_SHARDS="${LR_SHARDS:-12}"
CONFIRM_SHARDS="${CONFIRM_SHARDS:-16}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_DIR}/logs"

for name in fi2010 openml_jannis; do
  if [[ ! -f "${DATA_ROOT}/benchmarks/${name}.npz" ]]; then
    echo "missing cache: ${DATA_ROOT}/benchmarks/${name}.npz" >&2
    exit 1
  fi
done

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},DATA_ROOT=${DATA_ROOT}"

submit_array() {
  local stage="$1" shards="$2" dependency="$3"
  local dep_args=()
  [[ -z "${dependency}" ]] || dep_args=(--dependency="afterok:${dependency}")
  sbatch --parsable "${SBATCH_ARGS[@]}" "${dep_args[@]}" \
    --array="0-$((shards - 1))%${MAX_CONCURRENT}" \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},STAGE=${stage},NUM_SHARDS=${shards}" \
    "${HERE}/run_h100.sbatch"
}

"${VENV_DIR}/bin/python" experiments/benchmark/run.py plan \
  --run-dir "${RUN_DIR}" --stage lr_search --depth 12 \
  --dataset fi2010 --dataset openml_jannis

lr_job=$(submit_array lr_search "${LR_SHARDS}" "")
echo "depth12 lr_search      ${lr_job}"

select_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run.py select \
--run-dir ${RUN_DIR} --stage lr_search && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run.py plan \
--run-dir ${RUN_DIR} --stage confirm --depth 12 --selection-stage lr_search \
--dataset fi2010 --dataset openml_jannis"

select_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${lr_job}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-depth12-select \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${select_wrap}")
echo "select + plan confirm ${select_job}"

confirm_job=$(submit_array confirm "${CONFIRM_SHARDS}" "${select_job}")
echo "depth12 confirm        ${confirm_job}"

aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run.py aggregate \
--run-dir ${RUN_DIR}"

aggregate_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${confirm_job}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-depth12-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${aggregate_wrap}")
echo "aggregate              ${aggregate_job}"

echo
echo "Track with: squeue -u \$USER"
echo "Progress:   python experiments/benchmark/run.py status --run-dir ${RUN_DIR}"
echo "W&B spool:  ${RUN_DIR}/wandb (sync after the arrays finish)"
