#!/usr/bin/env bash
# Submit the whole three-stage benchmark suite as one dependency chain.
#
#   export PROJECT_DIR=$PROJECT/dropout-universality-experiments
#   export RUN_DIR=$PROJECT/runs/benchmark-suite-20260811
#   export VENV_DIR=$PROJECT/venvs/dropout311
#   export DATA_ROOT=$PROJECT/data
#   ./experiments/benchmark/slurm/submit_suite.sh -A YOUR_ALLOCATION
#
# Nothing runs until the caches exist, so run prepare_benchmark_data.sbatch (or
# the login-node script) first.  Every stage is resumable: rerunning this after
# a partial failure re-verifies finished trials and only computes what is left.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT to the W&B project receiving the suite}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Concurrency caps, not shard counts: each array task loops over its own shard.
LR_SHARDS="${LR_SHARDS:-16}"
BUDGET_SHARDS="${BUDGET_SHARDS:-24}"
CONFIRM_SHARDS="${CONFIRM_SHARDS:-16}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
CONTROL_PARTITION="${CONTROL_PARTITION:-RM-shared}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-}"

CONTROL_ARGS=(--partition="${CONTROL_PARTITION}" --cpus-per-task=8)
if [[ "${CONTROL_PARTITION}" == GPU* ]]; then
  : "${CONTROL_GPU_TYPE:?Set CONTROL_GPU_TYPE (for example v100-16)}"
  CONTROL_ARGS+=(--gpus="${CONTROL_GPU_TYPE}:1")
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_DIR}/logs"

missing=0
for name in fi2010 tiny_imagenet amazon_reviews speech_commands openml_jannis; do
  if [[ ! -f "${DATA_ROOT}/benchmarks/${name}.npz" ]]; then
    echo "missing cache: ${DATA_ROOT}/benchmarks/${name}.npz" >&2
    missing=1
  fi
done
if (( missing )); then
  echo "Run experiments/benchmark/prepare_data.py first." >&2
  exit 1
fi

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

submit_select() {
  local stage="$1" next_stage="$2" dependency="$3"
  sbatch --parsable "${SBATCH_ARGS[@]}" "${CONTROL_ARGS[@]}" \
    --dependency="afterok:${dependency}" \
    --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
    --export="ALL,${common},STAGE=${stage},NEXT_STAGE=${next_stage}" \
    "${HERE}/select.sbatch"
}

# The first manifest needs no prior selection, so plan it here on the login node.
"${VENV_DIR}/bin/python" -m dropout_mft.experiments.benchmark benchmark plan \
  --run-dir "${RUN_DIR}" --stage lr_search

lr_job=$(submit_array lr_search "${LR_SHARDS}" "")
echo "lr_search array       ${lr_job}"

select_lr=$(submit_select lr_search budget_search "${lr_job}")
echo "select + plan budget  ${select_lr}"

budget_job=$(submit_array budget_search "${BUDGET_SHARDS}" "${select_lr}")
echo "budget_search array   ${budget_job}"

select_budget=$(submit_select budget_search confirm "${budget_job}")
echo "select + plan confirm ${select_budget}"

confirm_job=$(submit_array confirm "${CONFIRM_SHARDS}" "${select_budget}")
echo "confirm array         ${confirm_job}"

aggregate_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  "${CONTROL_ARGS[@]}" \
  --dependency="afterok:${confirm_job}" \
  --job-name=dropout-bench-aggregate \
  --nodes=1 --ntasks=1 \
  --time=00:30:00 --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python -m dropout_mft.experiments.benchmark benchmark aggregate \
--run-dir ${RUN_DIR}")
echo "aggregate             ${aggregate_job}"

echo
echo "Track with: squeue -u \$USER"
echo "Progress:   python -m dropout_mft.experiments.benchmark benchmark status --run-dir ${RUN_DIR}"
echo "W&B spool:  ${RUN_DIR}/wandb (sync after the arrays finish)"
