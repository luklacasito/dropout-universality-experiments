#!/usr/bin/env bash
# Transformer-only extension: linear-early/late plus independently tuned no-dropout.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new sidecar directory}"
: "${BASELINE_RUN_DIR:?Set BASELINE_RUN_DIR to the corrected depth-12 pilot}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR_SHARDS="${LR_SHARDS:-6}"
CONFIRM_SHARDS="${CONFIRM_SHARDS:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"
BASELINE_DEPENDENCY_JOB="${BASELINE_DEPENDENCY_JOB:-}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${RUN_DIR}/manifests/lr_search.jsonl" ]]; then
  echo "Refusing to reuse sidecar run directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -d "${BASELINE_RUN_DIR}" ]]; then
  echo "Missing corrected baseline run: ${BASELINE_RUN_DIR}" >&2
  exit 2
fi
for dataset in fi2010 openml_jannis; do
  if [[ ! -f "${DATA_ROOT}/benchmarks/${dataset}.npz" ]]; then
    echo "Missing cache: ${DATA_ROOT}/benchmarks/${dataset}.npz" >&2
    exit 2
  fi
done

mkdir -p "${RUN_DIR}/logs"
python experiments/benchmark/run_transformer_sidecar.py plan \
  --run-dir "${RUN_DIR}" --stage lr_search --depth 12

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

lr_job=$(submit_array lr_search "${LR_SHARDS}" "")
echo "sidecar lr_search      ${lr_job}"

select_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run_transformer_sidecar.py select \
--run-dir ${RUN_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run_transformer_sidecar.py plan \
--run-dir ${RUN_DIR} --stage confirm --depth 12"

select_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${lr_job}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-linear-select \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${select_wrap}")
echo "select + plan confirm ${select_job}"

confirm_job=$(submit_array confirm "${CONFIRM_SHARDS}" "${select_job}")
echo "sidecar confirm        ${confirm_job}"

aggregate_dependency="afterok:${confirm_job}"
if [[ -n "${BASELINE_DEPENDENCY_JOB}" ]]; then
  aggregate_dependency="${aggregate_dependency}:${BASELINE_DEPENDENCY_JOB}"
fi
aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run_transformer_sidecar.py aggregate \
--run-dir ${RUN_DIR} --baseline-run-dir ${BASELINE_RUN_DIR}"

aggregate_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="${aggregate_dependency}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-linear-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${aggregate_wrap}")
echo "aggregate              ${aggregate_job}"

echo
python experiments/benchmark/run_transformer_sidecar.py cost
echo "Track with: squeue -u \$USER"
echo "Progress:   python experiments/benchmark/run.py status --run-dir ${RUN_DIR}"
echo "W&B spool:  ${RUN_DIR}/wandb"
