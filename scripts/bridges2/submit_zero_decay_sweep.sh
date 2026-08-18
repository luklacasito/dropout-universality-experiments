#!/usr/bin/env bash
# Shared depth-12 benchmark sweep with weight decay fixed to exactly zero.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new zero-decay cohort directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR_SHARDS="${LR_SHARDS:-12}"
BUDGET_SHARDS="${BUDGET_SHARDS:-20}"
CONFIRM_SHARDS="${CONFIRM_SHARDS:-12}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"
DATASETS="${DATASETS:-fi2010 openml_jannis}"
COHORT_ID="${COHORT_ID:-zero-weight-decay-depth12-dropout-sweep-v1}"
RUN_CANARIES="${RUN_CANARIES:-0}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
export WANDB_PROJECT
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${RUN_DIR}/manifests/lr_search.jsonl" ]]; then
  echo "Refusing to reuse zero-decay run directory: ${RUN_DIR}" >&2
  exit 2
fi
read -r -a dataset_names <<<"${DATASETS}"
planner_dataset_args=()
for dataset in "${dataset_names[@]}"; do
  if [[ ! -f "${DATA_ROOT}/benchmarks/${dataset}.npz" ]]; then
    echo "Missing cache: ${DATA_ROOT}/benchmarks/${dataset}.npz" >&2
    exit 2
  fi
  planner_dataset_args+=(--dataset "${dataset}")
done

mkdir -p "${RUN_DIR}/logs"
python scripts/run_zero_decay_sweep.py plan \
  --run-dir "${RUN_DIR}" --stage lr_search --depth 12 \
  --cohort-id "${COHORT_ID}" "${planner_dataset_args[@]}"

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},DATA_ROOT=${DATA_ROOT}"

submit_array() {
  local stage="$1" shards="$2" dependency="$3" save_checkpoint="$4"
  local dep_args=()
  [[ -z "${dependency}" ]] || dep_args=(--dependency="afterok:${dependency}")
  sbatch --parsable "${SBATCH_ARGS[@]}" "${dep_args[@]}" \
    --array="0-$((shards - 1))%${MAX_CONCURRENT}" \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},STAGE=${stage},NUM_SHARDS=${shards},SAVE_BEST_CHECKPOINT=${save_checkpoint}" \
    "${HERE}/run_benchmark_h100.sbatch"
}

submit_select_and_plan() {
  local stage="$1" next_stage="$2" dependency="$3" job_name="$4"
  local wrap
  wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python scripts/run_benchmark_suite.py select \
--run-dir ${RUN_DIR} --stage ${stage} && \
PYTHONPATH=${PROJECT_DIR}/src python scripts/run_zero_decay_sweep.py plan \
--run-dir ${RUN_DIR} --stage ${next_stage} --depth 12 \
--cohort-id ${COHORT_ID} $(printf -- '--dataset %q ' "${dataset_names[@]}")"
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --dependency="afterok:${dependency}" \
    --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
    --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
    --job-name="${job_name}" \
    --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
    --wrap="${wrap}"
}

canary_jobs=()
if [[ "${RUN_CANARIES}" == "1" ]]; then
  for dataset in "${dataset_names[@]}"; do
    for model_kind in mlp transformer; do
      canary_jobs+=("$(sbatch --parsable "${SBATCH_ARGS[@]}" \
        --array=0-0 \
        --job-name="dropout-canary-${dataset}-${model_kind}" \
        --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
        --export="ALL,${common},STAGE=lr_search,NUM_SHARDS=5,FILTER_DATASET=${dataset},FILTER_MODEL_KIND=${model_kind},FILTER_PROFILE=uniform,SAVE_BEST_CHECKPOINT=0" \
        "${HERE}/run_benchmark_h100.sbatch")")
    done
  done
  echo "zero-decay canaries       ${canary_jobs[*]}"
fi

canary_dependency=""
if (( ${#canary_jobs[@]} )); then
  canary_dependency="$(IFS=:; echo "${canary_jobs[*]}")"
fi

lr_job=$(submit_array lr_search "${LR_SHARDS}" "${canary_dependency}" 0)
echo "zero-decay lr_search      ${lr_job}"

select_lr=$(submit_select_and_plan \
  lr_search budget_search "${lr_job}" dropout-wd0-select-lr)
echo "select + plan budget      ${select_lr}"

budget_job=$(submit_array budget_search "${BUDGET_SHARDS}" "${select_lr}" 0)
echo "zero-decay budget_search  ${budget_job}"

select_budget=$(submit_select_and_plan \
  budget_search confirm "${budget_job}" dropout-wd0-select-p)
echo "select + plan confirm     ${select_budget}"

confirm_job=$(submit_array confirm "${CONFIRM_SHARDS}" "${select_budget}" 1)
echo "zero-decay confirm        ${confirm_job}"

aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python scripts/run_benchmark_suite.py aggregate \
--run-dir ${RUN_DIR}"

aggregate_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${confirm_job}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-wd0-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${aggregate_wrap}")
echo "aggregate                 ${aggregate_job}"

echo
python scripts/run_zero_decay_sweep.py cost "${planner_dataset_args[@]}"
chain_jobs=("${canary_jobs[@]}")
chain_jobs+=("${lr_job}" "${select_lr}" "${budget_job}" "${select_budget}" "${confirm_job}" "${aggregate_job}")
echo "CHAIN_JOBS=$(IFS=,; echo "${chain_jobs[*]}")"
echo "Track: python scripts/run_benchmark_suite.py status --run-dir ${RUN_DIR}"
echo "Checkpoints: ${RUN_DIR}/checkpoints (confirmation only)"
echo "W&B spool: ${RUN_DIR}/wandb"
