#!/usr/bin/env bash
# Canary-gated zero-decay sweep for one prespecified sample-size regime.
# Every full-array task owns exactly one trial after model filtering.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new data-regime cohort directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${REGIME:?Set REGIME to amazon_n2000, amazon_n5000, amazon_n20000, or tiny_n80000}"

case "${REGIME}" in
  amazon_n2000 | amazon_n5000 | amazon_n20000)
    DATASET="amazon_reviews"
    default_mlp_time="00:30:00"
    default_transformer_time="00:30:00"
    ;;
  tiny_n80000)
    DATASET="tiny_imagenet"
    default_mlp_time="00:30:00"
    default_transformer_time="00:45:00"
    ;;
  *)
    echo "Unknown REGIME=${REGIME}" >&2
    exit 2
    ;;
esac

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR_SHARDS="${LR_SHARDS:-30}"
BUDGET_SHARDS="${BUDGET_SHARDS:-60}"
CONFIRM_SHARDS="${CONFIRM_SHARDS:-30}"
CONCURRENT_PER_MODEL="${CONCURRENT_PER_MODEL:-1}"
MLP_TIME="${MLP_TIME:-${default_mlp_time}}"
TRANSFORMER_TIME="${TRANSFORMER_TIME:-${default_transformer_time}}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"
CHAIN_DEPENDENCY="${CHAIN_DEPENDENCY:-}"

if ! [[ "${CONCURRENT_PER_MODEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONCURRENT_PER_MODEL must be a positive integer" >&2
  exit 2
fi
for shard_count in "${LR_SHARDS}" "${BUDGET_SHARDS}" "${CONFIRM_SHARDS}"; do
  if ! [[ "${shard_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Shard counts must be positive integers" >&2
    exit 2
  fi
done

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
export WANDB_PROJECT
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${RUN_DIR}/manifests/lr_search.jsonl" ]]; then
  echo "Refusing to reuse data-regime run directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DATA_ROOT}/benchmarks/${DATASET}.npz" ]]; then
  echo "Missing cache: ${DATA_ROOT}/benchmarks/${DATASET}.npz" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs"
python experiments/benchmark/run_data_regime.py plan \
  --run-dir "${RUN_DIR}" --regime "${REGIME}" --stage lr_search

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},DATA_ROOT=${DATA_ROOT}"

# `sbatch --parsable` can append `;cluster`; dependencies require the bare ID.
parse_job_id() {
  local submitted="$1"
  submitted="${submitted%%;*}"
  if ! [[ "${submitted}" =~ ^[0-9]+([_][0-9]+)?$ ]]; then
    echo "Could not parse sbatch job ID from: $1" >&2
    return 2
  fi
  printf '%s\n' "${submitted}"
}

model_time() {
  case "$1" in
    mlp) printf '%s\n' "${MLP_TIME}" ;;
    transformer) printf '%s\n' "${TRANSFORMER_TIME}" ;;
    *) return 2 ;;
  esac
}

submit_canary() {
  local model_kind="$1" raw
  local dependency_args=()
  [[ -z "${CHAIN_DEPENDENCY}" ]] || dependency_args=(
    --dependency="afterok:${CHAIN_DEPENDENCY}"
  )
  raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
    "${dependency_args[@]}" \
    --partition=GPU-shared --gpus=h100-80:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G \
    --time="$(model_time "${model_kind}")" \
    --array=0-0 \
    --job-name="dropout-regime-canary-${model_kind}" \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},STAGE=lr_search,NUM_SHARDS=5,FILTER_DATASET=${DATASET},FILTER_MODEL_KIND=${model_kind},FILTER_PROFILE=uniform,SAVE_BEST_CHECKPOINT=0" \
    "${HERE}/run_h100.sbatch")
  parse_job_id "${raw}"
}

submit_model_array() {
  local stage="$1" model_kind="$2" shards="$3" dependency="$4"
  local save_checkpoint="$5" raw
  raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
    --partition=GPU-shared --gpus=h100-80:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G \
    --time="$(model_time "${model_kind}")" \
    --dependency="afterok:${dependency}" \
    --array="0-$((shards - 1))%${CONCURRENT_PER_MODEL}" \
    --job-name="dropout-regime-${stage}-${model_kind}" \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},STAGE=${stage},NUM_SHARDS=${shards},FILTER_DATASET=${DATASET},FILTER_MODEL_KIND=${model_kind},SAVE_BEST_CHECKPOINT=${save_checkpoint}" \
    "${HERE}/run_h100.sbatch")
  parse_job_id "${raw}"
}

submit_select_and_plan() {
  local stage="$1" next_stage="$2" dependency="$3" job_name="$4"
  local wrap raw
  wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run.py select \
--run-dir ${RUN_DIR} --stage ${stage} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run_data_regime.py plan \
--run-dir ${RUN_DIR} --regime ${REGIME} --stage ${next_stage}"
  raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
    --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
    --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G --time=00:15:00 \
    --dependency="afterok:${dependency}" \
    --job-name="${job_name}" \
    --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
    --wrap="${wrap}")
  parse_job_id "${raw}"
}

canary_mlp=$(submit_canary mlp)
canary_transformer=$(submit_canary transformer)
echo "${REGIME} MLP canary          ${canary_mlp}"
echo "${REGIME} Transformer canary  ${canary_transformer}"

lr_mlp=$(submit_model_array lr_search mlp "${LR_SHARDS}" "${canary_mlp}" 0)
lr_transformer=$(submit_model_array \
  lr_search transformer "${LR_SHARDS}" "${canary_transformer}" 0)
echo "${REGIME} LR MLP              ${lr_mlp}"
echo "${REGIME} LR Transformer      ${lr_transformer}"

select_lr=$(submit_select_and_plan \
  lr_search budget_search "${lr_mlp}:${lr_transformer}" \
  dropout-regime-select-lr)
echo "select + plan budget          ${select_lr}"

budget_mlp=$(submit_model_array \
  budget_search mlp "${BUDGET_SHARDS}" "${select_lr}" 0)
budget_transformer=$(submit_model_array \
  budget_search transformer "${BUDGET_SHARDS}" "${select_lr}" 0)
echo "${REGIME} budget MLP          ${budget_mlp}"
echo "${REGIME} budget Transformer  ${budget_transformer}"

select_budget=$(submit_select_and_plan \
  budget_search confirm "${budget_mlp}:${budget_transformer}" \
  dropout-regime-select-p)
echo "select + plan confirm         ${select_budget}"

confirm_mlp=$(submit_model_array \
  confirm mlp "${CONFIRM_SHARDS}" "${select_budget}" 1)
confirm_transformer=$(submit_model_array \
  confirm transformer "${CONFIRM_SHARDS}" "${select_budget}" 1)
echo "${REGIME} confirm MLP         ${confirm_mlp}"
echo "${REGIME} confirm Transformer ${confirm_transformer}"

aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/benchmark/run.py aggregate \
--run-dir ${RUN_DIR}"
aggregate_raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G --time=00:15:00 \
  --dependency="afterok:${confirm_mlp}:${confirm_transformer}" \
  --job-name=dropout-regime-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${aggregate_wrap}")
aggregate_job=$(parse_job_id "${aggregate_raw}")
echo "aggregate                     ${aggregate_job}"

echo
python experiments/benchmark/run_data_regime.py cost --regime "${REGIME}"
chain_jobs=(
  "${canary_mlp}" "${canary_transformer}"
  "${lr_mlp}" "${lr_transformer}" "${select_lr}"
  "${budget_mlp}" "${budget_transformer}" "${select_budget}"
  "${confirm_mlp}" "${confirm_transformer}" "${aggregate_job}"
)
chain_jobs_csv="$(IFS=,; echo "${chain_jobs[*]}")"
printf '%s\n' "${chain_jobs_csv}" >"${RUN_DIR}/slurm-chain-jobs.txt"
printf '%s\n' "${aggregate_job}" >"${RUN_DIR}/slurm-aggregate-job.txt"
echo "At most $((2 * CONCURRENT_PER_MODEL)) H100s run concurrently."
echo "Canary success gates each model's one-trial-per-task arrays."
echo "CHAIN_JOBS=${chain_jobs_csv}"
echo "Track: python experiments/benchmark/run.py status --run-dir ${RUN_DIR}"
echo "Checkpoints: ${RUN_DIR}/checkpoints (confirmation only)"
echo "W&B spool: ${RUN_DIR}/wandb"
