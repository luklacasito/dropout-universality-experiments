#!/usr/bin/env bash
# Canary-gated Tiny ImageNet zero-decay sweep.  Every GPU task runs one trial.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new Tiny ImageNet zero-decay directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LR_SHARDS="${LR_SHARDS:-30}"
BUDGET_SHARDS="${BUDGET_SHARDS:-60}"
CONFIRM_SHARDS="${CONFIRM_SHARDS:-30}"
CONCURRENT_PER_MODEL="${CONCURRENT_PER_MODEL:-2}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
export WANDB_PROJECT
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${RUN_DIR}/manifests/lr_search.jsonl" ]]; then
  echo "Refusing to reuse Tiny ImageNet zero-decay run directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DATA_ROOT}/benchmarks/tiny_imagenet.npz" ]]; then
  echo "Missing cache: ${DATA_ROOT}/benchmarks/tiny_imagenet.npz" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs"
python scripts/run_tiny_imagenet_zero_decay.py plan \
  --run-dir "${RUN_DIR}" --stage lr_search --depth 12

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},DATA_ROOT=${DATA_ROOT}"

submit_canary() {
  local model_kind="$1"
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --array=0-0 \
    --job-name="dropout-vision-canary-${model_kind}" \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},STAGE=lr_search,NUM_SHARDS=5,FILTER_DATASET=tiny_imagenet,FILTER_MODEL_KIND=${model_kind},FILTER_PROFILE=uniform,SAVE_BEST_CHECKPOINT=0" \
    "${HERE}/run_benchmark_h100.sbatch"
}

submit_model_array() {
  local stage="$1" model_kind="$2" shards="$3" dependency="$4" save_checkpoint="$5"
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --dependency="afterok:${dependency}" \
    --array="0-$((shards - 1))%${CONCURRENT_PER_MODEL}" \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},STAGE=${stage},NUM_SHARDS=${shards},FILTER_DATASET=tiny_imagenet,FILTER_MODEL_KIND=${model_kind},SAVE_BEST_CHECKPOINT=${save_checkpoint}" \
    "${HERE}/run_benchmark_h100.sbatch"
}

submit_select_and_plan() {
  local stage="$1" next_stage="$2" dependency="$3" job_name="$4"
  local wrap
  wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python scripts/run_benchmark_suite.py select \
--run-dir ${RUN_DIR} --stage ${stage} && \
PYTHONPATH=${PROJECT_DIR}/src python scripts/run_tiny_imagenet_zero_decay.py plan \
--run-dir ${RUN_DIR} --stage ${next_stage} --depth 12"
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --dependency="afterok:${dependency}" \
    --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
    --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
    --job-name="${job_name}" \
    --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
    --wrap="${wrap}"
}

canary_mlp=$(submit_canary mlp)
canary_vit=$(submit_canary transformer)
echo "Tiny ImageNet MLP canary  ${canary_mlp}"
echo "Tiny ImageNet ViT canary  ${canary_vit}"
canary_dependency="${canary_mlp}:${canary_vit}"

lr_mlp=$(submit_model_array lr_search mlp "${LR_SHARDS}" "${canary_dependency}" 0)
lr_vit=$(submit_model_array lr_search transformer "${LR_SHARDS}" "${canary_dependency}" 0)
echo "vision wd0 lr MLP         ${lr_mlp}"
echo "vision wd0 lr ViT         ${lr_vit}"

select_lr=$(submit_select_and_plan \
  lr_search budget_search "${lr_mlp}:${lr_vit}" dropout-vision-wd0-select-lr)
echo "select + plan budget      ${select_lr}"

budget_mlp=$(submit_model_array budget_search mlp "${BUDGET_SHARDS}" "${select_lr}" 0)
budget_vit=$(submit_model_array budget_search transformer "${BUDGET_SHARDS}" "${select_lr}" 0)
echo "vision wd0 budget MLP     ${budget_mlp}"
echo "vision wd0 budget ViT     ${budget_vit}"

select_budget=$(submit_select_and_plan \
  budget_search confirm "${budget_mlp}:${budget_vit}" dropout-vision-wd0-select-p)
echo "select + plan confirm     ${select_budget}"

confirm_mlp=$(submit_model_array confirm mlp "${CONFIRM_SHARDS}" "${select_budget}" 1)
confirm_vit=$(submit_model_array confirm transformer "${CONFIRM_SHARDS}" "${select_budget}" 1)
echo "vision wd0 confirm MLP    ${confirm_mlp}"
echo "vision wd0 confirm ViT    ${confirm_vit}"

aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python scripts/run_benchmark_suite.py aggregate \
--run-dir ${RUN_DIR}"
aggregate_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${confirm_mlp}:${confirm_vit}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-vision-wd0-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${aggregate_wrap}")
echo "aggregate                 ${aggregate_job}"

echo
python scripts/run_tiny_imagenet_zero_decay.py cost
echo "At most $((2 * CONCURRENT_PER_MODEL)) H100s run concurrently."
echo "Each full array task owns one trial; canary success gates the full chain."
echo "CHAIN_JOBS=${canary_mlp},${canary_vit},${lr_mlp},${lr_vit},${select_lr},${budget_mlp},${budget_vit},${select_budget},${confirm_mlp},${confirm_vit},${aggregate_job}"
echo "Track: python scripts/run_benchmark_suite.py status --run-dir ${RUN_DIR}"
echo "W&B spool: ${RUN_DIR}/wandb"
