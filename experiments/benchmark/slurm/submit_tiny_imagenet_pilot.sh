#!/usr/bin/env bash
# Full depth-12 Tiny ImageNet all-schedule cohort after the timing canary passes.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new full-cohort directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARDS_PER_MODEL="${SHARDS_PER_MODEL:-6}"
CONCURRENT_PER_MODEL="${CONCURRENT_PER_MODEL:-2}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -e "${RUN_DIR}/manifests/lr_search.jsonl" ]]; then
  echo "Refusing to reuse full vision run directory: ${RUN_DIR}" >&2
  exit 2
fi
test -f "${DATA_ROOT}/benchmarks/tiny_imagenet.npz"
mkdir -p "${RUN_DIR}/logs"

python -m dropout_mft.experiments.benchmark vision plan \
  --run-dir "${RUN_DIR}" --stage lr_search --depth 12

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},DATA_ROOT=${DATA_ROOT}"

submit_model_array() {
  local stage="$1" model_kind="$2" dependency="$3"
  local dep_args=()
  [[ -z "${dependency}" ]] || dep_args=(--dependency="afterok:${dependency}")
  sbatch --parsable "${SBATCH_ARGS[@]}" "${dep_args[@]}" \
    --array="0-$((SHARDS_PER_MODEL - 1))%${CONCURRENT_PER_MODEL}" \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},STAGE=${stage},NUM_SHARDS=${SHARDS_PER_MODEL},FILTER_DATASET=tiny_imagenet,FILTER_MODEL_KIND=${model_kind}" \
    "${HERE}/run_h100.sbatch"
}

lr_mlp=$(submit_model_array lr_search mlp "")
lr_vit=$(submit_model_array lr_search transformer "")
echo "vision lr_search MLP   ${lr_mlp}"
echo "vision lr_search ViT   ${lr_vit}"

select_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python -m dropout_mft.experiments.benchmark benchmark select \
--run-dir ${RUN_DIR} --stage lr_search && \
PYTHONPATH=${PROJECT_DIR}/src python -m dropout_mft.experiments.benchmark vision plan \
--run-dir ${RUN_DIR} --stage confirm --depth 12"

select_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${lr_mlp}:${lr_vit}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-vision-select \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${select_wrap}")
echo "select + plan confirm ${select_job}"

confirm_mlp=$(submit_model_array confirm mlp "${select_job}")
confirm_vit=$(submit_model_array confirm transformer "${select_job}")
echo "vision confirm MLP     ${confirm_mlp}"
echo "vision confirm ViT     ${confirm_vit}"

aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python -m dropout_mft.experiments.benchmark benchmark aggregate \
--run-dir ${RUN_DIR}"

aggregate_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${confirm_mlp}:${confirm_vit}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --time=00:15:00 \
  --job-name=dropout-vision-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" \
  --wrap="${aggregate_wrap}")
echo "aggregate              ${aggregate_job}"

echo
python -m dropout_mft.experiments.benchmark vision cost
echo "At most $((2 * CONCURRENT_PER_MODEL)) H100s run concurrently."
echo "Track with: squeue -u \$USER"
echo "Progress: python -m dropout_mft.experiments.benchmark benchmark status --run-dir ${RUN_DIR}"
