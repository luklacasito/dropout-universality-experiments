#!/usr/bin/env bash
# Two timing canaries: one Tiny ImageNet MLP trial and one ViT trial.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new canary directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_DIR##*/}}"
[[ -z "${WANDB_ENTITY:-}" ]] || export WANDB_ENTITY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -e "${RUN_DIR}/manifests/lr_search.jsonl" ]]; then
  echo "Refusing to reuse canary run directory: ${RUN_DIR}" >&2
  exit 2
fi
test -f "${DATA_ROOT}/benchmarks/tiny_imagenet.npz"
mkdir -p "${RUN_DIR}/logs"

python -m dropout_mft.experiments.benchmark vision plan \
  --run-dir "${RUN_DIR}" --stage lr_search --depth 12

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},DATA_ROOT=${DATA_ROOT},STAGE=lr_search,NUM_SHARDS=5"

submit_canary() {
  local model_kind="$1"
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --array=0-0 \
    --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
    --export="ALL,${common},FILTER_DATASET=tiny_imagenet,FILTER_MODEL_KIND=${model_kind},FILTER_PROFILE=uniform" \
    "${HERE}/run_h100.sbatch"
}

mlp_job=$(submit_canary mlp)
vit_job=$(submit_canary transformer)
echo "Tiny ImageNet MLP canary ${mlp_job}"
echo "Tiny ImageNet ViT canary ${vit_job}"
echo "Track: squeue -j ${mlp_job},${vit_job}"
echo "Logs:  ${RUN_DIR}/logs"
