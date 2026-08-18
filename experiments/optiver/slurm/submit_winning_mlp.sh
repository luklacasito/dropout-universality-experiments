#!/usr/bin/env bash
# Canary-gated Optiver winner-MLP versus parameter-matched depth-12 sweep.

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR to a new Optiver run directory}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${OPTIVER_CACHE:?Set OPTIVER_CACHE}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
TRIAL_TIME="${TRIAL_TIME:-01:00:00}"
CONTROL_GPU_TYPE="${CONTROL_GPU_TYPE:-v100-16}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${RUN_DIR}/manifests/selection.jsonl" ]]; then
  echo "Refusing to reuse Optiver run directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${OPTIVER_CACHE}" ]]; then
  echo "Missing Optiver cache: ${OPTIVER_CACHE}" >&2
  exit 2
fi
if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/manifests"
python experiments/optiver/run_winning_mlp.py plan \
  --manifest "${RUN_DIR}/manifests/selection.jsonl" --seeds 3

parse_job_id() {
  local value="${1%%;*}"
  [[ "${value}" =~ ^[0-9]+$ ]] || return 2
  printf '%s\n' "${value}"
}

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR}"
common="${common},OPTIVER_CACHE=${OPTIVER_CACHE}"

# Indices 0 and 10 are respectively the first exact-shallow and depth-12
# trials. Their outputs are content-addressed and are reused by the full array.
canary_raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --partition=GPU-shared --gpus=h100-80:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G --time="${TRIAL_TIME}" \
  --array="0,10%2" --job-name=dropout-optiver-canary \
  --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
  --export="ALL,${common},OPTIVER_STAGE=selection" \
  "${HERE}/run_winning_mlp_h100.sbatch")
canary_job=$(parse_job_id "${canary_raw}")
echo "Optiver shallow/deep canaries ${canary_job}"

selection_raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --partition=GPU-shared --gpus=h100-80:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G --time="${TRIAL_TIME}" \
  --dependency="afterok:${canary_job}" \
  --array="0-92%${MAX_CONCURRENT}" --job-name=dropout-optiver-selection \
  --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
  --export="ALL,${common},OPTIVER_STAGE=selection" \
  "${HERE}/run_winning_mlp_h100.sbatch")
selection_job=$(parse_job_id "${selection_raw}")
echo "Optiver validation selection  ${selection_job}"

select_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/optiver/run_winning_mlp.py \
plan-confirmation --selection-dir ${RUN_DIR}/selection \
--manifest ${RUN_DIR}/manifests/confirmation.jsonl \
--selection-json ${RUN_DIR}/selection.json"
select_raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${selection_job}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G --time=00:15:00 \
  --job-name=dropout-optiver-select-p \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" --wrap="${select_wrap}")
select_job=$(parse_job_id "${select_raw}")
echo "Select budgets + plan test   ${select_job}"

confirmation_raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --partition=GPU-shared --gpus=h100-80:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G --time="${TRIAL_TIME}" \
  --dependency="afterok:${select_job}" \
  --array="0-49%${MAX_CONCURRENT}" --job-name=dropout-optiver-confirm \
  --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" \
  --export="ALL,${common},OPTIVER_STAGE=confirmation" \
  "${HERE}/run_winning_mlp_h100.sbatch")
confirmation_job=$(parse_job_id "${confirmation_raw}")
echo "Optiver fresh-seed test      ${confirmation_job}"

aggregate_wrap="source ${VENV_DIR}/bin/activate && cd ${PROJECT_DIR} && \
PYTHONPATH=${PROJECT_DIR}/src python experiments/optiver/run_winning_mlp.py \
aggregate --run-dir ${RUN_DIR}"
aggregate_raw=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
  --dependency="afterok:${confirmation_job}" \
  --partition=GPU-shared --gpus="${CONTROL_GPU_TYPE}:1" \
  --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=10G --time=00:15:00 \
  --job-name=dropout-optiver-aggregate \
  --output="${RUN_DIR}/logs/slurm-%x-%j.out" --wrap="${aggregate_wrap}")
aggregate_job=$(parse_job_id "${aggregate_raw}")
echo "Aggregate                     ${aggregate_job}"

chain_jobs="${canary_job},${selection_job},${select_job},${confirmation_job},${aggregate_job}"
printf '%s\n' "${chain_jobs}" >"${RUN_DIR}/slurm-chain-jobs.txt"
echo "CHAIN_JOBS=${chain_jobs}"
echo "Trials: 93 validation-selection + 50 fresh-seed confirmation = 143"
echo "At most ${MAX_CONCURRENT} H100s run concurrently."
echo "Status: python experiments/optiver/run_winning_mlp.py status --run-dir ${RUN_DIR}"
