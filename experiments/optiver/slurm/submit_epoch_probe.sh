#!/usr/bin/env bash
# Thin Slurm wrapper around the existing Optiver manifest/worker modules.

set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${OPTIVER_CACHE:?Set OPTIVER_CACHE}"

SBATCH_ARGS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
TRIAL_TIME="${TRIAL_TIME:-02:00:00}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
manifest="${RUN_DIR}/manifests/epoch_probe.jsonl"
[[ ! -e "${manifest}" ]] || { echo "Refusing existing manifest: ${manifest}" >&2; exit 2; }
[[ -f "${OPTIVER_CACHE}" ]] || { echo "Missing cache: ${OPTIVER_CACHE}" >&2; exit 2; }
mkdir -p "${RUN_DIR}/logs" "$(dirname "${manifest}")"
python experiments/optiver/run_winning_mlp.py plan-epoch-probe --manifest "${manifest}"

common="PROJECT_DIR=${PROJECT_DIR},RUN_DIR=${RUN_DIR},VENV_DIR=${VENV_DIR},OPTIVER_CACHE=${OPTIVER_CACHE},OPTIVER_STAGE=epoch_probe"
canary=$(sbatch --parsable "${SBATCH_ARGS[@]}" --time="${TRIAL_TIME}" \
  --array="0,20,50,70%4" --job-name=optiver-epoch-canary \
  --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" --export="ALL,${common}" \
  "${HERE}/run_winning_mlp_h100.sbatch")
canary="${canary%%;*}"
[[ "${canary}" =~ ^[0-9]+$ ]]

probe=$(sbatch --parsable "${SBATCH_ARGS[@]}" --time="${TRIAL_TIME}" \
  --dependency="afterok:${canary}" --array="0-99%${MAX_CONCURRENT}" \
  --job-name=optiver-epoch-probe \
  --output="${RUN_DIR}/logs/slurm-%x-%A_%a.out" --export="ALL,${common}" \
  "${HERE}/run_winning_mlp_h100.sbatch")
probe="${probe%%;*}"
[[ "${probe}" =~ ^[0-9]+$ ]]

printf '%s\n' "${canary},${probe}" >"${RUN_DIR}/slurm-chain-jobs.txt"
echo "canaries ${canary}"
echo "paired probe ${probe}"
echo "CHAIN_JOBS=${canary},${probe}"
