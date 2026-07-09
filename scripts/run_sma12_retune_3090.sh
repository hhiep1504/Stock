#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

CONFIG_PATH="${PROJECT_ROOT}/configs/tune_baseline_plus_sma12.json"
FAMILIES="lstm,gru,cnn_lstm,temporal_gcn,gat_lstm"
TRIALS=""
MAX_EPOCHS=""
PRUNE_AFTER=""
QUICK=0
CUSTOM_CONFIG=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      TRIALS="${TRIALS:-2}"
      MAX_EPOCHS="${MAX_EPOCHS:-2}"
      PRUNE_AFTER="${PRUNE_AFTER:-1}"
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      CUSTOM_CONFIG=1
      shift 2
      ;;
    --families)
      FAMILIES="$2"
      shift 2
      ;;
    --trials)
      TRIALS="$2"
      shift 2
      ;;
    --max-epochs-per-trial)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --prune-after-epochs)
      PRUNE_AFTER="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${QUICK}" -eq 1 && "${CUSTOM_CONFIG}" -eq 0 ]]; then
  CONFIG_PATH="${PROJECT_ROOT}/configs/tune_baseline_plus_sma12_quick.json"
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/main.py"
  --mode tune
  --config "${CONFIG_PATH}"
  --families "${FAMILIES}"
)

if [[ -n "${TRIALS}" ]]; then
  CMD+=(--trials "${TRIALS}")
fi
if [[ -n "${MAX_EPOCHS}" ]]; then
  CMD+=(--max-epochs-per-trial "${MAX_EPOCHS}")
fi
if [[ -n "${PRUNE_AFTER}" ]]; then
  CMD+=(--prune-after-epochs "${PRUNE_AFTER}")
fi

echo "[sma12-retune] config=${CONFIG_PATH}"
echo "[sma12-retune] families=${FAMILIES}"
echo "[sma12-retune] command=${CMD[*]} ${EXTRA_ARGS[*]}"

"${CMD[@]}" "${EXTRA_ARGS[@]}"

echo "[sma12-retune] done"
