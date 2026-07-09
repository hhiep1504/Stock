#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

CONFIG_PATH="${PROJECT_ROOT}/configs/tune_baseline4_static_gcn_lstm.json"
TRIALS=""
MAX_EPOCHS=""
PRUNE_AFTER=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      TRIALS="${TRIALS:-2}"
      MAX_EPOCHS="${MAX_EPOCHS:-2}"
      PRUNE_AFTER="${PRUNE_AFTER:-1}"
      shift
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

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/main.py"
  --mode tune
  --config "${CONFIG_PATH}"
  --families gcn_lstm
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

echo "[baseline4-static-gcn-lstm] config=${CONFIG_PATH}"
echo "[baseline4-static-gcn-lstm] command=${CMD[*]} ${EXTRA_ARGS[*]}"

"${CMD[@]}" "${EXTRA_ARGS[@]}"

echo "[baseline4-static-gcn-lstm] done"
