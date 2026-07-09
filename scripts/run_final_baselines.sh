#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

if [[ $# -eq 0 ]]; then
  families=(lstm gru cnn_lstm)
  for family in "${families[@]}"; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_arg.py" final-baseline --family "${family}"
  done
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_arg.py" final-baseline "$@"
fi

read -r -p "Done. Press Enter to exit..." _
