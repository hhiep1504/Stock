#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
  shift
fi

EXTRA_ARGS=()
if [[ "${QUICK}" -eq 1 ]]; then
  EXTRA_ARGS=(
    --epochs 5
    --runs 1
    --max-candidates 8
    --top-candidates 8
    --max-greedy-features 3
  )
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/sweep_feature_windows_weekly.py" \
  --device auto \
  "${EXTRA_ARGS[@]}" \
  "$@"
