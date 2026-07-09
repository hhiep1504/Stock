#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

QUICK=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

EXTRA_ARGS=()
if [[ "${QUICK}" -eq 1 ]]; then
  EXTRA_ARGS+=(--quick)
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_common_feature_benchmark.py" \
  --device auto \
  "${EXTRA_ARGS[@]}" \
  "$@"
