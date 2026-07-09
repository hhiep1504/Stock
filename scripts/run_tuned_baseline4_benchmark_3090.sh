#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

ARGS=()
if [[ "${1:-}" == "--quick" ]]; then
  ARGS+=(--quick)
  shift
fi

echo "[tuned-baseline4] manifest=${PROJECT_ROOT}/configs/tuned_baseline4_benchmark_manifest.json"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_tuned_feature_benchmark.py" \
  --manifest "${PROJECT_ROOT}/configs/tuned_baseline4_benchmark_manifest.json" \
  "${ARGS[@]}" \
  "$@"

echo "[tuned-baseline4] done"
