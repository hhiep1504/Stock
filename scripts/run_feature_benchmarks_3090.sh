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

CONFIGS=(
  "${PROJECT_ROOT}/configs/feature_benchmark_baseline4.json"
  "${PROJECT_ROOT}/configs/feature_benchmark_baseline_plus_sma12.json"
  "${PROJECT_ROOT}/configs/feature_benchmark_screened_with_hurst.json"
)

EXTRA_ARGS=()
if [[ "${QUICK}" -eq 1 ]]; then
  EXTRA_ARGS=(--gnn-epochs 10 --baseline-epochs 10 --benchmark-runs 1)
fi

for config_path in "${CONFIGS[@]}"; do
  echo "[feature-benchmark] config=${config_path}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/main.py" \
    --mode benchmark-walkforward \
    --config "${config_path}" \
    "${EXTRA_ARGS[@]}"
done

echo "[feature-benchmark] done"
