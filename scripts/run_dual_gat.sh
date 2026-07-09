#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_arg.py" tune-graph --preset dual_graph_fixed --family gat_dual "$@"

read -r -p "Done. Press Enter to exit..." _
