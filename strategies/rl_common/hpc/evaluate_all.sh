#!/bin/bash
# ============================================================================
# evaluate_all.sh - Evaluate all trained RL strategies and build comparison figs
#
# Usage:
#   bash evaluate_all.sh <run_dir> [extra run_rl_pipeline args]
# Example:
#   bash evaluate_all.sh outputs/rl_training/20260225_210000 --eval_episodes 200
# ============================================================================

set -euo pipefail

RUN_DIR="${1:?Usage: bash evaluate_all.sh <run_dir> [extra args]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATEGIES_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORKFLOW_DIR="$(dirname "$STRATEGIES_DIR")"

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true

if [ -d "${WORKFLOW_DIR}/venv" ]; then
    source "${WORKFLOW_DIR}/venv/bin/activate"
elif [ -d "${HOME}/venvs/rl_packing" ]; then
    source "${HOME}/venvs/rl_packing/bin/activate"
fi

export PYTHONPATH="${WORKFLOW_DIR}:${PYTHONPATH:-}"
cd "${WORKFLOW_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_rl_pipeline.py" --mode evaluate --run_dir "${RUN_DIR}" "$@"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_rl_pipeline.py" --mode visualize --run_dir "${RUN_DIR}" "$@"
