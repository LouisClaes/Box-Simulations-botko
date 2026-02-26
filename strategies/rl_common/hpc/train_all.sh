#!/bin/bash
# ============================================================================
# train_all.sh - One-command RL pipeline launcher for HPC and local runs
#
# Default behavior:
#   bash train_all.sh
#   sbatch train_all.sh
#
# Explicit mode examples:
#   bash train_all.sh --mode train
#   bash train_all.sh --mode full --profile quick --dry_run
# ============================================================================

#SBATCH --job-name=rl_pipeline
#SBATCH --output=rl_pipeline_%j.out
#SBATCH --error=rl_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATEGIES_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORKFLOW_DIR="$(dirname "$STRATEGIES_DIR")"

# Load modules if available
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true
module load cuDNN/8.9.2.26-CUDA-12.1.1 2>/dev/null || true

# Activate venv if found
if [ -d "${WORKFLOW_DIR}/venv" ]; then
    source "${WORKFLOW_DIR}/venv/bin/activate"
elif [ -d "${HOME}/venvs/rl_packing" ]; then
    source "${HOME}/venvs/rl_packing/bin/activate"
fi

export PYTHONPATH="${WORKFLOW_DIR}:${PYTHONPATH:-}"
cd "${WORKFLOW_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ "$#" -eq 0 ]; then
    set -- --mode full
fi

echo "=================================================="
echo "RL Pipeline Launcher"
echo "Workflow dir: ${WORKFLOW_DIR}"
echo "Python:       ${PYTHON_BIN}"
echo "Args:         $*"
echo "GPU visible:  ${CUDA_VISIBLE_DEVICES:-auto}"
echo "=================================================="

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_rl_pipeline.py" "$@"
