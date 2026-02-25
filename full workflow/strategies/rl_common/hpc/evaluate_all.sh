#!/bin/bash
# ============================================================================
# evaluate_all.sh — Evaluate all trained RL strategies and compare
#
# Runs after training completes. Evaluates each strategy on the same
# test episodes and generates thesis-quality comparison plots.
#
# Usage:
#   bash evaluate_all.sh <training_output_dir>
#   bash evaluate_all.sh outputs/rl_training/20260222_120000
# ============================================================================

set -euo pipefail

TRAIN_DIR="${1:?Usage: bash evaluate_all.sh <training_output_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATEGIES_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORKFLOW_DIR="$(dirname "$STRATEGIES_DIR")"
EVAL_DIR="${TRAIN_DIR}/evaluation"

echo "=================================================="
echo "  RL Strategy Evaluation"
echo "  Training dir: ${TRAIN_DIR}"
echo "  Eval output:  ${EVAL_DIR}"
echo "=================================================="

mkdir -p "${EVAL_DIR}"

# Activate virtual environment
if [ -d "${WORKFLOW_DIR}/venv" ]; then
    source "${WORKFLOW_DIR}/venv/bin/activate"
elif [ -d "${HOME}/venvs/rl_packing" ]; then
    source "${HOME}/venvs/rl_packing/bin/activate"
fi

export PYTHONPATH="${WORKFLOW_DIR}:${PYTHONPATH:-}"
cd "$WORKFLOW_DIR"

# Evaluate each strategy
declare -a STRATEGIES=("rl_dqn" "rl_ppo" "rl_a2c_masked" "rl_hybrid_hh" "rl_pct_transformer" "rl_mcts_hybrid")

for STRAT_NAME in "${STRATEGIES[@]}"; do
    STRAT_DIR="${STRATEGIES_DIR}/${STRAT_NAME}"
    CHECKPOINT_DIR="${TRAIN_DIR}/${STRAT_NAME}/checkpoints"

    echo ""
    echo "── Evaluating: ${STRAT_NAME} ──"

    if [ -d "$CHECKPOINT_DIR" ] && [ -f "${STRAT_DIR}/evaluate.py" ]; then
        python "${STRAT_DIR}/evaluate.py" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --output_dir "${EVAL_DIR}/${STRAT_NAME}" \
            --num_episodes 100 \
            --seed 42 \
            2>&1 || echo "  WARNING: ${STRAT_NAME} evaluation failed"
    else
        echo "  SKIPPED: No checkpoint found at ${CHECKPOINT_DIR}"
    fi
done

# Generate comparison plots
echo ""
echo "── Generating comparison plots ──"
python "${SCRIPT_DIR}/../compare_strategies.py" \
    --eval_dir "${EVAL_DIR}" \
    --output_dir "${EVAL_DIR}/comparison" \
    2>&1 || echo "  WARNING: Comparison plot generation failed"

echo ""
echo "=================================================="
echo "  Evaluation complete!"
echo "  Results: ${EVAL_DIR}"
echo "  Plots:   ${EVAL_DIR}/comparison/"
echo "=================================================="
