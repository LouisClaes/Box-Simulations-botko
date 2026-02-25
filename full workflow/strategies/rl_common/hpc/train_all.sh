#!/bin/bash
# ============================================================================
# train_all.sh — Launch ALL RL strategy training jobs on SLURM HPC
#
# Usage:
#   sbatch train_all.sh              # Submit all jobs
#   bash train_all.sh --local        # Run sequentially on local machine
#
# This script submits 6 independent training jobs that run in parallel,
# each training a different RL strategy. Total GPU hours: ~48-72h.
#
# Job dependency: None — all strategies train independently.
# After training: run evaluate_all.sh to compare results.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATEGIES_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORKFLOW_DIR="$(dirname "$STRATEGIES_DIR")"
OUTPUT_DIR="${WORKFLOW_DIR}/outputs/rl_training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=================================================="
echo "  RL Strategy Training Launcher"
echo "  Strategies dir: ${STRATEGIES_DIR}"
echo "  Output dir:     ${OUTPUT_DIR}/${TIMESTAMP}"
echo "=================================================="

mkdir -p "${OUTPUT_DIR}/${TIMESTAMP}/logs"

# Check if running locally or on SLURM
if [[ "${1:-}" == "--local" ]]; then
    echo "Running locally (sequential)..."
    LOCAL_MODE=true
else
    LOCAL_MODE=false
    echo "Submitting SLURM jobs (parallel)..."
fi

# ── Strategy configurations ──────────────────────────────────────────────
# Format: strategy_name | train_script | gpu_time | num_gpus | num_cpus | memory

declare -a STRATEGIES=(
    "rl_dqn|train.py|12:00:00|1|8|32G|--episodes 50000 --batch_size 256 --lr 0.001"
    "rl_ppo|train.py|16:00:00|1|16|48G|--total_timesteps 5000000 --num_envs 16 --lr 3e-4"
    "rl_a2c_masked|train.py|16:00:00|1|16|48G|--num_updates 200000 --num_envs 16 --lr 1e-4"
    "rl_hybrid_hh|train.py|04:00:00|0|8|16G|--mode dqn --episodes 50000"
    "rl_pct_transformer|train.py|16:00:00|1|16|48G|--episodes 200000 --num_envs 16 --lr 3e-4"
    "rl_mcts_hybrid|train.py|24:00:00|1|16|48G|--total_timesteps 5000000 --num_envs 16 --lr 3e-4"
)

JOB_IDS=()

for entry in "${STRATEGIES[@]}"; do
    IFS='|' read -r STRAT_NAME TRAIN_SCRIPT GPU_TIME NUM_GPUS NUM_CPUS MEMORY EXTRA_ARGS <<< "$entry"

    STRAT_DIR="${STRATEGIES_DIR}/${STRAT_NAME}"
    LOG_FILE="${OUTPUT_DIR}/${TIMESTAMP}/logs/${STRAT_NAME}.log"

    echo ""
    echo "── ${STRAT_NAME} ──"
    echo "  Script: ${STRAT_DIR}/${TRAIN_SCRIPT}"
    echo "  GPUs: ${NUM_GPUS} | CPUs: ${NUM_CPUS} | Memory: ${MEMORY}"
    echo "  Time limit: ${GPU_TIME}"
    echo "  Args: ${EXTRA_ARGS}"

    if [[ "$LOCAL_MODE" == true ]]; then
        echo "  Running locally..."
        cd "$WORKFLOW_DIR"
        python "${STRAT_DIR}/${TRAIN_SCRIPT}" \
            --output_dir "${OUTPUT_DIR}/${TIMESTAMP}/${STRAT_NAME}" \
            ${EXTRA_ARGS} \
            2>&1 | tee "$LOG_FILE" || echo "  WARNING: ${STRAT_NAME} failed"
    else
        # Submit SLURM job
        JOB_SCRIPT=$(mktemp /tmp/slurm_${STRAT_NAME}_XXXXXX.sh)
        cat > "$JOB_SCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=rl_${STRAT_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE%.log}.err
#SBATCH --time=${GPU_TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NUM_CPUS}
#SBATCH --mem=${MEMORY}
$([ "${NUM_GPUS}" -gt 0 ] && echo "#SBATCH --gres=gpu:${NUM_GPUS}")
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL

# Load modules (adjust for your HPC)
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true
module load cuDNN/8.9.2.26-CUDA-12.1.1 2>/dev/null || true

# Activate virtual environment
if [ -d "${WORKFLOW_DIR}/venv" ]; then
    source "${WORKFLOW_DIR}/venv/bin/activate"
elif [ -d "\${HOME}/venvs/rl_packing" ]; then
    source "\${HOME}/venvs/rl_packing/bin/activate"
fi

# Set environment
export PYTHONPATH="${WORKFLOW_DIR}:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=${NUM_CPUS}
export MKL_NUM_THREADS=${NUM_CPUS}

echo "============================================="
echo "Job: rl_${STRAT_NAME}"
echo "Node: \$(hostname)"
echo "GPUs: \$(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Python: \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA: \$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "Start: \$(date)"
echo "============================================="

cd "${WORKFLOW_DIR}"
python "${STRAT_DIR}/${TRAIN_SCRIPT}" \\
    --output_dir "${OUTPUT_DIR}/${TIMESTAMP}/${STRAT_NAME}" \\
    ${EXTRA_ARGS}

echo "============================================="
echo "End: \$(date)"
echo "============================================="
SLURM_EOF

        JOB_ID=$(sbatch "$JOB_SCRIPT" 2>/dev/null | awk '{print $NF}')
        if [ -n "$JOB_ID" ]; then
            JOB_IDS+=("$JOB_ID")
            echo "  Submitted: Job ID ${JOB_ID}"
            rm -f "$JOB_SCRIPT"
        else
            echo "  WARNING: sbatch not available. Script saved at: ${JOB_SCRIPT}"
        fi
    fi
done

echo ""
echo "=================================================="
if [[ "$LOCAL_MODE" == true ]]; then
    echo "  All local training complete!"
else
    echo "  Submitted ${#JOB_IDS[@]} SLURM jobs:"
    for jid in "${JOB_IDS[@]}"; do
        echo "    - Job ${jid}"
    done
    echo ""
    echo "  Monitor: squeue -u \$USER"
    echo "  Logs:    ${OUTPUT_DIR}/${TIMESTAMP}/logs/"
    echo "  Cancel:  scancel ${JOB_IDS[*]}"
fi
echo "=================================================="
