# HPC Training Guide

## Quick Start

### 1. Transfer to HPC

```bash
# From your local machine, copy the strategies folder to HPC
scp -r "python/full workflow/" user@hpc:/path/to/project/
```

### 2. Setup Environment (one-time)

```bash
bash strategies/rl_common/hpc/setup_hpc.sh
```

This creates a virtual environment at `~/venvs/rl_packing` with all dependencies.

### 3. Train All Strategies

```bash
# Submit all 5 strategies as parallel SLURM jobs
sbatch strategies/rl_common/hpc/train_all.sh

# Or run locally (sequential)
bash strategies/rl_common/hpc/train_all.sh --local
```

### 4. Monitor Training

```bash
# Check SLURM queue
squeue -u $USER

# Watch logs
tail -f outputs/rl_training/*/logs/*.log

# TensorBoard (if port forwarded)
tensorboard --logdir outputs/rl_training/*/*/logs/tensorboard/
```

### 5. Evaluate

```bash
bash strategies/rl_common/hpc/evaluate_all.sh outputs/rl_training/<timestamp>
```

## Resource Requirements

| Strategy | GPUs | CPUs | Memory | Time | Notes |
|----------|------|------|--------|------|-------|
| rl_dqn | 1 | 8 | 32GB | ~12h | Replay buffer memory-heavy |
| rl_ppo | 1 | 16 | 48GB | ~16h | 16 parallel envs |
| rl_a2c_masked | 1 | 16 | 48GB | ~16h | Mask computation overhead |
| rl_hybrid_hh | 0 | 8 | 16GB | ~4h | CPU-only (small network) |
| rl_pct_transformer | 1 | 16 | 48GB | ~16h | Transformer attention |

**Total**: 4 GPUs, ~48h wall time (all parallel), ~160GB combined memory

## Training Individual Strategies

```bash
cd "python/full workflow"

# DDQN
python strategies/rl_dqn/train.py --episodes 50000 --batch_size 256

# PPO
python strategies/rl_ppo/train.py --total_timesteps 5000000 --num_envs 16

# A2C with masking
python strategies/rl_a2c_masked/train.py --num_updates 200000 --num_envs 16

# Hybrid Hyper-Heuristic (fast!)
python strategies/rl_hybrid_hh/train.py --mode dqn --episodes 50000

# PCT Transformer
python strategies/rl_pct_transformer/train.py --episodes 200000 --num_envs 16
```

## SLURM Customisation

Edit `train_all.sh` to match your HPC:
- `#SBATCH --partition=gpu` → your GPU partition name
- `module load Python/3.10.8-GCCcore-12.2.0` → your Python module
- `module load CUDA/12.1.1` → your CUDA module

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Set `PYTHONPATH` to workflow root |
| Out of GPU memory | Reduce `--batch_size` or `--num_envs` |
| SLURM timeout | Increase `--time` or reduce `--episodes` |
| No GPU detected | Check `module load CUDA` and `nvidia-smi` |
