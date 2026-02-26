# HPC RL Pipeline Guide

## What changed

As of February 25, 2026, `rl_common/hpc` uses a single orchestrator script:

- `run_rl_pipeline.py`

This replaces fragile per-strategy shell argument wiring and enforces:

- per-strategy CLI adapters (no wrong flags)
- `rl_mcts_hybrid` priority (runs first)
- strict fail-closed evaluation (no dummy/fake comparison plots)
- normalized `eval_results.json` for all strategies
- thesis-ready comparison figures in one run directory

## One-command run

```bash
bash strategies/rl_common/hpc/train_all.sh
```

This defaults to `--mode full` and runs:

1. train all selected strategies
2. evaluate all selected strategies
3. generate comparison + thesis visualization artifacts

## Evaluate an existing run

```bash
bash strategies/rl_common/hpc/evaluate_all.sh outputs/rl_training/<timestamp>
```

## Direct orchestrator usage

```bash
python strategies/rl_common/hpc/run_rl_pipeline.py --mode full
python strategies/rl_common/hpc/run_rl_pipeline.py --mode train --profile quick
python strategies/rl_common/hpc/run_rl_pipeline.py --mode evaluate --run_dir outputs/rl_training/<timestamp>
python strategies/rl_common/hpc/run_rl_pipeline.py --mode visualize --run_dir outputs/rl_training/<timestamp>
```

## Important flags

```bash
--strategies rl_mcts_hybrid,rl_dqn,rl_ppo,rl_a2c_masked,rl_pct_transformer,rl_hybrid_hh
--gpus 0,1,2,3          # or auto
--max_parallel 4
--eval_episodes 100
--continue_on_error
--dry_run
```

## Output structure

```text
outputs/rl_training/<timestamp>/
  run_manifest.json
  logs/
    train_<strategy>.log
    eval_<strategy>.log
    compare_strategies.log
  <strategy>/
    ... training artifacts ...
  evaluation/
    <strategy>/
      eval_results.json
      ... raw evaluator outputs ...
    comparison/
      fill_rate_comparison.png/.pdf
      fill_distribution_grid.png/.pdf
      training_fill_grid.png/.pdf
      summary_table.csv
```

## SLURM notes

`train_all.sh` includes default `#SBATCH` lines (`gpu:4`, `24:00:00`, `--requeue`).
Adjust these for your cluster before production.

## Reproducibility + stability checks

- `run_manifest.json` stores command history and per-strategy status.
- `rl_mcts_hybrid/train.py` now supports robust resume (`--resume auto`), atomic checkpoints, latest checkpoint aliasing, and signal-safe interruption checkpoints.
- comparison generation fails if evaluation results are missing.
