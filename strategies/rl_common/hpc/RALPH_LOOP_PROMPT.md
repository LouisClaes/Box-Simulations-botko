# Ralph Loop Prompt - RL HPC Production Pipeline (Botko 3D Packing)

Use this prompt with `mcp__ralph-loop__ralph_loop`.

Recommended completion promise:
`RL_HPC_READY`

---

## Prompt

You are working in:
`python/strategies`

You may only rely on code and assets inside these folders because these are the only folders transferred to HPC:

- `python/strategies`
- `python/strategies/rl_a2c_masked`
- `python/strategies/rl_common`
- `python/strategies/rl_hybrid_hh`
- `python/strategies/rl_dqn`
- `python/strategies/rl_mcts_hybrid`
- `python/strategies/rl_pct_transformer`
- `python/strategies/rl_ppo`

Primary objective:

1. Make `rl_mcts_hybrid` production-stable first.
2. Ensure one command trains all RL strategies across all available HPC GPUs.
3. Ensure 24h continuous runtime reliability (checkpointing/resume/preemption-safe behavior).
4. Ensure evaluation and visual outputs are thesis-grade and fail-closed (never fake data).

Current orchestrator target file:
`strategies/rl_common/hpc/run_rl_pipeline.py`

Current wrappers:
- `strategies/rl_common/hpc/train_all.sh`
- `strategies/rl_common/hpc/evaluate_all.sh`

Current strict comparison file:
- `strategies/rl_common/compare_strategies.py`

Current mcts trainer:
- `strategies/rl_mcts_hybrid/train.py`

---

## Non-negotiable constraints

1. `rl_mcts_hybrid` must run before other strategies in full pipeline mode.
2. No synthetic plotting data is allowed in production paths.
3. Any missing checkpoint or missing eval artifact must produce explicit failure (unless `--continue_on_error` is set).
4. Every strategy run must produce structured status entries in a run manifest (`run_manifest.json`).
5. Every long run must be resumable with deterministic state restoration where supported.
6. Every figure in `evaluation/comparison/` must be generated from real run outputs only.

---

## Required deliverables

1. One-command pipeline works from shell wrapper:
   - `bash strategies/rl_common/hpc/train_all.sh`
2. Pipeline supports modes:
   - `--mode train`
   - `--mode evaluate`
   - `--mode visualize`
   - `--mode full`
3. `rl_mcts_hybrid/train.py` includes:
   - atomic checkpoints
   - `latest.pt`
   - `--resume auto|latest|<path>`
   - signal-safe interrupted checkpointing
   - stateful resume (optimizer, counters, stage, RNG)
4. Normalized evaluator output schema exists for all strategies:
   - `avg_fill`
   - `fill_std`
   - `placement_rate`
   - `avg_pallets_closed`
   - `ms_per_box`
   - `support_mean`
   - `fill_rates`
5. Thesis visuals exist in:
   - `evaluation/comparison/`
   - include at least bar comparison, distribution grid, training grid, summary CSV
6. Readme/docs updated for new pipeline usage and the 6-strategy setup.

---

## Validation checklist (must pass)

1. Syntax:
   - `python -m py_compile strategies/rl_common/hpc/run_rl_pipeline.py`
   - `python -m py_compile strategies/rl_mcts_hybrid/train.py`
   - `python -m py_compile strategies/rl_common/compare_strategies.py`
2. Dry run:
   - `python strategies/rl_common/hpc/run_rl_pipeline.py --mode full --profile quick --dry_run`
3. Minimal real smoke run (quick profile):
   - run pipeline with at least `rl_mcts_hybrid` + one other strategy
   - confirm `run_manifest.json` status entries
   - confirm `evaluation/<strategy>/eval_results.json` normalized files
4. Visualization integrity:
   - no files generated from dummy data fallback
   - `compare_strategies.py --strict` succeeds only when real data exists

---

## Iteration strategy

On each iteration:

1. Focus first on highest-risk breakpoints:
   - wrong CLI adapters
   - checkpoint discovery mismatch
   - broken eval output normalization
   - silent failure behavior
2. Run targeted smoke checks after each patch.
3. Record exact command and artifact paths in iteration notes.
4. Keep modifications restricted to transferred strategy folders.

---

## Paper-grounded analysis requirements for thesis outputs

Ensure docs and visualization captions reference these works where relevant:

- Zhao et al. 2021 (AAAI), constrained DRL for online 3D BPP, DOI: 10.1609/aaai.v35i1.16155
- Zhao et al. 2022/2025 PCT and deliberate planning line (ICLR/journal extension)
- Verma et al. 2020 (AAAI), generalized RL for online 3D BPP
- Tsang et al. 2025, dual-bin DRL in Computers in Industry, DOI: 10.1016/j.compind.2024.104202
- Ali et al. 2025, stability vs packing efficiency, DOI: 10.1016/j.cor.2025.107005
- Gao et al. 2025, fast stability + rearrangement planning (arXiv)

Do not fabricate citation metadata. If uncertain, mark as "verify before final manuscript".

---

## Completion condition

Only finish when all of the following are true:

1. Full pipeline runs from one command.
2. `rl_mcts_hybrid` is first and stable with robust resume.
3. Evaluation is strict and normalized.
4. Thesis figures are generated from real artifacts.
5. Updated docs explain how to run end-to-end on HPC.

When done, output exactly:

`<promise>RL_HPC_READY</promise>`

---

## Suggested Ralph call

```text
mcp__ralph-loop__ralph_loop(
  prompt=<paste prompt section above>,
  completion_promise="RL_HPC_READY",
  max_iterations=12,
  git_enabled=true,
  auto_commit=false
)
```
