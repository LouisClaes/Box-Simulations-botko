# HPC Best Practices for 24h+ RL Runs

Deze checklist is bedoeld voor lange runs op multi-GPU clusters, met focus op reproduceerbaarheid, herstartbaarheid en correcte evaluatie.

## 1) Reproduceerbaarheid
1. Zet alle relevante seeds (Python, NumPy, Torch, CUDA).
2. Sla RNG-state op in checkpoints en herstel die bij resume.
3. Log de exacte config per run (`config.json`) en commit hash (indien beschikbaar).

Bronnen:
- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

## 2) Resume en checkpointing
1. Gebruik periodieke checkpoints + `latest` symlink/bestand.
2. Schrijf checkpoints atomisch (temp file -> replace).
3. Sla minimaal op: model, optimizer, scheduler, global step/episode, best metric, RNG-state.
4. Test expliciet of interrupted -> resume dezelfde run correct hervat.

Bronnen:
- PyTorch Saving & Loading Models: https://pytorch.org/tutorials/beginner/saving_loading_models.html

## 3) Multi-GPU discipline
1. Bind processen expliciet aan GPU’s (`CUDA_VISIBLE_DEVICES` per job).
2. Gebruik DDP alleen als model-training echt intra-model parallel moet; anders strategy-level parallelism.
3. Monitor NCCL timeouts/hangs en gebruik alleen noodzakelijke NCCL env settings.

Bronnen:
- PyTorch DDP notes: https://pytorch.org/docs/stable/notes/ddp.html
- PyTorch Distributed package: https://pytorch.org/docs/stable/distributed.html
- NCCL env vars: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

## 4) Slurm/preemptie
1. Configureer `--signal` en vang `SIGTERM`/`SIGINT` op voor graceful checkpoint.
2. Gebruik `--requeue` waar mogelijk.
3. Test preemptiepad met een korte job waarin je proces bewust onderbreekt.

Bronnen:
- `sbatch` docs (`--signal`, `--requeue`): https://slurm.schedmd.com/sbatch.html
- Slurm overview: https://slurm.schedmd.com/overview.html

## 5) Evaluatie en visualisatie-integriteit
1. Fail closed: geen dummy-evaluaties in strict mode.
2. Eis volledige strategie-dekking voor vergelijkingen.
3. Scheid train/eval metrics logisch en voorkom CSV schema-crashes.
4. Bewaar genormaliseerde eval output per strategie (`eval_results.json`).

## 6) RL/MCTS methodologische richtlijnen
1. Rapporteer altijd ablations: policy-only vs MCTS-enabled.
2. Houd planning-parameters vast in logs (simulations, depth, temperature).
3. Meet niet alleen fill-rate: ook placement-rate, snelheid (`ms/box`), en robuustheid.

Bronnen:
- AlphaGo Zero (policy + value + MCTS): https://www.nature.com/articles/nature24270
- MuZero (learned model + search): https://www.nature.com/articles/s41586-020-03051-4
- PPO: https://arxiv.org/abs/1707.06347

## 7) Operationele run-gate vóór lange run
1. `--profile quick` smoke-run slaagt end-to-end.
2. Ten minste 1 checkpoint + 1 evaluatieresultaat per strategie.
3. Compare/visualize fase slaagt zonder handmatige interventie.
4. Resume-test (stop + hervat) slaagt op dezelfde run_dir.
