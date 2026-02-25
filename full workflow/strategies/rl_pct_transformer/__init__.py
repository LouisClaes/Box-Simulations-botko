"""
rl_pct_transformer — PCT-inspired Transformer RL strategy for 3D bin packing.

WHAT THIS STRATEGY DOES
-----------------------
This strategy uses a Transformer-based actor-critic trained with PPO to
select box placements on EUR pallets.  Unlike grid-based approaches that
score every cell on a fixed-resolution map, this strategy generates a small
set of geometrically meaningful candidate placements (extreme points, EMS
corners, floor gaps) and uses a Transformer with a pointer decoder to select
the best one.  The result is an architecture that can handle variable-size
action spaces naturally and attends globally over all candidate placements
before committing to one.

HOW IT WORKS
------------
- **Candidate generation**: For each box, the candidate generator produces
  up to 200 physically-motivated placement positions (extreme points, EMS
  corners, floor grid scan) across both bins.  Each candidate is described by
  12 features: bin identity, normalised x/y/z position, support ratio, fill
  after placing, surface contact ratio, height gap, and orientation index.
- **Transformer encoder**: The current box is encoded by an MLP to a 128-dim
  embedding.  Each candidate is independently encoded to 128-dim.  All
  embeddings are concatenated into a sequence of length (1 + N_candidates)
  and passed through 3 layers of multi-head self-attention (4 heads).  Every
  token attends to every other, so the box embedding "sees" all candidate
  options globally before any decision is made.
- **Pointer decoder**: A single cross-attention layer uses the box token as
  query and all candidate tokens as keys/values.  The resulting attention
  logits, optionally divided by sqrt(d_model), are the action log-probabilities
  over the candidate set (pointer-network style from Vinyals et al. 2015).
- **Variable action space**: Because the Transformer operates on sequences of
  arbitrary length, the strategy handles episodes where the number of valid
  candidates changes from step to step without padding or fixed-size outputs.
- **PPO training**: The agent is trained with PPO using 16 parallel environments,
  undiscounted returns (gamma=1.0 following the PCT paper), and GAE for
  advantage estimation.

PAPER BASIS
-----------
- Zhao et al. (ICLR 2022): "Learning Efficient Online 3D Bin Packing on
  Packing Configuration Trees" — the PCT architecture, pointer mechanism,
  and candidate-set representation that this module extends.
- Zhao et al. (IJRR 2025): Extended PCT with MCTS buffer search — the
  enable_buffer_search lookahead feature is drawn from this work.
- Vaswani et al. (2017): "Attention Is All You Need" — the Transformer
  architecture used as the encoder.
- Vinyals et al. (2015): "Pointer Networks" — the pointer decoder that
  selects among variable-size candidate sets using attention.
- Kool et al. (2019): Attention Model for combinatorial optimisation —
  confirms that pointer-style attention works well for routing/packing.

QUICK USAGE EXAMPLE
-------------------
After training::

    from strategies import get_strategy
    strategy = get_strategy("rl_pct_transformer")
    # Requires a checkpoint at outputs/rl_pct_transformer/logs/best_model.pt
    # Use with PackingSession, benchmark_all.py, run_experiment.py, etc.

To train from scratch (PPO + parallel envs, ~200k episodes)::

    python strategies/rl_pct_transformer/train.py --total_episodes 200000

To evaluate against heuristic baselines::

    python strategies/rl_pct_transformer/evaluate.py --checkpoint outputs/rl_pct_transformer/logs/best_model.pt

KEY HYPERPARAMETERS
-------------------
- ``d_model = 128``           — Transformer embedding dimension (increase for capacity)
- ``nhead = 4``               — Attention heads (d_model must be divisible by nhead)
- ``num_encoder_layers = 3``  — Depth of Transformer encoder
- ``max_candidates = 200``    — Max placement candidates per step (memory/speed tradeoff)
- ``gamma = 1.0``             — Undiscounted returns (finite episodes, following PCT paper)
- ``learning_rate = 3e-4``    — Adam LR with cosine decay and 2% warmup

EXPECTED PERFORMANCE
--------------------
- Training time: ~24-48 hours on CPU, ~6-12 hours with GPU (200,000 episodes)
- Fill rate after training: ~64-72% avg closed fill (best expected RL result)
- Inference speed: ~20-80ms per box depending on candidate count
- Fallback: uses extreme_points heuristic when no checkpoint is loaded

NETWORK ARCHITECTURE SUMMARY
-----------------------------
Input: box features (5-dim) + N candidate placement features (12-dim each)

    ItemEncoder:        (5,) -> Dense(64) -> ReLU -> Dense(128) -> 128-dim
    CandidateEncoder:   (12,) -> Dense(64) -> ReLU -> Dense(128) -> 128-dim
    TransformerEncoder: (1+N, 128) -> 3x MultiHeadSelfAttention(4 heads) -> (1+N, 128)
    PointerDecoder:     query=item_token(128), keys=candidate_tokens(128) -> logits(N,)
    ValueHead:          mean_pool(1+N, 128) -> Dense(64) -> Dense(1) -> V(s)

Modules:
    config               Hyperparameter dataclass (PCTTransformerConfig)
    network              Transformer actor-critic (PCTTransformerNet)
    candidate_generator  Placement candidate generation (CandidateGenerator)
    train                PPO training loop with parallel environments
    evaluate             Evaluation and benchmarking utilities
    strategy             BaseStrategy wrapper for inference (RLPCTTransformerStrategy)
"""

import sys
import os

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

# Register the strategy on import
from strategies.rl_pct_transformer.strategy import RLPCTTransformerStrategy

__all__ = ["RLPCTTransformerStrategy"]
