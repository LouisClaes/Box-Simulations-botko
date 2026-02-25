"""
rl_common — Shared infrastructure for all RL-based packing strategies.

WHAT THIS PACKAGE PROVIDES
--------------------------
This package contains the building blocks that ALL five RL strategies
(rl_dqn, rl_ppo, rl_a2c_masked, rl_hybrid_hh, rl_pct_transformer) share.
It exists so that each strategy only needs to implement its own network
and training loop — all the plumbing for environments, observations,
rewards, and logging is handled here in one place.

COMPONENTS
----------

BinPackingEnv (environment.py)
    A Gymnasium-compatible environment that wraps PackingSession for RL
    training.  Models the Botko BV dual-pallet setup:
      - 2 EUR pallets (1200x800mm), height cap 2700mm, close at 1800mm
      - Conveyor belt with 8 visible boxes, pick window of 4
      - Flat discrete action space: (box_idx, bin_idx, x_grid, y_grid, orient)
    Observation dict keys:
      - "heightmaps":     (num_bins, 120, 80) normalised to [0, 1]
      - "box_features":   (pick_window, 4) current grippable boxes
      - "buffer_features": (buffer_size, 4) all visible boxes
      - "bin_stats":      (num_bins, 4) fill, max_height, roughness, n_boxes
      - "action_mask":    (total_actions,) 1=valid, 0=invalid

make_env (environment.py)
    Factory function to create one or many BinPackingEnvs for parallel
    training.  Uses SyncVectorEnv for <=8 envs, AsyncVectorEnv for >8.
    Example::

        envs = make_env(config, num_envs=16, seed=42)

RewardShaper (rewards.py)
    Converts raw packing outcomes into shaped reward signals.  The default
    configuration combines five components calibrated for the Botko BV setup:
      1. Volume weight (10.0)  — primary signal: reward placed volume
      2. Fill delta weight (5.0) — bonus for increasing pallet fill rate
      3. Surface contact weight (2.0) — bonus for boxes resting on others
      4. Height penalty (-1.0) — penalise placing at high Z (fill bottom first)
      5. Close bonus (5.0)  — reward when a pallet closes with good fill
    All weights are configurable via RewardConfig for ablation studies.

TrainingLogger (logger.py)
    Unified logger that simultaneously writes to:
      - CSV files (for analysis in pandas/Excel after training)
      - TensorBoard (for live monitoring during training via tensorboard --logdir)
      - Console (human-readable progress for HPC job logs)
    Also generates matplotlib training-curve plots automatically.
    Example::

        logger = TrainingLogger(log_dir="outputs/rl_dqn/logs", strategy_name="rl_dqn")
        logger.log_episode(episode, reward=12.5, fill=0.72, loss=0.05)
        logger.plot_training_curves()
        logger.close()

encode_heightmap / encode_box_features / encode_buffer_features (obs_utils.py)
    Utility functions that convert raw BinState and Box objects into
    normalised numpy arrays for network input.  All strategies use these
    functions to ensure consistent observation encoding across experiments.
    The 4-channel encoding (height + item_l + item_w + item_h) follows
    Zhao et al. (AAAI 2021).  The 2-channel version (height + action_map)
    follows Tsang et al. (2025).

HOW STRATEGIES USE THIS PACKAGE
--------------------------------
Each RL strategy imports from rl_common like this::

    from strategies.rl_common.environment import BinPackingEnv, make_env
    from strategies.rl_common.rewards import RewardShaper, RewardConfig
    from strategies.rl_common.logger import TrainingLogger
    from strategies.rl_common.obs_utils import encode_heightmap, encode_box_features

The environment is then instantiated with strategy-specific EnvConfig settings
(action grid resolution, number of bins, etc.) while the reward shaping and
logging are shared unchanged.

Provides:
    BinPackingEnv           Gymnasium environment wrapping PackingSession
    make_env                Factory for single or vectorised environments
    RewardShaper            Configurable multi-component reward computation
    RewardConfig            Dataclass of reward weight coefficients
    TrainingLogger          TensorBoard + CSV + matplotlib logger
    encode_heightmap        Normalised 2D heightmap encoder
    encode_box_features     Box dimension/volume/weight encoder
    encode_buffer_features  Buffer (multi-box) feature encoder
"""

from strategies.rl_common.environment import BinPackingEnv, make_env
from strategies.rl_common.rewards import RewardShaper
from strategies.rl_common.logger import TrainingLogger
from strategies.rl_common.obs_utils import (
    encode_heightmap,
    encode_box_features,
    encode_buffer_features,
)

__all__ = [
    "BinPackingEnv",
    "make_env",
    "RewardShaper",
    "TrainingLogger",
    "encode_heightmap",
    "encode_box_features",
    "encode_buffer_features",
]
