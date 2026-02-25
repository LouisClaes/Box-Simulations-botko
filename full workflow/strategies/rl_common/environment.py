"""
BinPackingEnv — Gymnasium environment wrapping PackingSession.

This environment models the Botko BV dual-pallet online packing problem:
  - 2 EUR pallets (1200×800mm), height cap 2700mm, close at 1800mm
  - Conveyor belt with 8 visible boxes, pick window of 4
  - Strict FIFO: no recirculation
  - Goal: maximise average volumetric fill of closed pallets

The environment exposes a FLAT action space:
  action = (box_index, bin_index, x_grid, y_grid, orientation_index)
  → flattened to a single integer for compatibility with DQN/PPO

Observation space is a dictionary:
  {
    "heightmaps":   (num_bins, grid_l, grid_w)  — normalised [0, 1]
    "box_features":  (pick_window, 4)            — (l, w, h, vol) normalised
    "buffer_features": (buffer_size, 4)          — all visible boxes
    "bin_stats":     (num_bins, 4)               — (fill, max_h, roughness, n_boxes)
    "action_mask":   (num_actions,)              — 1 = valid, 0 = invalid
  }

Usage:
    env = BinPackingEnv(config=env_config)
    obs, info = env.reset()
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

For vectorized training (HPC):
    envs = make_env(env_config, num_envs=128, seed=42)

References:
    - Zhao et al. (AAAI 2021): Feasibility masking approach
    - Tsang et al. (2025): Dual-bin DDQN environment design
    - Xiong et al. (RA-L 2024): Masked PPO with EMS candidates
"""

from __future__ import annotations

import sys
import os
import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# Adjust path so we can import from the full workflow root
_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, BinConfig, Orientation
from simulator.session import PackingSession, SessionConfig, StepObservation
from simulator.close_policy import HeightClosePolicy
from strategies.rl_common.rewards import RewardShaper, RewardConfig


# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """
    Configuration for the BinPackingEnv.

    Controls the physical setup, action space discretisation, and training
    parameters.  All defaults match the Botko BV thesis setup.

    Attributes:
        bin_config:       EUR pallet dimensions.
        num_bins:         Number of simultaneous pallets.
        buffer_size:      Conveyor belt capacity (visible boxes).
        pick_window:      Front N boxes the robot can reach.
        close_height:     Height at which pallets are closed (mm).
        max_consecutive_rejects: Safety valve.
        num_orientations: Number of allowed orientations (2=flat, 6=all).
        action_grid_step: Grid step for action space discretisation (mm).
                          Larger = faster training, smaller = finer placement.
        num_boxes_per_episode: Number of boxes per training episode.
        box_size_range:   (min_dim, max_dim) for random box generation (mm).
        reward_config:    Reward shaping configuration.
        seed:             Random seed for reproducibility.
    """

    # Physical setup (Botko BV defaults)
    bin_config: BinConfig = field(default_factory=lambda: BinConfig(
        length=1200.0, width=800.0, height=2700.0, resolution=10.0,
    ))
    num_bins: int = 2
    buffer_size: int = 8
    pick_window: int = 4
    close_height: float = 1800.0
    max_consecutive_rejects: int = 10

    # Action space
    num_orientations: int = 2          # 2 = flat only, 6 = all
    action_grid_step: float = 10.0     # mm per action grid cell

    # Episode
    num_boxes_per_episode: int = 100
    box_size_range: Tuple[float, float] = (100.0, 600.0)
    box_weight_range: Tuple[float, float] = (1.0, 30.0)

    # Reward
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Reproducibility
    seed: Optional[int] = None

    @property
    def action_grid_l(self) -> int:
        """Action grid cells along length axis."""
        return max(1, int(self.bin_config.length / self.action_grid_step))

    @property
    def action_grid_w(self) -> int:
        """Action grid cells along width axis."""
        return max(1, int(self.bin_config.width / self.action_grid_step))

    @property
    def total_placement_actions(self) -> int:
        """Total placement actions = grid_l × grid_w × orientations × bins."""
        return self.action_grid_l * self.action_grid_w * self.num_orientations * self.num_bins

    @property
    def total_actions(self) -> int:
        """Total actions = placements per box × pick_window + 1 (skip)."""
        # For each grippable box: all placements across all bins
        # Plus one "skip/advance" action
        return self.pick_window * self.total_placement_actions + 1

    def get_session_config(self) -> SessionConfig:
        """Convert to SessionConfig for PackingSession."""
        return SessionConfig(
            bin_config=self.bin_config,
            num_bins=self.num_bins,
            buffer_size=self.buffer_size,
            pick_window=self.pick_window,
            close_policy=HeightClosePolicy(max_height=self.close_height),
            max_consecutive_rejects=self.max_consecutive_rejects,
            enable_stability=False,
            allow_all_orientations=(self.num_orientations == 6),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Box generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_boxes(
    n: int,
    size_range: Tuple[float, float] = (100.0, 600.0),
    weight_range: Tuple[float, float] = (1.0, 30.0),
    rng: Optional[np.random.Generator] = None,
) -> List[Box]:
    """
    Generate random boxes for training episodes.

    Box dimensions are uniformly sampled from [min_dim, max_dim].
    Dimensions are rounded to the nearest 10mm for realism.

    Args:
        n:            Number of boxes.
        size_range:   (min_dim, max_dim) in mm.
        weight_range: (min_weight, max_weight) in kg.
        rng:          Numpy random generator (for reproducibility).

    Returns:
        List of n Box objects.
    """
    if rng is None:
        rng = np.random.default_rng()

    boxes = []
    lo, hi = size_range
    wlo, whi = weight_range
    for i in range(n):
        l = float(rng.integers(int(lo / 10), int(hi / 10) + 1) * 10)
        w = float(rng.integers(int(lo / 10), int(hi / 10) + 1) * 10)
        h = float(rng.integers(int(lo / 10), int(hi / 10) + 1) * 10)
        weight = float(rng.uniform(wlo, whi))
        boxes.append(Box(id=i, length=l, width=w, height=h, weight=weight))
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium environment
# ─────────────────────────────────────────────────────────────────────────────

class BinPackingEnv(gym.Env):
    """
    Gymnasium environment for online 3D bin packing.

    Wraps PackingSession with step-mode API to provide a standard RL
    interface.  Supports action masking for invalid placements.

    Action space:
        Discrete — flattened index encoding:
          (box_idx, bin_idx, x_grid, y_grid, orient_idx)
        Plus one "skip" action (advance conveyor).

    Observation space:
        Dict with heightmaps, box features, bin stats, and action mask.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        box_generator: Optional[Callable] = None,
    ):
        super().__init__()
        self.env_config = config or EnvConfig()
        self._box_generator = box_generator
        self._rng = np.random.default_rng(self.env_config.seed)
        self._reward_shaper = RewardShaper(self.env_config.reward_config)

        cfg = self.env_config
        bc = cfg.bin_config

        # ── Observation space ──
        self.observation_space = spaces.Dict({
            # Normalised heightmaps for each bin
            "heightmaps": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.num_bins, bc.grid_l, bc.grid_w),
                dtype=np.float32,
            ),
            # Grippable box features: (l, w, h, vol) normalised
            "box_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.pick_window, 4),
                dtype=np.float32,
            ),
            # All visible box features
            "buffer_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.buffer_size, 4),
                dtype=np.float32,
            ),
            # Per-bin statistics: (fill_rate, max_h_norm, roughness_norm, n_boxes_norm)
            "bin_stats": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.num_bins, 4),
                dtype=np.float32,
            ),
            # Action mask: 1 = valid, 0 = invalid
            "action_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(cfg.total_actions,),
                dtype=np.float32,
            ),
        })

        # ── Action space ──
        self.action_space = spaces.Discrete(cfg.total_actions)

        # ── Internal state ──
        self._session: Optional[PackingSession] = None
        self._obs: Optional[StepObservation] = None
        self._prev_fill_rates: Optional[List[float]] = None
        self._episode_rewards: List[float] = []
        self._episode_placements: int = 0

    # ── Gym API ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment: generate new boxes, initialise session.

        Args:
            seed:    Optional RNG seed override.
            options: Optional dict with "boxes" key for custom box list.

        Returns:
            (observation, info) tuple.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Generate boxes
        if options and "boxes" in options:
            boxes = options["boxes"]
        elif self._box_generator is not None:
            boxes = self._box_generator(self._rng)
        else:
            boxes = generate_random_boxes(
                n=self.env_config.num_boxes_per_episode,
                size_range=self.env_config.box_size_range,
                weight_range=self.env_config.box_weight_range,
                rng=self._rng,
            )

        # Create and reset session
        session_cfg = self.env_config.get_session_config()
        self._session = PackingSession(session_cfg)
        self._obs = self._session.reset(boxes, strategy_name="rl_agent")
        self._prev_fill_rates = [0.0] * self.env_config.num_bins
        self._episode_rewards = []
        self._episode_placements = 0

        obs = self._build_observation()
        info = {"step": 0, "pallets_closed": 0}
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one action.

        Args:
            action: Flattened action index.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        cfg = self.env_config
        session = self._session
        skip_action = cfg.total_actions - 1

        # Decode action
        if action == skip_action:
            # Skip: advance conveyor
            rejected = session.advance_conveyor()
            reward = self._reward_shaper.skip_penalty()
            placed = False
            pallet_closed = False
        else:
            # Decode (box_idx, bin_idx, x_grid, y_grid, orient_idx)
            box_idx, bin_idx, gx, gy, orient_idx = self._decode_action(action)

            grippable = self._obs.grippable
            if box_idx >= len(grippable):
                # Invalid box index — treat as skip
                session.advance_conveyor()
                reward = self._reward_shaper.invalid_action_penalty()
                placed = False
                pallet_closed = False
            else:
                box = grippable[box_idx]
                x = gx * cfg.action_grid_step
                y = gy * cfg.action_grid_step

                step_result = session.step(box.id, bin_idx, x, y, orient_idx)
                placed = step_result.placed
                pallet_closed = step_result.pallet_closed

                if placed:
                    self._episode_placements += 1
                    # Get current fill rates
                    new_obs = session.observe()
                    new_fills = [bs.get_fill_rate() for bs in new_obs.bin_states]

                    reward = self._reward_shaper.placement_reward(
                        box=box,
                        bin_state=new_obs.bin_states[bin_idx],
                        bin_config=cfg.bin_config,
                        fill_delta=new_fills[bin_idx] - self._prev_fill_rates[bin_idx],
                        pallet_closed=pallet_closed,
                        closed_fill=step_result.closed_pallet_result.fill_rate if pallet_closed else 0.0,
                    )
                    self._prev_fill_rates = new_fills
                else:
                    reward = self._reward_shaper.rejection_penalty()

        # Get new observation
        self._obs = session.observe()
        self._episode_rewards.append(reward)

        terminated = self._obs.done
        truncated = False

        # Terminal reward bonus
        if terminated:
            result = session.result()
            reward += self._reward_shaper.terminal_reward(
                avg_fill=result.avg_closed_fill,
                pallets_closed=result.pallets_closed,
                placement_rate=result.placement_rate,
            )

        info = {
            "step": self._obs.step_num,
            "placed": placed if action != skip_action else False,
            "episode_placements": self._episode_placements,
            "episode_return": sum(self._episode_rewards),
        }

        if terminated:
            result = session.result()
            info["final_avg_fill"] = result.avg_closed_fill
            info["pallets_closed"] = result.pallets_closed
            info["placement_rate"] = result.placement_rate
            info["total_placed"] = result.total_placed
            info["total_rejected"] = result.total_rejected

        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    # ── Observation building ─────────────────────────────────────────────

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build the observation dict from current session state."""
        cfg = self.env_config
        bc = cfg.bin_config
        obs = self._obs

        # Heightmaps: normalised by bin height
        heightmaps = np.zeros(
            (cfg.num_bins, bc.grid_l, bc.grid_w), dtype=np.float32,
        )
        for i, bs in enumerate(obs.bin_states):
            heightmaps[i] = bs.heightmap.astype(np.float32) / bc.height

        # Box features: (l, w, h, vol) normalised
        max_dim = max(bc.length, bc.width, bc.height)
        max_vol = bc.volume

        box_features = np.zeros((cfg.pick_window, 4), dtype=np.float32)
        for i, box in enumerate(obs.grippable[:cfg.pick_window]):
            box_features[i] = [
                box.length / max_dim,
                box.width / max_dim,
                box.height / max_dim,
                box.volume / max_vol,
            ]

        buffer_features = np.zeros((cfg.buffer_size, 4), dtype=np.float32)
        for i, box in enumerate(obs.buffer_view[:cfg.buffer_size]):
            buffer_features[i] = [
                box.length / max_dim,
                box.width / max_dim,
                box.height / max_dim,
                box.volume / max_vol,
            ]

        # Bin stats
        bin_stats = np.zeros((cfg.num_bins, 4), dtype=np.float32)
        for i, bs in enumerate(obs.bin_states):
            bin_stats[i] = [
                bs.get_fill_rate(),
                bs.get_max_height() / bc.height,
                min(bs.get_surface_roughness() / 100.0, 1.0),
                min(len(bs.placed_boxes) / 50.0, 1.0),  # normalise by ~max boxes
            ]

        # Action mask
        action_mask = self._compute_action_mask()

        return {
            "heightmaps": heightmaps,
            "box_features": box_features,
            "buffer_features": buffer_features,
            "bin_stats": bin_stats,
            "action_mask": action_mask,
        }

    def _compute_action_mask(self) -> np.ndarray:
        """
        Compute validity mask for all actions.

        For each (box, bin, x, y, orient) combination, checks whether the
        placement would be physically valid (within bounds, not floating).

        Uses vectorised numpy operations for the inner loops to avoid the
        ~1.2s per call cost of pure Python loops over 120x80 grids.
        Only falls back to per-cell get_height_at for positions that pass
        the bounds check.

        The skip action is always valid.
        """
        cfg = self.env_config
        bc = cfg.bin_config
        obs = self._obs
        mask = np.zeros(cfg.total_actions, dtype=np.float32)

        grippable = obs.grippable[:cfg.pick_window]
        if not grippable or obs.done:
            mask[-1] = 1.0  # Only skip is valid
            return mask

        res = bc.resolution
        grid_l = cfg.action_grid_l
        grid_w = cfg.action_grid_w
        step = cfg.action_grid_step

        # Pre-compute grid coordinates (mm) — reused across boxes/bins
        gx_coords = np.arange(grid_l, dtype=np.float64) * step
        gy_coords = np.arange(grid_w, dtype=np.float64) * step

        for box_idx, box in enumerate(grippable):
            if cfg.num_orientations == 6:
                orients = Orientation.get_all(box.length, box.width, box.height)
            else:
                orients = Orientation.get_flat(box.length, box.width, box.height)

            for orient_idx, (ol, ow, oh) in enumerate(orients):
                if orient_idx >= cfg.num_orientations:
                    break

                # Vectorised bounds check: which gx/gy are in bounds?
                valid_gx = np.where(gx_coords + ol <= bc.length + 0.01)[0]
                valid_gy = np.where(gy_coords + ow <= bc.width + 0.01)[0]
                if len(valid_gx) == 0 or len(valid_gy) == 0:
                    continue

                # Grid cells for the oriented box (in heightmap units)
                ol_cells = max(1, int(round(ol / res)))
                ow_cells = max(1, int(round(ow / res)))

                for bin_idx, bs in enumerate(obs.bin_states):
                    hm = bs.heightmap  # shape (grid_l_full, grid_w_full)

                    for gx in valid_gx:
                        x = gx_coords[gx]
                        # Heightmap slice for this x-row
                        hm_x0 = int(round(x / res))
                        hm_x1 = min(hm_x0 + ol_cells, hm.shape[0])

                        for gy in valid_gy:
                            y = gy_coords[gy]
                            hm_y0 = int(round(y / res))
                            hm_y1 = min(hm_y0 + ow_cells, hm.shape[1])

                            # Max height in the footprint (vectorised slice)
                            region = hm[hm_x0:hm_x1, hm_y0:hm_y1]
                            z = float(region.max()) if region.size > 0 else 0.0

                            # Height cap check
                            if z + oh > bc.height:
                                continue

                            # Support check (only for non-floor placements)
                            if z > 0.01:
                                # Fraction of cells at height z (±tolerance)
                                tol = 5.0
                                support_cells = np.sum(
                                    np.abs(region - z) <= tol
                                )
                                total_cells = region.size
                                if total_cells > 0 and support_cells / total_cells < 0.30:
                                    continue

                            # Valid action
                            action_idx = self._encode_action(
                                box_idx, bin_idx, gx, gy, orient_idx,
                            )
                            mask[action_idx] = 1.0

        # Skip is always valid
        mask[-1] = 1.0
        return mask

    # ── Action encoding/decoding ─────────────────────────────────────────

    def _encode_action(
        self,
        box_idx: int,
        bin_idx: int,
        gx: int,
        gy: int,
        orient_idx: int,
    ) -> int:
        """Encode (box, bin, x, y, orient) → flat action index."""
        cfg = self.env_config
        # box_idx * (bins * grid_l * grid_w * orients) + ...
        return (
            box_idx * (cfg.num_bins * cfg.action_grid_l * cfg.action_grid_w * cfg.num_orientations)
            + bin_idx * (cfg.action_grid_l * cfg.action_grid_w * cfg.num_orientations)
            + gx * (cfg.action_grid_w * cfg.num_orientations)
            + gy * cfg.num_orientations
            + orient_idx
        )

    def _decode_action(self, action: int) -> Tuple[int, int, int, int, int]:
        """Decode flat action index → (box_idx, bin_idx, gx, gy, orient_idx)."""
        cfg = self.env_config
        n_orient = cfg.num_orientations
        n_gy = cfg.action_grid_w
        n_gx = cfg.action_grid_l
        n_bin = cfg.num_bins

        orient_idx = action % n_orient
        action //= n_orient
        gy = action % n_gy
        action //= n_gy
        gx = action % n_gx
        action //= n_gx
        bin_idx = action % n_bin
        action //= n_bin
        box_idx = action

        return box_idx, bin_idx, gx, gy, orient_idx

    # ── Render (optional) ────────────────────────────────────────────────

    def render(self):
        """Print current state to console."""
        if self._obs is None:
            return
        obs = self._obs
        print(f"Step {obs.step_num} | Grippable: {len(obs.grippable)} | "
              f"Stream: {obs.stream_remaining} | Done: {obs.done}")
        for i, bs in enumerate(obs.bin_states):
            print(f"  Bin {i}: fill={bs.get_fill_rate():.1%} "
                  f"height={bs.get_max_height():.0f}mm "
                  f"boxes={len(bs.placed_boxes)}")


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(
    config: Optional[EnvConfig] = None,
    num_envs: int = 1,
    seed: int = 42,
    box_generator: Optional[Callable] = None,
) -> gym.Env:
    """
    Create vectorized BinPackingEnv instances for parallel training.

    On HPC, use num_envs = number of CPU cores for maximum throughput.

    Args:
        config:        Environment configuration (defaults to Botko BV).
        num_envs:      Number of parallel environments.
        seed:          Base seed (each env gets seed + i).
        box_generator: Optional custom box generator.

    Returns:
        Vectorized Gymnasium environment.
    """
    if config is None:
        config = EnvConfig()

    def _make_single(idx: int):
        def _init():
            cfg = copy.deepcopy(config)
            cfg.seed = seed + idx
            env = BinPackingEnv(config=cfg, box_generator=box_generator)
            return env
        return _init

    if num_envs == 1:
        return _make_single(0)()

    try:
        from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
        if num_envs <= 8:
            return SyncVectorEnv([_make_single(i) for i in range(num_envs)])
        else:
            return AsyncVectorEnv([_make_single(i) for i in range(num_envs)])
    except ImportError:
        # Fallback: return single env
        return _make_single(0)()
