"""
Training script for the RL Hybrid Hyper-Heuristic.

Two training modes:

Mode 1: Tabular Q-learning (fast, ~1 hour)
    python train.py --mode tabular --episodes 10000

    Uses discretised state space and Q-table updates.  Simple, fast,
    and convergence-guaranteed with sufficient exploration.

Mode 2: DQN selector (better, ~4-8 hours)
    python train.py --mode dqn --episodes 50000

    Uses a small neural network with experience replay and target network.
    Handles continuous state space and generalises better.

Training loop overview:
    For each episode:
        1. Generate random boxes
        2. Create PackingSession with Botko BV config
        3. For each box on the conveyor:
            a. Extract state features
            b. Agent selects heuristic (epsilon-greedy)
            c. Call selected heuristic's decide_placement()
            d. Step the session with the heuristic's decision
            e. Compute reward based on placement quality
            f. Store transition, update agent
        4. Log episode metrics

Reward design:
    - Successful placement:  volume_ratio * 10 + fill_delta * 5
    - Heuristic fails:       -0.5 (selected heuristic returned None)
    - Skip action:           -0.3
    - Diversity bonus:       +0.1 when switching heuristics appropriately
    - Terminal bonus:         avg_closed_fill * 10

Key insight: The agent does NOT learn placement positions.  It learns
WHICH heuristic to call.  The heuristic handles all spatial reasoning.
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import random
from typing import List, Optional, Dict, Any

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, BinConfig, ExperimentConfig, PlacementDecision
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy
from strategies.base_strategy import get_strategy
from strategies.rl_common.logger import TrainingLogger
from strategies.rl_common.environment import generate_random_boxes

from strategies.rl_hybrid_hh.config import HHConfig
from strategies.rl_hybrid_hh.state_features import (
    FeatureTracker,
    extract_state_features,
    discretise_state,
)
from strategies.rl_hybrid_hh.network import TabularQLearner, ReplayBuffer

# DQN imports (optional)
try:
    from strategies.rl_hybrid_hh.network import DQNTrainer, HAS_TORCH
except ImportError:
    HAS_TORCH = False
    DQNTrainer = None


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    config: HHConfig,
    action: int,
    placed: bool,
    box: Optional[Box],
    fill_before: float,
    fill_after: float,
    bin_config: BinConfig,
    prev_action: int,
    pallet_closed: bool,
    closed_fill: float,
) -> float:
    """
    Compute the step reward for a heuristic selection.

    The reward is designed to teach the agent WHICH heuristic works best
    in which situation, not how to place boxes.

    Components:
        1. Volume ratio reward: Placed box volume / bin volume * weight
        2. Fill delta reward: Improvement in fill rate * weight
        3. Failure penalty: If heuristic returned None
        4. Skip penalty: If SKIP action was chosen
        5. Diversity bonus: If switched heuristic and placement succeeded

    Args:
        config:        HHConfig with reward weights.
        action:        The heuristic action index taken.
        placed:        Whether placement was successful.
        box:           The box that was attempted.
        fill_before:   Fill rate before the step.
        fill_after:    Fill rate after the step.
        bin_config:    Bin configuration.
        prev_action:   Previous heuristic action (for diversity).
        pallet_closed: Whether a pallet was closed.
        closed_fill:   Fill rate of the closed pallet.

    Returns:
        Scalar reward.
    """
    is_skip = config.include_skip and action == config.num_actions - 1

    if is_skip:
        return config.reward_skip_penalty

    if not placed:
        return config.reward_failure_penalty

    reward = 0.0

    # Volume ratio reward
    if box is not None:
        vol_ratio = box.volume / bin_config.volume if bin_config.volume > 0 else 0.0
        reward += config.reward_volume_weight * vol_ratio

    # Fill delta reward
    fill_delta = fill_after - fill_before
    reward += config.reward_fill_delta_weight * fill_delta

    # Diversity bonus: switching heuristic and succeeding
    if prev_action >= 0 and action != prev_action:
        reward += config.reward_diversity_bonus

    # Pallet close bonus (scaled by fill quality)
    if pallet_closed:
        reward += config.reward_terminal_weight * closed_fill

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_training_episode(
    session: PackingSession,
    portfolio: Dict[int, Any],
    config: HHConfig,
    bin_config: BinConfig,
    boxes: List[Box],
    agent,
    mode: str,
    epsilon: float,
    tracker: FeatureTracker,
    exp_config: ExperimentConfig,
    store_transitions: bool = True,
) -> Dict[str, Any]:
    """
    Run a single training episode.

    The loop:
        1. Reset session with new boxes
        2. For each step:
            a. Get observation (grippable boxes, bin states)
            b. For each grippable box, try to find a heuristic that places it
            c. Extract features, select heuristic, call it
            d. Step session with the result
            e. Compute reward, store transition

    Args:
        session:           Configured PackingSession.
        portfolio:         Dict mapping action -> strategy instance.
        config:            HHConfig.
        bin_config:        BinConfig.
        boxes:             List of boxes for this episode.
        agent:             TabularQLearner or DQNTrainer.
        mode:              "tabular" or "dqn".
        epsilon:           Exploration rate.
        tracker:           FeatureTracker (reset inside).
        exp_config:        ExperimentConfig for strategy initialisation.
        store_transitions: Whether to store transitions (False for eval).

    Returns:
        Dict with episode metrics.
    """
    # Reset
    obs = session.reset(boxes, strategy_name="rl_hybrid_hh")
    tracker.reset(total_boxes=len(boxes))

    # Initialise portfolio strategies
    for strat in portfolio.values():
        strat.on_episode_start(exp_config)

    episode_reward = 0.0
    prev_action = -1
    step_count = 0
    heuristic_choices = []
    losses = []

    while not obs.done:
        grippable = obs.grippable
        if not grippable:
            break

        # Try the first grippable box (FIFO)
        box = grippable[0]
        bin_states = obs.bin_states

        # Extract state features
        state = extract_state_features(
            box=box,
            bin_states=bin_states,
            bin_config=bin_config,
            grippable=grippable,
            buffer_view=obs.buffer_view,
            tracker=tracker,
            config=config,
        )

        # Select action (heuristic)
        if mode == "tabular":
            state_idx = discretise_state(state, config)
            action = agent.select_action(state_idx, epsilon)
        else:  # dqn
            action = agent.select_action(state, epsilon)

        heuristic_choices.append(action)

        # Get fill rates before
        fill_before = max(bs.get_fill_rate() for bs in bin_states)

        # Handle SKIP action
        is_skip = config.include_skip and action == config.num_actions - 1
        placed = False
        pallet_closed = False
        closed_fill = 0.0

        if is_skip:
            session.advance_conveyor()
            tracker.record_choice(action, False)
        else:
            # Call the selected heuristic
            decision = None
            if action in portfolio:
                # Try placing on each bin, pick the best
                best_decision = None
                best_bin_idx = 0
                best_fill = -1.0

                for bin_idx, bs in enumerate(bin_states):
                    dec = portfolio[action].decide_placement(box, bs)
                    if dec is not None:
                        # Prefer the bin with higher fill (focus_fill logic)
                        fill = bs.get_fill_rate()
                        if fill > best_fill:
                            best_fill = fill
                            best_decision = dec
                            best_bin_idx = bin_idx

                if best_decision is not None:
                    step_result = session.step(
                        box.id, best_bin_idx,
                        best_decision.x, best_decision.y,
                        best_decision.orientation_idx,
                    )
                    placed = step_result.placed
                    pallet_closed = step_result.pallet_closed
                    if pallet_closed and step_result.closed_pallet_result:
                        closed_fill = step_result.closed_pallet_result.fill_rate
                else:
                    # Heuristic failed for all bins -- advance conveyor
                    session.advance_conveyor()

            else:
                # Invalid action (heuristic not in portfolio)
                session.advance_conveyor()

            tracker.record_choice(action, placed)
            if pallet_closed:
                tracker.record_pallet_close()

        # Get new observation
        obs = session.observe()

        # Compute fill after
        fill_after = max(
            (bs.get_fill_rate() for bs in obs.bin_states), default=0.0
        )

        # Compute reward
        reward = compute_reward(
            config=config,
            action=action,
            placed=placed,
            box=box,
            fill_before=fill_before,
            fill_after=fill_after,
            bin_config=bin_config,
            prev_action=prev_action,
            pallet_closed=pallet_closed,
            closed_fill=closed_fill,
        )

        # Terminal bonus
        if obs.done:
            result = session.result()
            reward += config.reward_terminal_weight * result.avg_closed_fill

        episode_reward += reward

        # Extract next state
        if not obs.done and obs.grippable:
            next_box = obs.grippable[0]
            next_state = extract_state_features(
                box=next_box,
                bin_states=obs.bin_states,
                bin_config=bin_config,
                grippable=obs.grippable,
                buffer_view=obs.buffer_view,
                tracker=tracker,
                config=config,
            )
        else:
            next_state = np.zeros(config.state_dim, dtype=np.float32)

        # Store transition and update
        if store_transitions:
            if mode == "tabular":
                next_state_idx = discretise_state(next_state, config)
                td_error = agent.update(
                    state_idx, action, reward, next_state_idx, obs.done,
                )
            else:  # dqn
                agent.store_transition(state, action, reward, next_state, obs.done)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

        prev_action = action
        step_count += 1

    # End portfolio strategies
    for strat in portfolio.values():
        strat.on_episode_end({})

    # Compute results
    result = session.result()

    # Action distribution
    action_counts = np.bincount(
        heuristic_choices, minlength=config.num_actions,
    )

    return {
        "reward": episode_reward,
        "fill": result.avg_closed_fill,
        "pallets_closed": result.pallets_closed,
        "total_placed": result.total_placed,
        "total_rejected": result.total_rejected,
        "placement_rate": result.placement_rate,
        "steps": step_count,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "action_counts": action_counts.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    session: PackingSession,
    portfolio: Dict[int, Any],
    config: HHConfig,
    bin_config: BinConfig,
    agent,
    mode: str,
    exp_config: ExperimentConfig,
    num_episodes: int = 5,
    seed: int = 9999,
) -> Dict[str, float]:
    """Run evaluation episodes with greedy policy (epsilon=0)."""
    rng = np.random.default_rng(seed)
    tracker = FeatureTracker(config)
    fills = []
    rewards = []

    for ep in range(num_episodes):
        boxes = generate_random_boxes(
            n=config.num_boxes_per_episode,
            size_range=config.box_size_range,
            weight_range=config.box_weight_range,
            rng=rng,
        )
        result = run_training_episode(
            session=session,
            portfolio=portfolio,
            config=config,
            bin_config=bin_config,
            boxes=boxes,
            agent=agent,
            mode=mode,
            epsilon=0.0,
            tracker=tracker,
            exp_config=exp_config,
            store_transitions=False,
        )
        fills.append(result["fill"])
        rewards.append(result["reward"])

    return {
        "eval_fill_mean": float(np.mean(fills)),
        "eval_fill_std": float(np.std(fills)),
        "eval_reward_mean": float(np.mean(rewards)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(config: HHConfig, mode: str = "tabular") -> str:
    """
    Run the full training pipeline.

    Args:
        config: HHConfig with all hyperparameters.
        mode:   "tabular" or "dqn".

    Returns:
        Path to the best model checkpoint.
    """
    print(f"\n{'='*70}")
    print(f"  RL Hybrid Hyper-Heuristic Training")
    print(f"  Mode: {mode.upper()}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Heuristic portfolio: {config.heuristic_names}")
    print(f"  State dim: {config.state_dim}")
    print(f"  Actions: {config.num_actions}")
    if mode == "tabular":
        print(f"  Tabular states: {config.tabular_state_size}")
    print(f"{'='*70}\n")

    # ── Setup ─────────────────────────────────────────────────────────────

    # Physical setup (Botko BV)
    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )
    session_config = SessionConfig(
        bin_config=bin_config,
        num_bins=config.num_bins,
        buffer_size=config.buffer_size,
        pick_window=config.pick_window,
        close_policy=HeightClosePolicy(max_height=config.close_height),
        max_consecutive_rejects=10,
    )
    session = PackingSession(session_config)
    exp_config = session_config.to_experiment_config("rl_hybrid_hh")

    # Portfolio: instantiate all heuristic strategies
    portfolio: Dict[int, Any] = {}
    for idx, name in enumerate(config.heuristic_names):
        try:
            portfolio[idx] = get_strategy(name)
        except ValueError as e:
            print(f"WARNING: {e}")

    print(f"Portfolio loaded: {len(portfolio)} heuristics")
    for idx, strat in portfolio.items():
        print(f"  [{idx}] {strat.name}")
    if config.include_skip:
        print(f"  [{config.num_actions - 1}] SKIP (advance conveyor)")
    print()

    # Agent
    if mode == "tabular":
        agent = TabularQLearner(config)
        print(f"Tabular Q-table: {agent.num_states} states x {agent.num_actions} actions")
    elif mode == "dqn":
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DQN mode. pip install torch")
        agent = DQNTrainer(config)
        print(f"DQN network: {agent.online_net.count_parameters()} parameters")
        print(f"Device: {agent.device}")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'tabular' or 'dqn'.")

    # Logger
    output_dir = os.path.join(_WORKFLOW_ROOT, config.output_dir)
    log_dir = os.path.join(output_dir, "logs", mode)
    logger = TrainingLogger(log_dir=log_dir, strategy_name=f"rl_hybrid_hh_{mode}")
    logger.log_config(config.to_dict())

    # Tracker
    tracker = FeatureTracker(config)

    # RNG for reproducibility
    rng = np.random.default_rng(42)

    # ── Training Loop ─────────────────────────────────────────────────────

    best_fill = 0.0
    best_model_path = ""
    t_start = time.time()

    for episode in range(config.num_episodes):
        # Epsilon decay
        epsilon = config.get_epsilon(episode)

        # Generate random boxes for this episode
        boxes = generate_random_boxes(
            n=config.num_boxes_per_episode,
            size_range=config.box_size_range,
            weight_range=config.box_weight_range,
            rng=rng,
        )

        # Run episode
        result = run_training_episode(
            session=session,
            portfolio=portfolio,
            config=config,
            bin_config=bin_config,
            boxes=boxes,
            agent=agent,
            mode=mode,
            epsilon=epsilon,
            tracker=tracker,
            exp_config=exp_config,
        )

        # Target network sync (DQN only)
        if mode == "dqn" and (episode + 1) % config.target_update_freq == 0:
            agent.sync_target()

        # Logging
        logger.log_episode(
            episode=episode,
            reward=result["reward"],
            fill=result["fill"],
            loss=result["loss"],
            epsilon=epsilon,
            pallets_closed=result["pallets_closed"],
            placement_rate=result["placement_rate"],
        )

        # Console progress
        if (episode + 1) % config.log_interval == 0:
            logger.print_progress(
                episode + 1, config.num_episodes,
                reward=result["reward"],
                fill=result["fill"],
                epsilon=epsilon,
                placed=result["total_placed"],
            )

        # Evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_metrics = evaluate(
                session=session,
                portfolio=portfolio,
                config=config,
                bin_config=bin_config,
                agent=agent if mode == "tabular" else agent,
                mode=mode,
                exp_config=exp_config,
                num_episodes=config.eval_episodes,
            )
            print(f"  EVAL: fill={eval_metrics['eval_fill_mean']:.4f} "
                  f"(+/- {eval_metrics['eval_fill_std']:.4f})")

            # Save best model
            if eval_metrics["eval_fill_mean"] > best_fill:
                best_fill = eval_metrics["eval_fill_mean"]
                if mode == "tabular":
                    best_model_path = os.path.join(output_dir, "best_model.npz")
                    agent.save(best_model_path)
                else:
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    agent.save(best_model_path)
                print(f"  NEW BEST: fill={best_fill:.4f} -> {best_model_path}")

        # Checkpoint
        if (episode + 1) % config.checkpoint_interval == 0:
            if mode == "tabular":
                ckpt_path = os.path.join(
                    output_dir, "checkpoints", f"tabular_ep{episode+1}.npz",
                )
                agent.save(ckpt_path)
            else:
                ckpt_path = os.path.join(
                    output_dir, "checkpoints", f"dqn_ep{episode+1}.pt",
                )
                agent.save(ckpt_path)

    # ── Final ─────────────────────────────────────────────────────────────

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Training Complete")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best eval fill: {best_fill:.4f}")
    print(f"  Best model: {best_model_path}")

    if mode == "tabular":
        print(f"\n{agent.summary()}")

    # Save final model
    if mode == "tabular":
        final_path = os.path.join(output_dir, "final_model.npz")
        agent.save(final_path)
    else:
        final_path = os.path.join(output_dir, "final_model.pt")
        agent.save(final_path)
    print(f"  Final model: {final_path}")

    # Generate training curves
    curves_path = logger.plot_training_curves(save=True, show=False)
    if curves_path:
        print(f"  Training curves: {curves_path}")

    logger.close()
    print(f"{'='*70}\n")

    return best_model_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the RL Hybrid Hyper-Heuristic for 3D bin packing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tabular Q-learning (fast, ~1 hour)
  python train.py --mode tabular --episodes 10000

  # DQN (better, ~4-8 hours)
  python train.py --mode dqn --episodes 50000

  # Quick test run
  python train.py --mode tabular --episodes 100 --log-interval 10
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="tabular",
        choices=["tabular", "dqn"],
        help="Training mode: tabular Q-learning or DQN (default: tabular)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of training episodes (default: 10000 tabular, 50000 dqn)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 0.1 tabular, 0.001 dqn)",
    )
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--eps-start", type=float, default=None,
        help="Starting epsilon for exploration (default: 1.0)",
    )
    parser.add_argument(
        "--eps-end", type=float, default=None,
        help="Final epsilon (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/rl_hybrid_hh)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=None,
        help="Console print interval (default: 100)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=None,
        help="Evaluation interval (default: 500)",
    )
    parser.add_argument(
        "--boxes-per-episode", type=int, default=None,
        help="Boxes per episode (default: 100)",
    )

    args = parser.parse_args()

    # Build config from args
    config = HHConfig()

    if args.mode == "tabular":
        config.lr = 0.1  # Higher LR for tabular
        config.num_episodes = 10_000
    else:
        config.lr = 0.001
        config.num_episodes = 50_000

    if args.episodes is not None:
        config.num_episodes = args.episodes
    if args.lr is not None:
        config.lr = args.lr
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.eps_start is not None:
        config.eps_start = args.eps_start
    if args.eps_end is not None:
        config.eps_end = args.eps_end
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.boxes_per_episode is not None:
        config.num_boxes_per_episode = args.boxes_per_episode

    train(config, mode=args.mode)


if __name__ == "__main__":
    main()
