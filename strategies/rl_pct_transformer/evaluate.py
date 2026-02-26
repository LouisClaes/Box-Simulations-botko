"""
Evaluation and benchmarking utilities for the PCT Transformer strategy.

Provides:
  - Single-episode evaluation with detailed per-pallet metrics
  - Batch evaluation across multiple seeds for statistical significance
  - Comparison against heuristic baselines
  - Dataset-based evaluation (use real box data instead of random)
  - Visualisation-ready output (JSON + summary tables)

CLI:
    python evaluate.py --checkpoint best.pt --episodes 50 --seed 42
    python evaluate.py --checkpoint best.pt --compare extreme_points,walle_scoring
    python evaluate.py --checkpoint best.pt --dataset path/to/boxes.json
"""

from __future__ import annotations

import sys
import os
import json
import time
import argparse
from typing import List, Optional, Dict, Any
from dataclasses import asdict

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

import torch

from config import Box, BinConfig, ExperimentConfig
from simulator.session import PackingSession, SessionConfig, SessionResult
from simulator.close_policy import HeightClosePolicy
from strategies.rl_common.obs_utils import encode_box_features

from strategies.rl_pct_transformer.config import PCTTransformerConfig
from strategies.rl_pct_transformer.network import PCTTransformerNet
from strategies.rl_pct_transformer.candidate_generator import CandidateGenerator
from strategies.rl_pct_transformer.train import PCTPackingEnv, pad_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    config: Optional[PCTTransformerConfig] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load a trained PCT Transformer model from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        config:          Override config (uses saved config if None).
        device:          Target device (auto-detected if None).

    Returns:
        (network, config, metadata) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint if not provided
    if config is None:
        saved_cfg = checkpoint.get('config', {})
        config = PCTTransformerConfig.from_dict(saved_cfg)

    network = PCTTransformerNet(config).to(device)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    metadata = {
        'episode': checkpoint.get('episode', -1),
        'best_fill': checkpoint.get('best_fill', -1),
        'checkpoint_path': checkpoint_path,
    }

    return network, config, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Single episode evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single_episode(
    network: PCTTransformerNet,
    config: PCTTransformerConfig,
    seed: int = 42,
    boxes: Optional[List[Box]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode and return detailed results.

    Args:
        network:  Trained model.
        config:   Configuration.
        seed:     Random seed for box generation.
        boxes:    Optional pre-defined box list (overrides random generation).
        verbose:  Print step-by-step progress.

    Returns:
        Dict with episode results, per-pallet stats, and step trace.
    """
    device = next(network.parameters()).device
    network.eval()

    env = PCTPackingEnv(config, seed=seed)

    if boxes is not None:
        # Override random box generation with provided boxes
        env.session = PackingSession(env.session_config)
        obs = env.session.reset(boxes, strategy_name="rl_pct_transformer")
        env._prev_fill_rates = [0.0] * config.num_bins
        env.done = False
        item_feat, candidates, info = env._observe(obs)
    else:
        item_feat, candidates, info = env.reset()

    total_reward = 0.0
    step_trace: List[Dict] = []
    step_count = 0
    t0 = time.perf_counter()

    while not env.done and candidates:
        # Prepare batch
        item_t = torch.from_numpy(item_feat).unsqueeze(0).to(device)
        cand_feat, cand_mask = pad_candidates(
            [candidates], config.candidate_input_dim, device,
        )

        with torch.no_grad():
            action, log_prob, _, value = network.get_action_and_value(
                item_t, cand_feat, cand_mask,
                deterministic=config.deterministic_inference,
            )

        action_idx = action.item()
        if action_idx >= len(candidates):
            action_idx = 0

        selected = candidates[action_idx]

        # Step
        item_feat, candidates, reward, done, step_info = env.step(action_idx)
        total_reward += reward

        # Record trace
        trace_entry = {
            "step": step_count,
            "box_id": env.current_box.id if env.current_box else -1,
            "bin_idx": selected.bin_idx,
            "x": selected.x,
            "y": selected.y,
            "z": selected.z,
            "orient": selected.orient_idx,
            "dims": (selected.oriented_l, selected.oriented_w, selected.oriented_h),
            "num_candidates": len(candidates) if candidates else 0,
            "reward": reward,
            "value": value.item(),
            "placed": step_info.get("placed", False),
        }
        step_trace.append(trace_entry)

        if verbose and step_count % 10 == 0:
            print(f"  Step {step_count}: placed={trace_entry['placed']}, "
                  f"reward={reward:.4f}, candidates={trace_entry['num_candidates']}")

        step_count += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Get session result
    result = env.get_session_result()

    episode_result = {
        "seed": seed,
        "total_reward": total_reward,
        "steps": step_count,
        "elapsed_ms": elapsed_ms,
        "ms_per_step": elapsed_ms / max(step_count, 1),
    }

    if result is not None:
        episode_result.update({
            "avg_closed_fill": result.avg_closed_fill,
            "avg_effective_fill": result.avg_closed_effective_fill,
            "pallets_closed": result.pallets_closed,
            "total_placed": result.total_placed,
            "total_rejected": result.total_rejected,
            "placement_rate": result.placement_rate,
            "ms_per_box": result.ms_per_box,
            "closed_pallets": [p.to_dict() for p in result.closed_pallets],
            "active_pallets": [p.to_dict() for p in result.active_pallets],
        })

    episode_result["step_trace"] = step_trace

    return episode_result


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    network: PCTTransformerNet,
    config: PCTTransformerConfig,
    num_episodes: int = 50,
    base_seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run multiple evaluation episodes and compute aggregate statistics.

    Args:
        network:      Trained model.
        config:       Configuration.
        num_episodes: Number of episodes to run.
        base_seed:    Base seed (each episode gets base_seed + i).
        verbose:      Print per-episode results.

    Returns:
        Dict with aggregate statistics and per-episode results.
    """
    episode_results = []
    fills = []
    placement_rates = []
    rewards = []

    for i in range(num_episodes):
        seed = base_seed + i
        result = evaluate_single_episode(
            network, config, seed=seed, verbose=False,
        )

        # Remove step trace for aggregate (too verbose)
        result_summary = {k: v for k, v in result.items() if k != "step_trace"}
        episode_results.append(result_summary)

        if "avg_closed_fill" in result:
            fills.append(result["avg_closed_fill"])
        if "placement_rate" in result:
            placement_rates.append(result["placement_rate"])
        rewards.append(result["total_reward"])

        if verbose:
            fill = result.get("avg_closed_fill", 0.0)
            placed = result.get("total_placed", 0)
            pr = result.get("placement_rate", 0.0)
            print(f"  Episode {i+1}/{num_episodes} (seed={seed}): "
                  f"fill={fill:.4f}, placed={placed}, pr={pr:.4f}")

    aggregate = {
        "num_episodes": num_episodes,
        "fill_mean": float(np.mean(fills)) if fills else 0.0,
        "fill_std": float(np.std(fills)) if fills else 0.0,
        "fill_min": float(np.min(fills)) if fills else 0.0,
        "fill_max": float(np.max(fills)) if fills else 0.0,
        "fill_median": float(np.median(fills)) if fills else 0.0,
        "placement_rate_mean": float(np.mean(placement_rates)) if placement_rates else 0.0,
        "placement_rate_std": float(np.std(placement_rates)) if placement_rates else 0.0,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
    }

    return {
        "aggregate": aggregate,
        "episodes": episode_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison with heuristic baselines
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_baselines(
    network: PCTTransformerNet,
    config: PCTTransformerConfig,
    baseline_names: List[str],
    num_episodes: int = 20,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare the RL policy against heuristic baselines.

    Runs both the RL policy and each baseline on the SAME box sequences
    for fair comparison.

    Args:
        network:        Trained model.
        config:         Configuration.
        baseline_names: List of strategy names to compare against.
        num_episodes:   Episodes per comparison.
        base_seed:      Base seed.

    Returns:
        Dict with per-strategy results and comparison table.
    """
    from strategies.base_strategy import get_strategy
    from simulator.session import FIFOBoxSelector, EmptiestFirst

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.bin_resolution,
    )
    session_config = SessionConfig(
        bin_config=bin_config,
        num_bins=config.num_bins,
        buffer_size=config.buffer_size,
        pick_window=config.pick_window,
        close_policy=HeightClosePolicy(max_height=config.close_height),
        max_consecutive_rejects=config.max_consecutive_rejects,
    )

    results = {}

    # Generate shared box sequences
    rng = np.random.default_rng(base_seed)
    all_box_sequences = []
    for ep in range(num_episodes):
        ep_rng = np.random.default_rng(base_seed + ep)
        boxes = []
        lo, hi = config.box_size_range
        wlo, whi = config.box_weight_range
        for i in range(config.num_boxes_per_episode):
            l = float(ep_rng.integers(int(lo/10), int(hi/10)+1) * 10)
            w = float(ep_rng.integers(int(lo/10), int(hi/10)+1) * 10)
            h = float(ep_rng.integers(int(lo/10), int(hi/10)+1) * 10)
            wt = float(ep_rng.uniform(wlo, whi))
            boxes.append(Box(id=i, length=l, width=w, height=h, weight=wt))
        all_box_sequences.append(boxes)

    # Evaluate RL policy
    print(f"\n  Evaluating: rl_pct_transformer ({num_episodes} episodes)...")
    rl_fills = []
    for ep, boxes in enumerate(all_box_sequences):
        ep_result = evaluate_single_episode(
            network, config, seed=base_seed + ep, boxes=boxes,
        )
        rl_fills.append(ep_result.get("avg_closed_fill", 0.0))

    results["rl_pct_transformer"] = {
        "fill_mean": float(np.mean(rl_fills)),
        "fill_std": float(np.std(rl_fills)),
        "fill_median": float(np.median(rl_fills)),
    }

    # Evaluate baselines
    for strat_name in baseline_names:
        print(f"  Evaluating: {strat_name} ({num_episodes} episodes)...")
        try:
            strategy = get_strategy(strat_name)
        except ValueError as e:
            print(f"    Skipping {strat_name}: {e}")
            continue

        baseline_fills = []
        for ep, boxes in enumerate(all_box_sequences):
            session = PackingSession(session_config)
            result = session.run(
                boxes, strategy,
                box_selector=FIFOBoxSelector(),
                bin_selector=EmptiestFirst(),
            )
            baseline_fills.append(result.avg_closed_fill)

        results[strat_name] = {
            "fill_mean": float(np.mean(baseline_fills)),
            "fill_std": float(np.std(baseline_fills)),
            "fill_median": float(np.median(baseline_fills)),
        }

    # Print comparison table
    print(f"\n{'Strategy':<30} {'Fill Mean':>10} {'Fill Std':>10} {'Fill Med':>10}")
    print("-" * 62)
    for name, stats in sorted(results.items(), key=lambda x: -x[1]['fill_mean']):
        print(f"{name:<30} {stats['fill_mean']:>10.4f} {stats['fill_std']:>10.4f} "
              f"{stats['fill_median']:>10.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate PCT Transformer RL strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode results")
    parser.add_argument("--compare", type=str, default=None,
                        help="Comma-separated baseline strategy names to compare")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use greedy action selection")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic action selection")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[PCT-Transformer Evaluation]")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print(f"  Episodes:   {args.episodes}")

    # Load model
    network, config, metadata = load_model(args.checkpoint, device=device)
    config.deterministic_inference = not args.stochastic

    print(f"  Model trained for {metadata['episode']} episodes")
    print(f"  Best training fill: {metadata['best_fill']:.4f}")
    print(f"  Parameters: {network.count_parameters():,}")

    # Run evaluation
    if args.compare:
        baseline_names = [s.strip() for s in args.compare.split(",")]
        results = compare_with_baselines(
            network, config,
            baseline_names=baseline_names,
            num_episodes=args.episodes,
            base_seed=args.seed,
        )
    else:
        results = evaluate_batch(
            network, config,
            num_episodes=args.episodes,
            base_seed=args.seed,
            verbose=args.verbose,
        )
        agg = results["aggregate"]
        print(f"\n  Results ({args.episodes} episodes):")
        print(f"    Fill:      {agg['fill_mean']:.4f} +/- {agg['fill_std']:.4f}")
        print(f"    Fill range: [{agg['fill_min']:.4f}, {agg['fill_max']:.4f}]")
        print(f"    Placement: {agg['placement_rate_mean']:.4f} +/- {agg['placement_rate_std']:.4f}")
        print(f"    Reward:    {agg['reward_mean']:.2f} +/- {agg['reward_std']:.2f}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
