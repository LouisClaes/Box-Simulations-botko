"""
Evaluation script for the MCTS-Guided Hierarchical Actor-Critic.

Evaluates a trained checkpoint across multiple seeds and box distributions,
producing publication-quality results for thesis comparison.

Evaluation modes:
  1. STANDARD: 100 episodes, random boxes, report mean/std fill rate
  2. COMPARISON: Run all strategies (heuristic + RL) on the SAME episodes
  3. ABLATION: Disable components (MCTS, world model, void) to measure impact
  4. DIFFICULTY: Vary box size distribution to test generalisation

Usage:
    # Standard evaluation
    python evaluate.py --checkpoint outputs/rl_mcts_hybrid/checkpoints/best_model.pt

    # Comparison with all strategies
    python evaluate.py --checkpoint best_model.pt --mode comparison

    # MCTS ablation (with vs without)
    python evaluate.py --checkpoint best_model.pt --mode ablation

    # Difficulty sweep
    python evaluate.py --checkpoint best_model.pt --mode difficulty
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_WORKFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _WORKFLOW_ROOT not in sys.path:
    sys.path.insert(0, _WORKFLOW_ROOT)

from config import Box, BinConfig, ExperimentConfig
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy

from strategies.rl_common.environment import generate_random_boxes
from strategies.rl_mcts_hybrid.config import MCTSHybridConfig


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_strategy_on_episodes(
    strategy_name: str,
    episodes: List[List[Box]],
    bin_config: BinConfig,
    config: MCTSHybridConfig,
) -> Dict[str, float]:
    """
    Evaluate a strategy on a fixed set of episodes.

    Returns mean/std fill rate and timing.
    """
    from strategies.base_strategy import get_strategy, STRATEGY_REGISTRY

    if strategy_name not in STRATEGY_REGISTRY:
        return {'mean_fill': 0.0, 'std_fill': 0.0, 'error': f'not found: {strategy_name}'}

    fills = []
    times = []

    for ep_idx, boxes in enumerate(episodes):
        strategy = get_strategy(strategy_name)

        session_cfg = SessionConfig(
            bin_config=bin_config,
            num_bins=config.num_bins,
            buffer_size=config.buffer_size,
            pick_window=config.pick_window,
            close_policy=HeightClosePolicy(config.close_height),
        )

        session = PackingSession(session_cfg)

        t0 = time.time()
        try:
            result = session.run(boxes, strategy=strategy)
            fill = result.avg_closed_fill if result.pallets_closed > 0 else 0.0
        except Exception as e:
            fill = 0.0
            print(f"  [{strategy_name}] Episode {ep_idx}: error: {e}")
        t1 = time.time()

        fills.append(fill)
        times.append(t1 - t0)

    return {
        'mean_fill': float(np.mean(fills)),
        'std_fill': float(np.std(fills)),
        'max_fill': float(np.max(fills)) if fills else 0.0,
        'min_fill': float(np.min(fills)) if fills else 0.0,
        'mean_time': float(np.mean(times)),
        'episodes': len(episodes),
    }


def evaluate_rl_mcts_on_episodes(
    episodes: List[List[Box]],
    bin_config: BinConfig,
    config: MCTSHybridConfig,
    checkpoint_path: str,
    use_mcts: bool = True,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the RL MCTS Hybrid strategy on fixed episodes.
    """
    from strategies.rl_mcts_hybrid.strategy import RLMCTSHybridStrategy

    fills = []
    times = []

    for ep_idx, boxes in enumerate(episodes):
        strategy = RLMCTSHybridStrategy(
            config=config,
            checkpoint_path=checkpoint_path,
            use_mcts=use_mcts,
            deterministic=deterministic,
        )

        session_cfg = SessionConfig(
            bin_config=bin_config,
            num_bins=config.num_bins,
            buffer_size=config.buffer_size,
            pick_window=config.pick_window,
            close_policy=HeightClosePolicy(config.close_height),
        )

        session = PackingSession(session_cfg)

        t0 = time.time()
        try:
            result = session.run(boxes, strategy=strategy)
            fill = result.avg_closed_fill if result.pallets_closed > 0 else 0.0
        except Exception as e:
            fill = 0.0
            print(f"  [rl_mcts_hybrid] Episode {ep_idx}: error: {e}")
        t1 = time.time()

        fills.append(fill)
        times.append(t1 - t0)

    return {
        'mean_fill': float(np.mean(fills)),
        'std_fill': float(np.std(fills)),
        'max_fill': float(np.max(fills)) if fills else 0.0,
        'min_fill': float(np.min(fills)) if fills else 0.0,
        'mean_time': float(np.mean(times)),
        'episodes': len(episodes),
        'use_mcts': use_mcts,
    }


# ---------------------------------------------------------------------------
# Generate fixed evaluation episodes
# ---------------------------------------------------------------------------

def generate_eval_episodes(
    num_episodes: int,
    num_boxes: int,
    size_range: Tuple[float, float],
    weight_range: Tuple[float, float],
    seed: int,
) -> List[List[Box]]:
    """Generate deterministic evaluation episodes."""
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(num_episodes):
        boxes = generate_random_boxes(
            n=num_boxes,
            size_range=size_range,
            weight_range=weight_range,
            rng=rng,
        )
        episodes.append(boxes)
    return episodes


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def standard_eval(
    checkpoint_path: str,
    config: MCTSHybridConfig,
    num_episodes: int = 100,
) -> Dict[str, float]:
    """Standard evaluation on random episodes."""
    print("\n=== Standard Evaluation ===")

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    episodes = generate_eval_episodes(
        num_episodes=num_episodes,
        num_boxes=config.num_boxes_per_episode,
        size_range=config.box_size_range,
        weight_range=config.box_weight_range,
        seed=42,
    )

    print(f"  Evaluating {num_episodes} episodes, {config.num_boxes_per_episode} boxes each...")

    # With MCTS
    results_mcts = evaluate_rl_mcts_on_episodes(
        episodes, bin_config, config, checkpoint_path,
        use_mcts=True,
    )
    print(f"  With MCTS:    {results_mcts['mean_fill']:.1%} +/- {results_mcts['std_fill']:.1%} "
          f"({results_mcts['mean_time']:.2f}s/ep)")

    # Without MCTS
    results_no_mcts = evaluate_rl_mcts_on_episodes(
        episodes, bin_config, config, checkpoint_path,
        use_mcts=False,
    )
    print(f"  Without MCTS: {results_no_mcts['mean_fill']:.1%} +/- {results_no_mcts['std_fill']:.1%} "
          f"({results_no_mcts['mean_time']:.2f}s/ep)")

    return {
        'with_mcts': results_mcts,
        'without_mcts': results_no_mcts,
    }


def comparison_eval(
    checkpoint_path: str,
    config: MCTSHybridConfig,
    num_episodes: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Compare rl_mcts_hybrid against all other strategies on the SAME episodes.
    """
    print("\n=== Comparison Evaluation ===")

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    episodes = generate_eval_episodes(
        num_episodes=num_episodes,
        num_boxes=config.num_boxes_per_episode,
        size_range=config.box_size_range,
        weight_range=config.box_weight_range,
        seed=42,
    )

    # Strategies to compare
    comparison_strategies = [
        "walle_scoring",
        "surface_contact",
        "baseline",
        "extreme_points",
        "rl_dqn",
        "rl_ppo",
        "rl_pct_transformer",
        "rl_hybrid_hh",
        "rl_a2c_masked",
    ]

    results = {}

    # Our strategy
    print(f"\n  Evaluating rl_mcts_hybrid ({num_episodes} episodes)...")
    results['rl_mcts_hybrid'] = evaluate_rl_mcts_on_episodes(
        episodes, bin_config, config, checkpoint_path,
        use_mcts=True,
    )
    print(f"    Fill: {results['rl_mcts_hybrid']['mean_fill']:.1%}")

    # Comparison strategies
    for name in comparison_strategies:
        print(f"  Evaluating {name}...")
        results[name] = evaluate_strategy_on_episodes(
            name, episodes, bin_config, config,
        )
        if 'error' not in results[name]:
            print(f"    Fill: {results[name]['mean_fill']:.1%}")
        else:
            print(f"    {results[name]['error']}")

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Strategy':<30} {'Mean Fill':>10} {'Std':>8} {'Time/ep':>10}")
    print("-" * 65)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('mean_fill', 0.0),
        reverse=True,
    )
    for name, r in sorted_results:
        if 'error' in r:
            continue
        marker = " <-- OURS" if name == "rl_mcts_hybrid" else ""
        print(
            f"  {name:<28} {r['mean_fill']:>8.1%}   "
            f"{r.get('std_fill', 0):>6.1%}   "
            f"{r.get('mean_time', 0):>7.2f}s{marker}"
        )
    print("=" * 65)

    return results


def ablation_eval(
    checkpoint_path: str,
    config: MCTSHybridConfig,
    num_episodes: int = 30,
) -> Dict[str, Dict[str, float]]:
    """
    Ablation study: measure impact of each component.
    """
    print("\n=== Ablation Study ===")

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    episodes = generate_eval_episodes(
        num_episodes=num_episodes,
        num_boxes=config.num_boxes_per_episode,
        size_range=config.box_size_range,
        weight_range=config.box_weight_range,
        seed=42,
    )

    results = {}

    # Full model with MCTS
    print("  [A] Full model + MCTS...")
    results['full_mcts'] = evaluate_rl_mcts_on_episodes(
        episodes, bin_config, config, checkpoint_path, use_mcts=True,
    )
    print(f"    Fill: {results['full_mcts']['mean_fill']:.1%}")

    # Without MCTS
    print("  [B] Full model, no MCTS...")
    results['no_mcts'] = evaluate_rl_mcts_on_episodes(
        episodes, bin_config, config, checkpoint_path, use_mcts=False,
    )
    print(f"    Fill: {results['no_mcts']['mean_fill']:.1%}")

    # Stochastic (sampling) vs deterministic
    print("  [C] Stochastic (sampling)...")
    results['stochastic'] = evaluate_rl_mcts_on_episodes(
        episodes, bin_config, config, checkpoint_path,
        use_mcts=False, deterministic=False,
    )
    print(f"    Fill: {results['stochastic']['mean_fill']:.1%}")

    # Fallback only (no model)
    print("  [D] Fallback heuristic only (walle_scoring)...")
    results['fallback'] = evaluate_strategy_on_episodes(
        config.fallback_strategy, episodes, bin_config, config,
    )
    print(f"    Fill: {results['fallback']['mean_fill']:.1%}")

    # Summary
    print("\n" + "-" * 50)
    print("Ablation Results:")
    for name, r in results.items():
        print(f"  {name:<25} {r['mean_fill']:.1%} +/- {r.get('std_fill', 0):.1%}")
    print("-" * 50)

    mcts_impact = (
        results['full_mcts']['mean_fill'] - results['no_mcts']['mean_fill']
    )
    model_impact = (
        results['no_mcts']['mean_fill'] - results['fallback']['mean_fill']
    )
    print(f"  MCTS impact:  {mcts_impact:+.1%}")
    print(f"  Model impact: {model_impact:+.1%}")

    return results


def difficulty_eval(
    checkpoint_path: str,
    config: MCTSHybridConfig,
    num_episodes: int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Difficulty sweep: vary box size distribution.
    """
    print("\n=== Difficulty Sweep ===")

    bin_config = BinConfig(
        length=config.bin_length,
        width=config.bin_width,
        height=config.bin_height,
        resolution=config.resolution,
    )

    difficulty_levels = {
        'easy (large boxes)':     (300.0, 600.0),
        'medium (mixed)':         (100.0, 600.0),
        'hard (small boxes)':     (50.0, 400.0),
        'very_hard (tiny boxes)': (50.0, 200.0),
    }

    results = {}

    for difficulty_name, size_range in difficulty_levels.items():
        print(f"\n  Difficulty: {difficulty_name} ({size_range})")

        episodes = generate_eval_episodes(
            num_episodes=num_episodes,
            num_boxes=config.num_boxes_per_episode,
            size_range=size_range,
            weight_range=config.box_weight_range,
            seed=42,
        )

        # RL MCTS Hybrid
        rl_result = evaluate_rl_mcts_on_episodes(
            episodes, bin_config, config, checkpoint_path,
        )
        print(f"    rl_mcts_hybrid: {rl_result['mean_fill']:.1%}")

        # Best heuristic (walle_scoring)
        heur_result = evaluate_strategy_on_episodes(
            "walle_scoring", episodes, bin_config, config,
        )
        print(f"    walle_scoring:  {heur_result['mean_fill']:.1%}")

        results[difficulty_name] = {
            'rl_mcts_hybrid': rl_result,
            'walle_scoring': heur_result,
            'gap': rl_result['mean_fill'] - heur_result['mean_fill'],
        }

    # Summary
    print("\n" + "-" * 60)
    print(f"{'Difficulty':<25} {'RL MCTS':>10} {'Heuristic':>10} {'Gap':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(
            f"  {name:<23} {r['rl_mcts_hybrid']['mean_fill']:>8.1%}   "
            f"{r['walle_scoring']['mean_fill']:>8.1%}   "
            f"{r['gap']:>+8.1%}"
        )
    print("-" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best checkpoint file in a directory."""
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir
    # Look for common checkpoint names in priority order
    for name in ["best_model.pt", "best.pt", "final_model.pt", "final.pt"]:
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            return path
    # Fallback: any .pt file, newest first
    import glob
    pts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")),
                 key=os.path.getmtime, reverse=True)
    return pts[0] if pts else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the MCTS-Guided Hierarchical Actor-Critic",
    )
    # Accept both --checkpoint (direct file) and --checkpoint_dir (directory)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint file",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Directory containing checkpoints (finds best automatically)",
    )
    parser.add_argument(
        "--mode", type=str, default="standard",
        choices=["standard", "comparison", "ablation", "difficulty", "all"],
        help="Evaluation mode",
    )
    # Accept both --episodes and --num_episodes
    parser.add_argument(
        "--episodes", "--num_episodes", type=int, default=None,
        dest="episodes",
        help="Number of evaluation episodes",
    )
    # Accept both --output and --output_dir
    parser.add_argument(
        "--output", "--output_dir", type=str, default=None,
        dest="output",
        help="Output path for results (directory or JSON file)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    config = MCTSHybridConfig()

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not checkpoint_path and args.checkpoint_dir:
        checkpoint_path = _find_best_checkpoint(args.checkpoint_dir)
    if not checkpoint_path:
        # Try default location
        default_dir = os.path.join(
            _WORKFLOW_ROOT, "outputs", "rl_mcts_hybrid", "checkpoints",
        )
        checkpoint_path = _find_best_checkpoint(default_dir)
    if not checkpoint_path:
        print("ERROR: No checkpoint found. Provide --checkpoint or --checkpoint_dir")
        sys.exit(1)

    print(f"Using checkpoint: {checkpoint_path}")

    # Output path — if it's a directory (or --output_dir was used), put JSON inside it
    output_path = args.output
    if output_path and os.path.isdir(output_path):
        output_path = os.path.join(output_path, "eval_results.json")
    elif output_path and not output_path.endswith(".json"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "eval_results.json")
    elif not output_path:
        output_path = os.path.join(
            _WORKFLOW_ROOT, "outputs", "rl_mcts_hybrid", "eval_results.json",
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_results = {}

    if args.mode in ("standard", "all"):
        n = args.episodes or 100
        all_results['standard'] = standard_eval(checkpoint_path, config, n)

    if args.mode in ("comparison", "all"):
        n = args.episodes or 50
        all_results['comparison'] = comparison_eval(checkpoint_path, config, n)

    if args.mode in ("ablation", "all"):
        n = args.episodes or 30
        all_results['ablation'] = ablation_eval(checkpoint_path, config, n)

    if args.mode in ("difficulty", "all"):
        n = args.episodes or 20
        all_results['difficulty'] = difficulty_eval(checkpoint_path, config, n)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
