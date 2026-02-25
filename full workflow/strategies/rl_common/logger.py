"""
TrainingLogger — Unified logging for RL training with visualisation.

Logs metrics to:
  1. CSV files (for analysis in pandas/Excel)
  2. TensorBoard (for live monitoring)
  3. Console (for HPC job output)

Provides:
  - Automatic training curve generation (matplotlib)
  - Reward / fill rate / loss plots over episodes
  - Comparison plots across strategies
  - Epoch-level and step-level logging

Usage:
    logger = TrainingLogger(log_dir="outputs/rl_dqn/logs", strategy_name="rl_dqn")
    for epoch in range(num_epochs):
        logger.log_episode(epoch, reward=12.5, fill=0.72, loss=0.05)
    logger.plot_training_curves()
    logger.close()
"""

from __future__ import annotations

import os
import csv
import json
import time
from typing import Dict, Optional, List, Any
from collections import defaultdict

import numpy as np


class TrainingLogger:
    """
    Unified training logger with CSV, TensorBoard, and matplotlib support.

    All log files are written to log_dir:
      log_dir/
        metrics.csv          — episode-level metrics
        config.json          — training configuration snapshot
        plots/
          training_curves.png
          reward_distribution.png
          fill_rate_progress.png

    TensorBoard logs go to log_dir/tensorboard/ if available.
    """

    def __init__(
        self,
        log_dir: str,
        strategy_name: str = "rl",
        use_tensorboard: bool = True,
    ):
        self.log_dir = os.path.abspath(log_dir)
        self.strategy_name = strategy_name
        self._use_tensorboard = use_tensorboard

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "plots"), exist_ok=True)

        # CSV writer
        self._csv_path = os.path.join(self.log_dir, "metrics.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = None  # Initialised on first write

        # In-memory history
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._step_count = 0
        self._start_time = time.time()

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(self.log_dir, "tensorboard")
                self._tb_writer = SummaryWriter(log_dir=tb_dir)
            except ImportError:
                print("[TrainingLogger] TensorBoard not available — logging to CSV only")

    def log_config(self, config: dict) -> None:
        """Save training configuration as JSON."""
        path = os.path.join(self.log_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_episode(self, episode: int, **metrics: float) -> None:
        """
        Log metrics for one episode.

        Args:
            episode: Episode number.
            **metrics: Named metrics (e.g., reward=12.5, fill=0.72).
        """
        # Add metadata
        metrics["episode"] = episode
        metrics["wall_time_s"] = time.time() - self._start_time

        # CSV
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=sorted(metrics.keys()),
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()

        # In-memory
        for k, v in metrics.items():
            self._history[k].append(v)

        # TensorBoard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                if k not in ("episode", "wall_time_s"):
                    self._tb_writer.add_scalar(f"{self.strategy_name}/{k}", v, episode)

        self._step_count += 1

    def log_step(self, global_step: int, **metrics: float) -> None:
        """Log metrics at step granularity (for loss curves)."""
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(f"{self.strategy_name}/step/{k}", v, global_step)

    def print_progress(
        self,
        episode: int,
        total_episodes: int,
        **metrics: float,
    ) -> None:
        """Print formatted progress to console."""
        pct = episode / max(total_episodes, 1) * 100
        elapsed = time.time() - self._start_time
        parts = [f"[{pct:5.1f}%] Ep {episode}/{total_episodes}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        parts.append(f"({elapsed:.0f}s)")
        print(" | ".join(parts), flush=True)

    def plot_training_curves(self, save: bool = True, show: bool = False) -> Optional[str]:
        """
        Generate and save training curve plots.

        Creates:
          - training_curves.png: reward + fill rate over episodes
          - reward_distribution.png: histogram of episode rewards
          - fill_rate_progress.png: fill rate with moving average

        Args:
            save: Save plots to log_dir/plots/.
            show: Display plots (set False for HPC).

        Returns:
            Path to training_curves.png if saved, else None.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend for HPC
            import matplotlib.pyplot as plt
        except ImportError:
            print("[TrainingLogger] matplotlib not available — skipping plots")
            return None

        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # ── Training curves (multi-panel) ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Curves: {self.strategy_name}", fontsize=14, fontweight="bold")

        # Panel 1: Episode reward
        if "reward" in self._history:
            rewards = self._history["reward"]
            episodes = list(range(len(rewards)))
            axes[0, 0].plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.5)
            # Moving average
            if len(rewards) > 10:
                window = min(50, len(rewards) // 5)
                ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
                axes[0, 0].plot(range(window-1, len(rewards)), ma, color="navy", linewidth=2)
            axes[0, 0].set_title("Episode Reward")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True, alpha=0.3)

        # Panel 2: Fill rate
        if "fill" in self._history:
            fills = self._history["fill"]
            episodes = list(range(len(fills)))
            axes[0, 1].plot(episodes, fills, alpha=0.3, color="forestgreen", linewidth=0.5)
            if len(fills) > 10:
                window = min(50, len(fills) // 5)
                ma = np.convolve(fills, np.ones(window)/window, mode="valid")
                axes[0, 1].plot(range(window-1, len(fills)), ma, color="darkgreen", linewidth=2)
            axes[0, 1].set_title("Average Closed Fill Rate")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Fill Rate")
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Loss
        if "loss" in self._history:
            losses = self._history["loss"]
            episodes = list(range(len(losses)))
            axes[1, 0].plot(episodes, losses, color="firebrick", linewidth=1)
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Panel 4: Epsilon / Entropy
        if "epsilon" in self._history:
            eps = self._history["epsilon"]
            axes[1, 1].plot(range(len(eps)), eps, color="darkorange", linewidth=2)
            axes[1, 1].set_title("Exploration (Epsilon)")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Epsilon")
            axes[1, 1].grid(True, alpha=0.3)
        elif "entropy" in self._history:
            ent = self._history["entropy"]
            axes[1, 1].plot(range(len(ent)), ent, color="purple", linewidth=2)
            axes[1, 1].set_title("Policy Entropy")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Entropy")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(plots_dir, "training_curves.png")
        if save:
            fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        # ── Reward distribution ──
        if "reward" in self._history and len(self._history["reward"]) > 20:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            rewards = self._history["reward"]
            n = len(rewards)
            # Compare first quarter vs last quarter
            q1 = rewards[:n//4]
            q4 = rewards[3*n//4:]
            ax2.hist(q1, bins=30, alpha=0.5, color="lightcoral", label=f"First {n//4} episodes")
            ax2.hist(q4, bins=30, alpha=0.5, color="steelblue", label=f"Last {n//4} episodes")
            ax2.set_title(f"Reward Distribution: {self.strategy_name}")
            ax2.set_xlabel("Episode Reward")
            ax2.set_ylabel("Count")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            if save:
                fig2.savefig(os.path.join(plots_dir, "reward_distribution.png"), dpi=150)
            plt.close(fig2)

        # ── Fill rate progress ──
        if "fill" in self._history and len(self._history["fill"]) > 20:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            fills = self._history["fill"]
            episodes = list(range(len(fills)))
            ax3.scatter(episodes, fills, alpha=0.2, s=5, color="forestgreen")
            # Rolling mean and std
            window = min(100, len(fills) // 5)
            if window > 1:
                ma = np.convolve(fills, np.ones(window)/window, mode="valid")
                x = range(window-1, len(fills))
                ax3.plot(x, ma, color="darkgreen", linewidth=2, label=f"MA({window})")
                # Rolling std band
                std_vals = []
                for i in range(len(fills) - window + 1):
                    std_vals.append(np.std(fills[i:i+window]))
                ax3.fill_between(
                    x, ma - np.array(std_vals), ma + np.array(std_vals),
                    alpha=0.15, color="green",
                )
            ax3.set_title(f"Fill Rate Progress: {self.strategy_name}")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Avg Closed Fill Rate")
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            if save:
                fig3.savefig(os.path.join(plots_dir, "fill_rate_progress.png"), dpi=150)
            plt.close(fig3)

        return curves_path

    def get_history(self) -> Dict[str, List[float]]:
        """Return the full metric history."""
        return dict(self._history)

    def get_best(self, metric: str = "fill") -> Optional[float]:
        """Return the best value of a metric."""
        if metric in self._history and self._history[metric]:
            return max(self._history[metric])
        return None

    def close(self) -> None:
        """Close all file handles."""
        if self._csv_file:
            self._csv_file.close()
        if self._tb_writer:
            self._tb_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
