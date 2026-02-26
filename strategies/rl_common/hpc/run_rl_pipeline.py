#!/usr/bin/env python
"""
Unified RL HPC pipeline for training, evaluation, and thesis visualizations.

Goals:
1. One command for all RL strategies.
2. Prioritize rl_mcts_hybrid (train it first, fail fast if broken).
3. Use all visible GPUs with per-process CUDA binding.
4. Produce normalized, comparable evaluation artifacts.
5. Fail closed (no synthetic/dummy results).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import queue
import shlex
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
STRATEGIES_ROOT = SCRIPT_PATH.parents[2]
WORKFLOW_ROOT = STRATEGIES_ROOT.parent

ALL_STRATEGIES = [
    "rl_mcts_hybrid",
    "rl_dqn",
    "rl_ppo",
    "rl_a2c_masked",
    "rl_pct_transformer",
    "rl_hybrid_hh",
]
GPU_STRATEGIES = {
    "rl_mcts_hybrid",
    "rl_dqn",
    "rl_ppo",
    "rl_a2c_masked",
    "rl_pct_transformer",
}

TRAIN_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "full": {
        "rl_mcts_hybrid": {"total_timesteps": 5_000_000, "num_envs": 1, "lr": "3e-4"},
        "rl_dqn": {"episodes": 50_000, "batch_size": 256, "lr": "0.001"},
        "rl_ppo": {"total_timesteps": 5_000_000, "num_envs": 16, "lr": "3e-4"},
        "rl_a2c_masked": {"num_updates": 200_000, "num_envs": 16, "lr": "1e-4"},
        "rl_pct_transformer": {"episodes": 200_000, "num_envs": 1, "lr": "3e-4"},
        "rl_hybrid_hh": {"mode": "dqn", "episodes": 50_000, "lr": "0.001"},
    },
    "quick": {
        "rl_mcts_hybrid": {"total_timesteps": 8_192, "num_envs": 1, "lr": "3e-4"},
        "rl_dqn": {"episodes": 200, "batch_size": 64, "lr": "0.001"},
        "rl_ppo": {"total_timesteps": 16_384, "num_envs": 2, "lr": "3e-4"},
        "rl_a2c_masked": {"num_updates": 300, "num_envs": 2, "lr": "1e-4"},
        "rl_pct_transformer": {"episodes": 300, "num_envs": 1, "lr": "3e-4"},
        "rl_hybrid_hh": {"mode": "dqn", "episodes": 200, "lr": "0.001"},
    },
}


@dataclass
class CommandResult:
    command: List[str]
    log_path: str
    return_code: int
    duration_s: float
    gpu_id: Optional[str]
    started_at: str
    finished_at: str
    stage: str
    strategy: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_strategy_list(raw: str) -> List[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("No strategies specified")
    invalid = [v for v in values if v not in ALL_STRATEGIES]
    if invalid:
        raise ValueError(f"Unknown strategies: {invalid}")
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def detect_gpu_ids(explicit: str) -> List[str]:
    """
    Resolve GPU IDs visible to this process.

    Order of precedence:
      1) --gpus
      2) CUDA_VISIBLE_DEVICES
      3) torch.cuda.device_count()
    """
    if explicit and explicit.lower() != "auto":
        return [x.strip() for x in explicit.split(",") if x.strip()]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [x.strip() for x in cvd.split(",") if x.strip()]

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        ids = [line.strip() for line in output.splitlines() if line.strip()]
        return ids
    except Exception:
        pass

    try:
        import torch

        count = int(torch.cuda.device_count())
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception:
        pass

    return []


def safe_json_dump(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def load_manifest(path: Path) -> Dict[str, Any]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def find_latest_run_dir(output_root: Path) -> Optional[Path]:
    if not output_root.is_dir():
        return None
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_run_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if args.mode in {"train", "full"}:
        run_dir = output_root / timestamp_tag()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    latest = find_latest_run_dir(output_root)
    if latest is None:
        raise FileNotFoundError(
            f"No run directory found under {output_root}. "
            "Provide --run_dir or run with --mode train/full first."
        )
    return latest


def discover_latest_file(patterns: Sequence[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        if "*" in str(pattern):
            candidates.extend(pattern.parent.glob(pattern.name))
        elif pattern.is_file():
            candidates.append(pattern)
    if not candidates:
        return None
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_resume_checkpoint(strategy: str, train_dir: Path) -> Optional[Path]:
    if strategy == "rl_mcts_hybrid":
        return discover_latest_file(
            [
                train_dir / "checkpoints" / "latest.pt",
                train_dir / "checkpoints" / "step_*.pt",
                train_dir / "checkpoints" / "final_model.pt",
            ]
        )
    if strategy == "rl_dqn":
        return discover_latest_file(
            [
                train_dir / "checkpoints" / "ep_*.pt",
                train_dir / "checkpoints" / "best_model.pt",
                train_dir / "checkpoints" / "final_model.pt",
            ]
        )
    if strategy == "rl_pct_transformer":
        return discover_latest_file(
            [
                train_dir / "logs" / "best.pt",
                train_dir / "logs" / "checkpoint_ep*.pt",
                train_dir / "logs" / "final.pt",
            ]
        )
    return None


def find_eval_checkpoint(strategy: str, train_dir: Path) -> Optional[Path]:
    if strategy == "rl_mcts_hybrid":
        return discover_latest_file(
            [
                train_dir / "checkpoints" / "best_model.pt",
                train_dir / "checkpoints" / "latest.pt",
                train_dir / "checkpoints" / "final_model.pt",
                train_dir / "checkpoints" / "step_*.pt",
            ]
        )
    if strategy == "rl_dqn":
        return discover_latest_file(
            [
                train_dir / "checkpoints" / "best_network.pt",
                train_dir / "checkpoints" / "final_network.pt",
                train_dir / "checkpoints" / "best_model.pt",
            ]
        )
    if strategy == "rl_ppo":
        return discover_latest_file(
            [
                train_dir / "logs" / "checkpoints" / "best_model.pt",
                train_dir / "logs" / "checkpoints" / "final_model.pt",
                train_dir / "logs" / "checkpoints" / "checkpoint_*.pt",
            ]
        )
    if strategy == "rl_a2c_masked":
        return discover_latest_file(
            [
                train_dir / "logs" / "checkpoints" / "best_model.pt",
                train_dir / "logs" / "checkpoints" / "final_model.pt",
                train_dir / "logs" / "checkpoints" / "checkpoint_*.pt",
            ]
        )
    if strategy == "rl_pct_transformer":
        return discover_latest_file(
            [
                train_dir / "logs" / "best.pt",
                train_dir / "logs" / "final.pt",
                train_dir / "logs" / "checkpoint_ep*.pt",
            ]
        )
    if strategy == "rl_hybrid_hh":
        return discover_latest_file(
            [
                train_dir / "best_model.pt",
                train_dir / "best_model.npz",
                train_dir / "final_model.pt",
                train_dir / "final_model.npz",
            ]
        )
    return None


def _cmd_train(strategy: str, python_bin: str, train_dir: Path, args: argparse.Namespace) -> List[str]:
    profile = TRAIN_PROFILES[args.profile][strategy]
    script = STRATEGIES_ROOT / strategy / "train.py"

    if strategy == "rl_mcts_hybrid":
        cmd = [
            python_bin,
            str(script),
            "--output_dir",
            str(train_dir),
            "--total_timesteps",
            str(profile["total_timesteps"]),
            "--num_envs",
            str(profile["num_envs"]),
            "--lr",
            str(profile["lr"]),
            "--seed",
            str(args.seed),
            "--resume",
            "auto",
        ]
        if args.profile == "quick":
            cmd.extend(
                [
                    "--skip_imitation",
                    "--checkpoint_interval",
                    "1024",
                    "--eval_interval",
                    "1024",
                    "--log_interval",
                    "256",
                ]
            )
        return cmd

    if strategy == "rl_dqn":
        cmd = [
            python_bin,
            str(script),
            "--output_dir",
            str(train_dir),
            "--episodes",
            str(profile["episodes"]),
            "--batch_size",
            str(profile["batch_size"]),
            "--lr",
            str(profile["lr"]),
        ]
        resume_path = find_resume_checkpoint(strategy, train_dir)
        if resume_path is not None:
            cmd.extend(["--resume", str(resume_path)])
        return cmd

    if strategy == "rl_ppo":
        return [
            python_bin,
            str(script),
            "--log_dir",
            str(train_dir / "logs"),
            "--total_timesteps",
            str(profile["total_timesteps"]),
            "--num_envs",
            str(profile["num_envs"]),
            "--lr",
            str(profile["lr"]),
            "--seed",
            str(args.seed),
        ]

    if strategy == "rl_a2c_masked":
        return [
            python_bin,
            str(script),
            "--log_dir",
            str(train_dir / "logs"),
            "--num_updates",
            str(profile["num_updates"]),
            "--num_envs",
            str(profile["num_envs"]),
            "--lr",
            str(profile["lr"]),
            "--seed",
            str(args.seed),
        ]

    if strategy == "rl_pct_transformer":
        cmd = [
            python_bin,
            str(script),
            "--log_dir",
            str(train_dir / "logs"),
            "--episodes",
            str(profile["episodes"]),
            "--num_envs",
            str(profile["num_envs"]),
            "--lr",
            str(profile["lr"]),
            "--seed",
            str(args.seed),
        ]
        resume_path = find_resume_checkpoint(strategy, train_dir)
        if resume_path is not None:
            cmd.extend(["--checkpoint", str(resume_path)])
        return cmd

    if strategy == "rl_hybrid_hh":
        return [
            python_bin,
            str(script),
            "--mode",
            str(profile["mode"]),
            "--episodes",
            str(profile["episodes"]),
            "--lr",
            str(profile["lr"]),
            "--output-dir",
            str(train_dir),
        ]

    raise ValueError(f"Unsupported strategy for training: {strategy}")


def _cmd_eval(
    strategy: str,
    python_bin: str,
    checkpoint_path: Path,
    eval_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    script = STRATEGIES_ROOT / strategy / "evaluate.py"
    if strategy == "rl_mcts_hybrid":
        return [
            python_bin,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--mode",
            "standard",
            "--episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--output",
            str(eval_dir / "eval_results_raw.json"),
        ]
    if strategy == "rl_dqn":
        return [
            python_bin,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(eval_dir),
        ]
    if strategy == "rl_ppo":
        return [
            python_bin,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--num_episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(eval_dir),
        ]
    if strategy == "rl_a2c_masked":
        return [
            python_bin,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--num_episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(eval_dir),
        ]
    if strategy == "rl_pct_transformer":
        return [
            python_bin,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--output",
            str(eval_dir / "eval_results_raw.json"),
        ]
    if strategy == "rl_hybrid_hh":
        return [
            python_bin,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(args.eval_episodes),
            "--output-dir",
            str(eval_dir),
        ]
    raise ValueError(f"Unsupported strategy for evaluation: {strategy}")


def run_command(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
    strategy: str,
    stage: str,
    gpu_id: Optional[str],
) -> CommandResult:
    started_at = utc_now()
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"[{started_at}] stage={stage} strategy={strategy}\n")
        log_file.write(f"[gpu] {gpu_id if gpu_id is not None else 'cpu'}\n")
        log_file.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
        if dry_run:
            log_file.write("[dry-run] command not executed\n")
            return CommandResult(
                command=cmd,
                log_path=str(log_path),
                return_code=0,
                duration_s=0.0,
                gpu_id=gpu_id,
                started_at=started_at,
                finished_at=utc_now(),
                stage=stage,
                strategy=strategy,
            )

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
        process.wait()

    finished_at = utc_now()
    return CommandResult(
        command=cmd,
        log_path=str(log_path),
        return_code=process.returncode,
        duration_s=time.time() - start,
        gpu_id=gpu_id,
        started_at=started_at,
        finished_at=finished_at,
        stage=stage,
        strategy=strategy,
    )

def locate_eval_json(eval_dir: Path) -> Optional[Path]:
    for name in (
        "eval_results_raw.json",
        "eval_results.json",
        "evaluation_results.json",
        "results.json",
    ):
        path = eval_dir / name
        if path.is_file():
            return path
    return None


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float)))


def normalize_eval_result(strategy: str, raw: Any, source_file: str) -> Dict[str, Any]:
    """
    Convert heterogeneous evaluator outputs to a unified schema:
      avg_fill, fill_std, placement_rate, avg_pallets_closed, ms_per_box, support_mean, fill_rates
    """
    result: Dict[str, Any] = {
        "strategy": strategy,
        "source_file": source_file,
        "avg_fill": 0.0,
        "fill_std": 0.0,
        "placement_rate": 0.0,
        "avg_pallets_closed": 0.0,
        "ms_per_box": 0.0,
        "support_mean": 0.0,
        "fill_rates": [],
    }

    # Already normalized
    if isinstance(raw, dict) and "avg_fill" in raw:
        result.update(raw)
        result["strategy"] = strategy
        result["source_file"] = source_file
        if not result.get("fill_rates"):
            result["fill_rates"] = [float(result["avg_fill"])]
        return result

    if strategy == "rl_mcts_hybrid" and isinstance(raw, dict):
        standard = raw.get("standard", raw)
        with_mcts = standard.get("with_mcts", standard)
        result["avg_fill"] = float(with_mcts.get("mean_fill", 0.0))
        result["fill_std"] = float(with_mcts.get("std_fill", 0.0))
        result["fill_rates"] = list(with_mcts.get("fill_rates", []))
        mean_time_s = float(with_mcts.get("mean_time", 0.0))
        if mean_time_s > 0:
            result["ms_per_box"] = mean_time_s * 1000.0 / max(1.0, float(raw.get("num_boxes", 100)))
        if not result["fill_rates"]:
            result["fill_rates"] = [result["avg_fill"]]
        return result

    if strategy == "rl_dqn" and isinstance(raw, dict):
        fills = list(raw.get("fills", []))
        result["avg_fill"] = float(raw.get("fill_mean", _mean(fills)))
        result["fill_std"] = float(raw.get("fill_std", _std(fills)))
        result["placement_rate"] = float(raw.get("placement_rate_mean", 0.0))
        result["avg_pallets_closed"] = float(raw.get("pallets_closed_mean", 0.0))
        result["ms_per_box"] = float(raw.get("ms_per_box_mean", 0.0))
        result["fill_rates"] = fills or [result["avg_fill"]]
        return result

    if strategy == "rl_ppo" and isinstance(raw, dict):
        ppo_metrics = raw.get("rl_ppo", raw)
        fill_rates = list(ppo_metrics.get("fill_rates", []))
        placement_rates = list(ppo_metrics.get("placement_rates", []))
        pallets_closed = list(ppo_metrics.get("pallets_closed", []))
        result["avg_fill"] = _mean(fill_rates)
        result["fill_std"] = _std(fill_rates)
        result["placement_rate"] = _mean(placement_rates)
        result["avg_pallets_closed"] = _mean(pallets_closed)
        result["fill_rates"] = fill_rates or [result["avg_fill"]]
        return result

    if strategy == "rl_a2c_masked" and isinstance(raw, dict):
        fill_block = raw.get("fill", {})
        result["avg_fill"] = float(fill_block.get("mean", 0.0))
        result["fill_std"] = float(fill_block.get("std", 0.0))
        episodes = raw.get("per_episode", raw.get("episodes", []))
        if isinstance(episodes, list):
            fills = [float(ep.get("fill", 0.0)) for ep in episodes if isinstance(ep, dict)]
            placed = [float(ep.get("placed", 0.0)) for ep in episodes if isinstance(ep, dict)]
            rejected = [float(ep.get("rejected", 0.0)) for ep in episodes if isinstance(ep, dict)]
            result["fill_rates"] = fills or [result["avg_fill"]]
            total_placed = float(np.sum(placed)) if placed else 0.0
            total_rejected = float(np.sum(rejected)) if rejected else 0.0
            denom = total_placed + total_rejected
            result["placement_rate"] = total_placed / denom if denom > 0 else 0.0
        else:
            result["fill_rates"] = [result["avg_fill"]]
        result["avg_pallets_closed"] = float(raw.get("pallets_closed", {}).get("mean", 0.0))
        return result

    if strategy == "rl_hybrid_hh":
        if isinstance(raw, list):
            target = next((x for x in raw if x.get("strategy") == "rl_hybrid_hh"), {})
        elif isinstance(raw, dict):
            target = raw.get("rl_hybrid_hh", raw)
        else:
            target = {}
        result["avg_fill"] = float(target.get("fill_mean", target.get("avg_fill", 0.0)))
        result["fill_std"] = float(target.get("fill_std", 0.0))
        result["placement_rate"] = float(target.get("placement_rate_mean", 0.0))
        result["ms_per_box"] = float(target.get("time_ms_mean", target.get("ms_per_box", 0.0)))
        fills = target.get("fills", [])
        result["fill_rates"] = list(fills) if isinstance(fills, list) and fills else [result["avg_fill"]]
        return result

    if strategy == "rl_pct_transformer" and isinstance(raw, dict):
        if "aggregate" in raw:
            agg = raw["aggregate"]
            result["avg_fill"] = float(agg.get("fill_mean", 0.0))
            result["fill_std"] = float(agg.get("fill_std", 0.0))
            result["placement_rate"] = float(agg.get("placement_rate_mean", 0.0))
            result["fill_rates"] = [result["avg_fill"]]
            return result
        target = raw.get("rl_pct_transformer", raw)
        result["avg_fill"] = float(target.get("fill_mean", target.get("avg_fill", 0.0)))
        result["fill_std"] = float(target.get("fill_std", 0.0))
        result["fill_rates"] = [result["avg_fill"]]
        return result

    # Best-effort fallback
    if isinstance(raw, dict):
        result["avg_fill"] = float(raw.get("fill_mean", raw.get("mean_fill", raw.get("avg_fill", 0.0))))
        result["fill_std"] = float(raw.get("fill_std", raw.get("std_fill", 0.0)))
        fill_rates = raw.get("fill_rates", [])
        if isinstance(fill_rates, list) and fill_rates:
            result["fill_rates"] = [float(x) for x in fill_rates]
        else:
            result["fill_rates"] = [result["avg_fill"]]
    return result


def read_training_history_csv(train_dir: Path) -> Optional[Tuple[List[float], List[float], List[float]]]:
    csv_candidates = [
        train_dir / "logs" / "metrics.csv",
        train_dir / "metrics.csv",
        train_dir / "logs" / "dqn" / "metrics.csv",
        train_dir / "logs" / "tabular" / "metrics.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.is_file()), None)
    if csv_path is None:
        return None

    xs: List[float] = []
    fills: List[float] = []
    rewards: List[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        step_counter = 0
        for row in reader:
            step_counter += 1
            x_val = (
                row.get("total_steps")
                or row.get("global_step")
                or row.get("step")
                or row.get("episode")
                or str(step_counter)
            )
            try:
                xs.append(float(x_val))
            except Exception:
                xs.append(float(step_counter))

            fill_value = None
            for key in ("fill", "avg_fill", "eval_fill", "fill_rate", "eval_fill_mean"):
                if key in row and row[key] not in ("", None):
                    fill_value = row[key]
                    break
            reward_value = None
            for key in ("reward", "mean_reward", "eval_reward", "return"):
                if key in row and row[key] not in ("", None):
                    reward_value = row[key]
                    break
            try:
                fills.append(float(fill_value) if fill_value is not None else np.nan)
            except Exception:
                fills.append(np.nan)
            try:
                rewards.append(float(reward_value) if reward_value is not None else np.nan)
            except Exception:
                rewards.append(np.nan)
    return xs, fills, rewards


def generate_visualizations(run_dir: Path, strategies: Sequence[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for visualization: {exc}") from exc

    eval_dir = run_dir / "evaluation"
    comparison_dir = eval_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, Any]] = {}
    for strategy in strategies:
        norm_path = eval_dir / strategy / "eval_results.json"
        if norm_path.is_file():
            with norm_path.open("r", encoding="utf-8") as handle:
                metrics[strategy] = json.load(handle)

    if not metrics:
        raise FileNotFoundError(
            f"No normalized evaluation metrics found in {eval_dir}. "
            "Run --mode evaluate first."
        )

    ordered = sorted(metrics.keys(), key=lambda s: metrics[s].get("avg_fill", 0.0), reverse=True)
    fills = [float(metrics[s].get("avg_fill", 0.0)) for s in ordered]
    stds = [float(metrics[s].get("fill_std", 0.0)) for s in ordered]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(ordered, fills, yerr=stds, capsize=4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average Fill Rate")
    ax.set_title("RL Strategy Fill-Rate Comparison")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, fills):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{value:.3f}", ha="center")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(comparison_dir / "fill_rate_comparison.png", dpi=200)
    fig.savefig(comparison_dir / "fill_rate_comparison.pdf")
    plt.close(fig)

    cols = 3
    rows = int(np.ceil(len(ordered) / cols))
    fig2, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for idx, strategy in enumerate(ordered):
        row = idx // cols
        col = idx % cols
        ax2 = axes_arr[row, col]
        values = metrics[strategy].get("fill_rates", [])
        values = [float(v) for v in values if isinstance(v, (int, float))]
        if values:
            ax2.hist(values, bins=min(30, max(5, len(values) // 2)), alpha=0.8)
            ax2.axvline(np.mean(values), color="red", linestyle="--", linewidth=1.5)
        ax2.set_title(strategy)
        ax2.set_xlim(0.0, 1.0)
        ax2.grid(alpha=0.2)
    for idx in range(len(ordered), rows * cols):
        row = idx // cols
        col = idx % cols
        axes_arr[row, col].axis("off")
    fig2.suptitle("Evaluation Fill-Rate Distributions", y=1.01)
    plt.tight_layout()
    fig2.savefig(comparison_dir / "fill_distribution_grid.png", dpi=200)
    fig2.savefig(comparison_dir / "fill_distribution_grid.pdf")
    plt.close(fig2)

    fig3, axes3 = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes3_arr = np.atleast_1d(axes3).reshape(rows, cols)
    for idx, strategy in enumerate(ordered):
        row = idx // cols
        col = idx % cols
        ax3 = axes3_arr[row, col]
        history = read_training_history_csv(run_dir / strategy)
        if history is None:
            ax3.set_title(f"{strategy} (no metrics.csv)")
            ax3.axis("off")
            continue
        xs, fill_vals, _reward_vals = history
        x_arr = np.asarray(xs, dtype=float)
        fill_arr = np.asarray(fill_vals, dtype=float)
        valid = np.isfinite(fill_arr)
        if np.any(valid):
            ax3.plot(x_arr[valid], fill_arr[valid], alpha=0.25, linewidth=1.0)
            window = min(100, max(5, int(np.sum(valid) / 20)))
            series = fill_arr[valid]
            if series.size >= window:
                kernel = np.ones(window, dtype=float) / float(window)
                smooth = np.convolve(series, kernel, mode="valid")
                smooth_x = x_arr[valid][window - 1 :]
                ax3.plot(smooth_x, smooth, linewidth=2.0)
        ax3.set_title(strategy)
        ax3.set_ylim(0.0, 1.0)
        ax3.grid(alpha=0.2)
    for idx in range(len(ordered), rows * cols):
        row = idx // cols
        col = idx % cols
        axes3_arr[row, col].axis("off")
    fig3.suptitle("Training Fill Curves (Raw + Smoothed)", y=1.01)
    plt.tight_layout()
    fig3.savefig(comparison_dir / "training_fill_grid.png", dpi=200)
    fig3.savefig(comparison_dir / "training_fill_grid.pdf")
    plt.close(fig3)

    summary_csv = comparison_dir / "summary_table.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "avg_fill",
                "fill_std",
                "placement_rate",
                "avg_pallets_closed",
                "ms_per_box",
            ]
        )
        for strategy in ordered:
            row = metrics[strategy]
            writer.writerow(
                [
                    strategy,
                    row.get("avg_fill", 0.0),
                    row.get("fill_std", 0.0),
                    row.get("placement_rate", 0.0),
                    row.get("avg_pallets_closed", 0.0),
                    row.get("ms_per_box", 0.0),
                ]
            )


def run_compare_script(
    python_bin: str,
    run_dir: Path,
    log_path: Path,
    dry_run: bool,
    expected_strategies: Sequence[str],
) -> CommandResult:
    compare_script = STRATEGIES_ROOT / "rl_common" / "compare_strategies.py"
    command = [
        python_bin,
        str(compare_script),
        "--eval_dir",
        str(run_dir / "evaluation"),
        "--output_dir",
        str(run_dir / "evaluation" / "comparison"),
        "--strict",
        "--expected_strategies",
        ",".join(expected_strategies),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKFLOW_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return run_command(
        command,
        cwd=WORKFLOW_ROOT,
        env=env,
        log_path=log_path,
        dry_run=dry_run,
        strategy="all",
        stage="compare",
        gpu_id=None,
    )


def make_base_manifest(args: argparse.Namespace, run_dir: Path, strategies: Sequence[str]) -> Dict[str, Any]:
    return {
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "mode": args.mode,
        "profile": args.profile,
        "run_dir": str(run_dir),
        "workflow_root": str(WORKFLOW_ROOT),
        "strategies": list(strategies),
        "gpu_ids": detect_gpu_ids(args.gpus),
        "jobs": {},
    }


def strategy_env(gpu_id: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKFLOW_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def train_phase(
    args: argparse.Namespace,
    run_dir: Path,
    strategies: Sequence[str],
    manifest: Dict[str, Any],
) -> None:
    gpu_ids = detect_gpu_ids(args.gpus)
    required_gpu_strategies = [s for s in strategies if s in GPU_STRATEGIES]
    if required_gpu_strategies and not gpu_ids and not args.dry_run:
        raise RuntimeError(
            f"GPU strategies requested ({required_gpu_strategies}) but no GPUs detected."
        )

    max_workers = args.max_parallel if args.max_parallel > 0 else max(1, len(gpu_ids) + 1)
    gpu_queue: "queue.Queue[str]" = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path)
    manifest_lock = threading.Lock()

    def execute_training(strategy: str) -> CommandResult:
        gpu_id: Optional[str] = None
        borrowed_gpu = False
        if strategy in GPU_STRATEGIES:
            if gpu_ids:
                gpu_id = gpu_queue.get()
                borrowed_gpu = True
            elif args.dry_run:
                gpu_id = "dryrun-gpu"
        try:
            strategy_train_dir = run_dir / strategy
            strategy_train_dir.mkdir(parents=True, exist_ok=True)
            command = _cmd_train(strategy, args.python_bin, strategy_train_dir, args)
            result = run_command(
                command,
                cwd=WORKFLOW_ROOT,
                env=strategy_env(gpu_id),
                log_path=logs_dir / f"train_{strategy}.log",
                dry_run=args.dry_run,
                strategy=strategy,
                stage="train",
                gpu_id=gpu_id,
            )
            return result
        finally:
            if borrowed_gpu and gpu_id is not None:
                gpu_queue.put(gpu_id)

    pending = list(strategies)
    if "rl_mcts_hybrid" in strategies and not args.skip_mcts_priority:
        mcts_res = execute_training("rl_mcts_hybrid")
        manifest["jobs"].setdefault("rl_mcts_hybrid", {})["train"] = mcts_res.__dict__
        safe_json_dump(manifest, manifest_path)
        if mcts_res.return_code != 0 and not args.continue_on_error:
            raise RuntimeError(
                "rl_mcts_hybrid training failed. "
                f"Check log: {mcts_res.log_path}"
            )
        pending = [s for s in pending if s != "rl_mcts_hybrid"]

    failures: List[CommandResult] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(execute_training, strategy): strategy for strategy in pending}
            for future in as_completed(futures):
                strategy = futures[future]
                result = future.result()
                with manifest_lock:
                    manifest["jobs"].setdefault(strategy, {})["train"] = result.__dict__
                    safe_json_dump(manifest, manifest_path)
                if result.return_code != 0:
                    failures.append(result)

    if failures and not args.continue_on_error:
        failed_names = ", ".join(sorted({f.strategy for f in failures}))
        raise RuntimeError(
            f"Training failed for: {failed_names}. "
            f"Inspect logs under {logs_dir}"
        )


def evaluate_phase(
    args: argparse.Namespace,
    run_dir: Path,
    strategies: Sequence[str],
    manifest: Dict[str, Any],
) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_root = run_dir / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path)

    gpu_ids = detect_gpu_ids(args.gpus)
    eval_gpu_id = gpu_ids[0] if gpu_ids else None

    for strategy in strategies:
        train_dir = run_dir / strategy
        eval_dir = eval_root / strategy
        eval_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = find_eval_checkpoint(strategy, train_dir)
        if checkpoint is None:
            entry = {
                "strategy": strategy,
                "stage": "evaluate",
                "status": "skipped_no_checkpoint",
                "checkpoint": None,
            }
            manifest["jobs"].setdefault(strategy, {})["eval"] = entry
            safe_json_dump(manifest, manifest_path)
            if not args.continue_on_error:
                raise FileNotFoundError(f"No checkpoint found for {strategy} in {train_dir}")
            continue

        command = _cmd_eval(strategy, args.python_bin, checkpoint, eval_dir, args)
        gpu_id = eval_gpu_id if strategy in GPU_STRATEGIES else None
        result = run_command(
            command,
            cwd=WORKFLOW_ROOT,
            env=strategy_env(gpu_id),
            log_path=logs_dir / f"eval_{strategy}.log",
            dry_run=args.dry_run,
            strategy=strategy,
            stage="evaluate",
            gpu_id=gpu_id,
        )
        manifest["jobs"].setdefault(strategy, {})["eval"] = result.__dict__
        manifest["jobs"][strategy]["eval"]["checkpoint"] = str(checkpoint)
        safe_json_dump(manifest, manifest_path)

        if result.return_code != 0:
            if args.continue_on_error:
                continue
            raise RuntimeError(f"Evaluation failed for {strategy}. See {result.log_path}")

        raw_json = locate_eval_json(eval_dir)
        if raw_json is None:
            if args.continue_on_error:
                continue
            raise FileNotFoundError(f"No evaluation JSON found for {strategy} in {eval_dir}")

        with raw_json.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        normalized = normalize_eval_result(strategy, raw_data, str(raw_json))
        norm_path = eval_dir / "eval_results.json"
        safe_json_dump(normalized, norm_path)
        manifest["jobs"][strategy]["normalized_eval"] = str(norm_path)
        manifest["jobs"][strategy]["normalized_metrics"] = normalized
        safe_json_dump(manifest, manifest_path)


def visualize_phase(
    args: argparse.Namespace,
    run_dir: Path,
    strategies: Sequence[str],
    manifest: Dict[str, Any],
) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path)

    compare_result = run_compare_script(
        args.python_bin,
        run_dir,
        log_path=logs_dir / "compare_strategies.log",
        dry_run=args.dry_run,
        expected_strategies=strategies,
    )
    manifest["jobs"].setdefault("all", {})["compare"] = compare_result.__dict__
    safe_json_dump(manifest, manifest_path)
    if compare_result.return_code != 0 and not args.continue_on_error:
        raise RuntimeError(
            "compare_strategies.py failed. "
            f"See {compare_result.log_path}"
        )

    if not args.dry_run:
        try:
            generate_visualizations(run_dir, strategies)
            manifest["jobs"].setdefault("all", {})["thesis_visualizations"] = {
                "status": "ok",
                "output_dir": str(run_dir / "evaluation" / "comparison"),
                "finished_at": utc_now(),
            }
            safe_json_dump(manifest, manifest_path)
        except Exception as exc:
            manifest["jobs"].setdefault("all", {})["thesis_visualizations"] = {
                "status": "failed",
                "error": str(exc),
                "finished_at": utc_now(),
            }
            safe_json_dump(manifest, manifest_path)
            if not args.continue_on_error:
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified RL HPC pipeline (train/evaluate/visualize).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["train", "evaluate", "visualize", "full"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "quick"],
        help="Training budget profile",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Existing run directory (auto-created for train/full if omitted)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(WORKFLOW_ROOT / "outputs" / "rl_training"),
        help="Root directory for run folders",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(ALL_STRATEGIES),
        help="Comma-separated strategy list",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="GPU IDs to use, e.g. 0,1,2,3 or auto",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=0,
        help="Maximum parallel training jobs (0 = auto)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="Episodes per strategy during evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used for subcommands",
    )
    parser.add_argument("--dry_run", action="store_true", help="Log commands without executing")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Do not fail the whole pipeline on per-strategy errors",
    )
    parser.add_argument(
        "--skip_mcts_priority",
        action="store_true",
        help="Do not force rl_mcts_hybrid to run before other strategies",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="Optional explicit path for run manifest JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = parse_strategy_list(args.strategies)
    run_dir = resolve_run_dir(args)
    if not args.manifest_path:
        args.manifest_path = str(run_dir / "run_manifest.json")

    manifest_path = Path(args.manifest_path)
    manifest = load_manifest(manifest_path)
    if not manifest:
        manifest = make_base_manifest(args, run_dir, strategies)
    manifest["last_invocation"] = {
        "timestamp": utc_now(),
        "argv": sys.argv,
        "mode": args.mode,
        "run_dir": str(run_dir),
        "strategies": strategies,
    }
    safe_json_dump(manifest, manifest_path)

    if args.mode in {"train", "full"}:
        train_phase(args, run_dir, strategies, manifest)
    if args.mode in {"evaluate", "full"}:
        evaluate_phase(args, run_dir, strategies, manifest)
    if args.mode in {"visualize", "full"}:
        visualize_phase(args, run_dir, strategies, manifest)

    manifest["completed_at"] = utc_now()
    manifest["status"] = "done"
    safe_json_dump(manifest, manifest_path)

    print(f"Pipeline complete. Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
