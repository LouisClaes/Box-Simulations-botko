"""
Centralized output management for the box stacking pipeline.

Output structure:
    strategies/<strategy_name>/output/
        single_bin/          -- single-bin experiment runs
            <timestamp>/
                results.json
                packing.png
                run_animation.gif
        multibin/            -- multi-bin orchestrator/pipeline runs
            <timestamp>/
                results.json
        buffer/              -- buffer-sweep batch runs
            <timestamp>/
                results.json

Usage:
    # Single-bin run
    manager = ResultManager("baseline", mode="single_bin")
    manager.save_json(result)
    path = manager.get_render_path(type="packing", ext="png")

    # Multi-bin run
    manager = ResultManager("surface_contact", mode="multibin")
    manager.save_json(result)

    # Buffer sweep
    manager = ResultManager("surface_contact", mode="buffer")
    manager.save_json(result)
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Valid output modes
OUTPUT_MODES = ("single_bin", "multibin", "buffer")


class ResultManager:
    """
    Manages all file I/O and result structuring for a specific strategy.

    Outputs are organized under:
        strategies/<strategy_name>/output/<mode>/<timestamp>/

    Args:
        strategy_name:   Name of the strategy (used for folder path).
        mode:            Output mode -- "single_bin", "multibin", or "buffer".
        base_output_dir: Override the default output directory entirely.
    """

    def __init__(
        self,
        strategy_name: str,
        mode: str = "single_bin",
        base_output_dir: Optional[str] = None,
    ):
        self.strategy_name = strategy_name
        self.mode = mode if mode in OUTPUT_MODES else "single_bin"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if base_output_dir:
            # Use the explicit override path as-is (plus timestamp)
            self.output_dir = os.path.join(base_output_dir, self.timestamp)
        else:
            # Standard structure: strategies/<name>/output/<mode>/<timestamp>/
            self.output_dir = os.path.join(
                PROJECT_ROOT, "strategies", strategy_name, "output",
                self.mode, self.timestamp,
            )

        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Path generation
    # ---------------------------------------------------------------------------

    def get_run_json_path(self, run_id: Optional[int] = None) -> str:
        """Standard JSON path: results[_runID].json"""
        suffix = f"_run_{run_id}" if run_id is not None else ""
        return os.path.join(self.output_dir, f"results{suffix}.json")

    def get_aggregate_json_path(self) -> str:
        """Standard aggregate JSON path."""
        return os.path.join(self.output_dir, "results.json")

    def get_render_path(self, type: str, ext: str = "png", run_id: Optional[int] = None) -> str:
        """
        Standard visualization path.

        Args:
            type:   "packing", "stacking", "packing_grid", "stacking_grid", etc.
            ext:    File extension -- "png" or "gif".
            run_id: Optional run ID suffix.

        Returns:
            Full path: strategies/<strat>/output/<mode>/<ts>/<type>[_runID].<ext>
        """
        suffix = f"_run_{run_id}" if run_id is not None else ""
        return os.path.join(self.output_dir, f"{type}{suffix}.{ext}")

    # ---------------------------------------------------------------------------
    # Result construction
    # ---------------------------------------------------------------------------

    @staticmethod
    def build_single_run_result(
        config: Any,
        summary: Dict,
        placements: List[Dict],
        logs: List[Dict],
        completed: bool,
        stop_reason: Optional[str] = None,
    ) -> Dict:
        """Construct the standard single-run result dictionary."""
        result = {
            "experiment": {
                "strategy_name": config.strategy_name,
                "dataset_path": config.dataset_path,
                "timestamp": datetime.now().isoformat(),
                "config": config.to_dict(),
            },
            "metrics": summary,
            "placements": placements,
            "step_log": logs,
            "completed": completed,
        }
        if stop_reason:
            result["stopped_early"] = stop_reason
        return result

    @staticmethod
    def build_aggregate_result(
        strategy_name: str,
        dataset_info: Dict,
        bin_config: Any,
        aggregate_stats: Dict,
        runs: List[Dict],
    ) -> Dict:
        """Construct the standard multi-run aggregate result dictionary."""
        return {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_info,
            "bin_config": bin_config.to_dict(),
            "aggregate": aggregate_stats,
            "runs": runs,
        }

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save_json(self, data: Dict, path: Optional[str] = None) -> str:
        """Save dictionary to JSON file."""
        if path is None:
            if "aggregate" in data:
                path = self.get_aggregate_json_path()
            else:
                path = self.get_run_json_path()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
