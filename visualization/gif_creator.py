"""
GIF creator — animated stacking visualization using PyVista.

Renders each placement step as a PyVista scene and combines
them into an animated GIF using Pillow.
"""

import io
import os
from typing import List

from PIL import Image
import pyvista as pv

from config import Placement, BinConfig
from visualization.render_3d import get_figure

def _plotter_to_pil(plotter: pv.Plotter) -> Image.Image:
    """Render a PyVista Plotter to a PIL Image (in-memory)."""
    img_array = plotter.screenshot(None, return_img=True)
    plotter.close()
    return Image.fromarray(img_array)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_stacking_gif(
    placements: List[Placement],
    bin_config: BinConfig,
    save_path: str,
    title: str = "",
    fps: int = 2,
    resolution: tuple = (2560, 1440),
) -> str:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    sorted_placements = sorted(placements, key=lambda p: p.step)
    frames: List[Image.Image] = []

    for i in range(1, len(sorted_placements) + 1):
        current = sorted_placements[:i]
        
        # Removed step_title overlay completely for clean generation
        plotter = get_figure(current, bin_config, title="")
        plotter.window_size = resolution
        
        try:
            frame = _plotter_to_pil(plotter)
            frames.append(frame)
        except Exception as e:
            print(f"  Warning: Could not render frame {i}: {e}")
            continue

    if not frames:
        print("  Warning: No frames rendered. GIF not created.")
        return save_path

    ms_per_frame = int(1000 / fps)
    durations = [ms_per_frame] * len(frames)
    durations[-1] = ms_per_frame * 3

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )

    return save_path
