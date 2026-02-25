"""
Grid creator — combine multiple packing renders into photo grids.

Creates:
  - PNG grids:  2×5 grid of 10 final packing renders
  - GIF grids:  2×5 animated grid of 10 stacking sequences

Requires: pyvista, Pillow (PIL), numpy
"""

import os
from typing import List, Dict, Optional, Tuple

import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

from config import Placement, BinConfig
from visualization.render_3d import get_figure

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plotter_to_pil(plotter: pv.Plotter) -> Image.Image:
    """Render a PyVista Plotter to a PIL Image (in-memory)."""
    # Make sure we ask pyvista nicely for the numpy array
    img_array = plotter.screenshot(None, return_img=True)
    plotter.close()
    return Image.fromarray(img_array)

def _add_label(img: Image.Image, label: str) -> Image.Image:
    """Add a text label bar at the top of an image."""
    bar_height = 28
    new_img = Image.new('RGB', (img.width, img.height + bar_height), (255, 255, 255))

    # Draw text on bar
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Center text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = (img.width - text_w) // 2
    draw.text((x, 4), label, fill=(0, 0, 0), font=font)

    # Paste original image below label bar
    new_img.paste(img, (0, bar_height))
    return new_img

def _assemble_grid(images: List[Image.Image], cols: int = 5) -> Image.Image:
    """Arrange images into a grid with `cols` columns."""
    if not images:
        return Image.new('RGB', (100, 100), (255, 255, 255))

    # Resize all to same dimensions (use first image as reference)
    cell_w = images[0].width
    cell_h = images[0].height
    resized = []
    for img in images:
        if img.size != (cell_w, cell_h):
            resized.append(img.resize((cell_w, cell_h), Image.LANCZOS))
        else:
            resized.append(img)

    rows = (len(resized) + cols - 1) // cols
    grid_w = cols * cell_w
    grid_h = rows * cell_h

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    for idx, img in enumerate(resized):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * cell_w, r * cell_h))

    return grid

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_png_grid(
    runs_data: List[Dict],
    bin_config: BinConfig,
    save_path: str,
    strategy_name: str = "",
    cols: int = 5,
) -> str:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    cell_images: List[Image.Image] = []

    for rd in runs_data:
        placements = rd['placements']
        run_idx = rd['run_index']
        fill_rate = rd['fill_rate']
        time_ms = rd['computation_time_ms']

        title = ""
        plotter = get_figure(placements, bin_config, title=title)
        # 4:3ish ratio for individual grid images
        plotter.window_size = (1600, 1200) 
        img = _plotter_to_pil(plotter)
        cell_images.append(img)

    grid = _assemble_grid(cell_images, cols=cols)

    # Removed overall title bar addition to keep the visualisation clean

    grid.save(save_path, quality=95)
    return save_path

def create_gif_grid(
    runs_data: List[Dict],
    bin_config: BinConfig,
    save_path: str,
    strategy_name: str = "",
    fps: int = 2,
    cols: int = 5,
) -> str:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    all_sorted_placements: List[List[Placement]] = []
    run_labels: List[str] = []
    max_steps = 0

    for rd in runs_data:
        placements = sorted(rd['placements'], key=lambda p: p.step)
        all_sorted_placements.append(placements)
        run_labels.append("")
        if len(placements) > max_steps:
            max_steps = len(placements)

    if max_steps == 0:
        print("  Warning: No placements found. GIF not created.")
        return save_path

    grid_frames: List[Image.Image] = []

    for step in range(1, max_steps + 1):
        cell_images: List[Image.Image] = []

        for run_idx, (placements, label) in enumerate(zip(all_sorted_placements, run_labels)):
            n = min(step, len(placements))
            current = placements[:n]

            step_label = ""
            plotter = get_figure(current, bin_config, title=step_label)
            plotter.window_size = (1600, 1200)
            img = _plotter_to_pil(plotter)
            cell_images.append(img)

        grid = _assemble_grid(cell_images, cols=cols)
        grid_frames.append(grid)

        if step % 5 == 0 or step == max_steps:
            print(f"    GIF grid frame {step}/{max_steps}")

    if not grid_frames:
        print("  Warning: No frames rendered. GIF not created.")
        return save_path

    ms_per_frame = int(1000 / fps)
    durations = [ms_per_frame] * len(grid_frames)
    durations[-1] = ms_per_frame * 3 

    grid_frames[0].save(
        save_path,
        save_all=True,
        append_images=grid_frames[1:],
        duration=durations,
        loop=0,
    )

    return save_path
