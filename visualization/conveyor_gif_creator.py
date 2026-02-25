"""
Conveyor-belt GIF creator — animated dual-pallet stacking with conveyor using PyVista.

Renders each placement step as a PyVista Plotter with three subplots:
  - Left:   Pallet 0
  - Center: Conveyor belt with buffered boxes
  - Right:  Pallet 1

Supports grid mode: render N experiments side-by-side in an (rows x cols) grid,
where each cell is one [pallet0 | conveyor | pallet1] triplet.

Requires: pyvista, numpy, Pillow
"""

import io
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pyvista as pv
from PIL import Image

from config import Placement, BinConfig, Box
from visualization.render_3d import _get_box_color

# ─────────────────────────────────────────────────────────────────────────────
# Step record — data collected per placement step for animation
# ─────────────────────────────────────────────────────────────────────────────

class ConveyorStep:
    """Data for one animation frame."""
    __slots__ = (
        "step", "box", "bin_index", "placed",
        "placement", "buffer_snapshot", "stream_remaining",
        "pallet0_placements", "pallet1_placements",
    )

    def __init__(
        self,
        step: int,
        box: Box,
        bin_index: int,
        placed: bool,
        placement: Optional[Placement],
        buffer_snapshot: List[Box],
        stream_remaining: int,
        pallet0_placements: List[Placement],
        pallet1_placements: List[Placement],
    ):
        self.step = step
        self.box = box
        self.bin_index = bin_index
        self.placed = placed
        self.placement = placement
        self.buffer_snapshot = buffer_snapshot
        self.stream_remaining = stream_remaining
        self.pallet0_placements = list(pallet0_placements)
        self.pallet1_placements = list(pallet1_placements)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_pallet_boxes(plotter: pv.Plotter, placements: List[Placement], bin_config: BinConfig, title: str):
    p_len, p_wid, p_hei = bin_config.length, bin_config.width, bin_config.height
    z_limit = max(p_hei, 3000) if p_hei > 2000 else p_hei

    bounds = (0, p_len, 0, p_wid, 0, z_limit)
    bbox = pv.Box(bounds=bounds)
    plotter.add_mesh(bbox, style='wireframe', color='gray', line_width=1.0, opacity=0.3)

    n_x = int(p_len / 200) + 1
    n_y = int(p_wid / 200) + 1
    n_z = int(z_limit / 200) + 1

    for p in placements:
        box_mesh = pv.Box(bounds=(p.x, p.x + p.oriented_l, p.y, p.y + p.oriented_w, p.z, p.z + p.oriented_h))
        color = _get_box_color(p.step)
        plotter.add_mesh(
            box_mesh, 
            color=color, 
            show_edges=True, 
            edge_color='black', 
            line_width=1.0, 
            specular=0.1,
            ambient=0.4,
            diffuse=0.8
        )

    plotter.show_bounds(
        grid='back',
        location='outer',
        ticks='inside',
        axes_ranges=[0, p_len, 0, p_wid, 0, z_limit],
        show_xlabels=True, show_ylabels=True, show_zlabels=True,
        xtitle='X', ytitle='Y', ztitle='Z',
        color='black', font_size=10, font_family='arial',
        n_xlabels=n_x, n_ylabels=n_y, n_zlabels=n_z,
    )
    
    # Removed title drawing for cleaner visualisations

    plotter.camera_position = 'iso'
    plotter.camera.azimuth = -100
    plotter.camera.elevation = 5
    plotter.camera.zoom(0.9)

def _draw_conveyor(plotter: pv.Plotter, buffer_boxes: List[Box], all_remaining: int,
                   highlight_box_id: Optional[int] = None,
                   arrow_side: Optional[str] = None,
                   bin_config: Optional[BinConfig] = None,
                   buffer_capacity: int = 8):
    """Draw conveyor belt with boxes at fixed slot positions for consistent framing.

    Every frame renders the same belt size and bounding volume, so the camera
    never auto-scales when boxes are added or removed.
    """
    BELT_LENGTH = 1200.0   # Fixed length (matches pallet width for visual balance)
    BELT_WIDTH = 400.0     # Fixed width
    MAX_SCENE_Z = 700.0    # Fixed ceiling — tallest Rajapack box is 600mm

    # Fixed scene frame — always present, anchors the camera to the same region
    frame = pv.Box(bounds=(0, BELT_LENGTH, 0, BELT_WIDTH, 0, MAX_SCENE_Z))
    plotter.add_mesh(frame, style='wireframe', color='lightgray', line_width=0.5, opacity=0.15)

    # Belt surface
    belt = pv.Box(bounds=(0, BELT_LENGTH, 0, BELT_WIDTH, -10, 0))
    plotter.add_mesh(belt, color='slategray', opacity=0.5, show_edges=True, edge_color='dimgray')

    # Place boxes at fixed slot positions (slot 0 = front/left, slot N-1 = back/right)
    # Each slot has a fixed center X regardless of how many boxes are currently visible.
    if buffer_boxes:
        slot_width = BELT_LENGTH / buffer_capacity

        for i, box in enumerate(buffer_boxes):
            x_center = slot_width * i + slot_width / 2
            x_pos = max(0.0, x_center - box.length / 2)
            if x_pos + box.length > BELT_LENGTH:
                x_pos = BELT_LENGTH - box.length
            y_pos = (BELT_WIDTH - box.width) / 2
            z_pos = 0.0

            is_highlighted = (highlight_box_id is not None and box.id == highlight_box_id)
            edge = "red" if is_highlighted else "black"
            lw = 6.0 if is_highlighted else 1.0
            color = _get_box_color(box.id)

            b = pv.Box(bounds=(
                x_pos, x_pos + box.length,
                y_pos, y_pos + box.width,
                z_pos, z_pos + box.height,
            ))
            plotter.add_mesh(
                b, color=color, show_edges=True, edge_color=edge, line_width=lw,
                specular=0.1, ambient=0.4, diffuse=0.8,
            )

    plotter.camera_position = 'iso'
    plotter.camera.azimuth = -100
    plotter.camera.elevation = 5
    plotter.camera.zoom(0.9)

def _plotter_to_pil(plotter: pv.Plotter) -> Image.Image:
    img_array = plotter.screenshot(None, return_img=True)
    plotter.close()
    return Image.fromarray(img_array)

# ─────────────────────────────────────────────────────────────────────────────
# Single-cell renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_cell(
    step_data: ConveyorStep,
    bin_config: BinConfig,
    cell_title: str = "",
    window_size: tuple = (3000, 1000),
    buffer_capacity: int = 8,
) -> pv.Plotter:
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, border=False)
    plotter.set_background('white')
    plotter.window_size = window_size
    plotter.enable_lightkit()

    # Left
    plotter.subplot(0, 0)
    _draw_pallet_boxes(plotter, step_data.pallet0_placements, bin_config,
                       title="")

    # Center (Conveyor)
    plotter.subplot(0, 1)
    arrow_side = None
    if step_data.placed:
        arrow_side = "left" if step_data.bin_index == 0 else "right"
    _draw_conveyor(plotter, step_data.buffer_snapshot, step_data.stream_remaining,
                   highlight_box_id=step_data.box.id if step_data.placed else None,
                   arrow_side=arrow_side, bin_config=bin_config,
                   buffer_capacity=buffer_capacity)
    
    # Removed cell_title string appending to keep views clean

    # Right
    plotter.subplot(0, 2)
    _draw_pallet_boxes(plotter, step_data.pallet1_placements, bin_config,
                       title="")

    return plotter

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_conveyor_gif(
    steps: List[ConveyorStep],
    bin_config: BinConfig,
    save_path: str,
    title: str = "",
    fps: int = 2,
) -> str:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    frames: List[Image.Image] = []

    for step_idx, step_data in enumerate(steps):
        plotter = _render_cell(step_data, bin_config)
        # Removed overlaying the step_title text

        try:
            frame = _plotter_to_pil(plotter)
            frames.append(frame)
        except Exception as e:
            print(f"  Warning: Could not render frame {step_idx}: {e}")

    if not frames:
        print("  Warning: No frames rendered. GIF not created.")
        return save_path

    ms_per_frame = int(1000 / fps)
    durations = [ms_per_frame] * len(frames)
    durations[-1] = ms_per_frame * 4

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  GIF saved: {save_path}  ({len(frames)} frames)")
    return save_path

def create_conveyor_grid_gif(
    experiments: List[Dict],
    bin_config: BinConfig,
    save_path: str,
    grid_cols: int = 3,
    title: str = "",
    fps: int = 2,
) -> str:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    n_experiments = len(experiments)
    grid_rows = (n_experiments + grid_cols - 1) // grid_cols
    max_steps = max(len(exp["steps"]) for exp in experiments)

    frames: List[Image.Image] = []

    for step_idx in range(max_steps):
        cell_images: List[Image.Image] = []

        for exp in experiments:
            steps_list = exp["steps"]
            if step_idx < len(steps_list):
                step_data = steps_list[step_idx]
            else:
                step_data = steps_list[-1] if steps_list else None

            if step_data is not None:
                plotter = _render_cell(step_data, bin_config, cell_title=exp.get("label", ""), window_size=(2400, 800))
                try:
                    img = _plotter_to_pil(plotter)
                    cell_images.append(img)
                except Exception as e:
                    print(f"  Warning: Could not render cell: {e}")
                    cell_images.append(Image.new('RGB', (2400, 800), (255, 255, 255)))
            else:
                cell_images.append(Image.new('RGB', (2400, 800), (255, 255, 255)))

        from visualization.grid_creator import _assemble_grid
        grid = _assemble_grid(cell_images, cols=grid_cols)
        
        # Removed Pil ImageDraw that adds text headers to the top of the grid
        frames.append(grid)
        
        if step_idx % 5 == 0 or step_idx == max_steps - 1:
            print(f"    Grid GIF frame {step_idx + 1}/{max_steps}")

    if not frames:
        print("  Warning: No frames rendered. Grid GIF not created.")
        return save_path

    ms_per_frame = int(1000 / fps)
    durations = [ms_per_frame] * len(frames)
    durations[-1] = ms_per_frame * 4

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  Grid GIF saved: {save_path}  ({len(frames)} frames, {grid_rows}x{grid_cols} grid)")
    return save_path
