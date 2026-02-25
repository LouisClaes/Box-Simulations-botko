"""
Botko Visualizer — rich animated GIF for pallet packing with conveyor.

Produces presentation-ready GIFs showing:
  - Left:   Pallet 0 (3D PyVista)
  - Center: Conveyor belt (4-box pick window, no overlaps)
  - Right:  Pallet 1 (3D PyVista)
  - Bottom: Info bar (Pillow text) with strategy, step, fills, close logic

Requires: pyvista, numpy, Pillow
"""

import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

from config import Placement, BinConfig, Box
from visualization.render_3d import _get_box_color


# ─────────────────────────────────────────────────────────────────────────────
# BotkoStep — enriched step record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BotkoStep:
    """One animation frame with rich context for the Botko visualizer."""
    step: int
    total_steps: int
    box: Box
    bin_index: int
    placed: bool
    placement: Optional[Placement]
    buffer_snapshot: List[Box]
    stream_remaining: int
    pallet0_placements: List[Placement]
    pallet1_placements: List[Placement]
    # Rich context
    strategy_name: str = ""
    action: str = ""              # "placed on pallet 0" / "rejected"
    consecutive_rejects: int = 0
    pallets_closed_so_far: int = 0
    pallet0_fill: float = 0.0
    pallet1_fill: float = 0.0
    close_policy_desc: str = "Close fullest after 4 global rejects (min fill: 50%)"
    total_placed: int = 0
    total_rejected: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# 3D Panel rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_pallet(plotter: pv.Plotter, placements: List[Placement],
                 bin_config: BinConfig, label: str = ""):
    """Draw pallet wireframe + placed boxes."""
    p_len, p_wid, p_hei = bin_config.length, bin_config.width, bin_config.height
    z_limit = max(p_hei, 3000) if p_hei > 2000 else p_hei

    bbox = pv.Box(bounds=(0, p_len, 0, p_wid, 0, z_limit))
    plotter.add_mesh(bbox, style='wireframe', color='gray', line_width=1.0, opacity=0.3)

    n_x = int(p_len / 200) + 1
    n_y = int(p_wid / 200) + 1
    n_z = int(z_limit / 200) + 1

    for p in placements:
        box_mesh = pv.Box(bounds=(
            p.x, p.x + p.oriented_l,
            p.y, p.y + p.oriented_w,
            p.z, p.z + p.oriented_h,
        ))
        color = _get_box_color(p.step)
        plotter.add_mesh(
            box_mesh, color=color, show_edges=True, edge_color='black',
            line_width=1.0, specular=0.1, ambient=0.4, diffuse=0.8,
        )

    plotter.show_bounds(
        grid='back', location='outer', ticks='inside',
        axes_ranges=[0, p_len, 0, p_wid, 0, z_limit],
        show_xlabels=True, show_ylabels=True, show_zlabels=True,
        xtitle='X', ytitle='Y', ztitle='Z',
        color='black', font_size=10, font_family='arial',
        n_xlabels=n_x, n_ylabels=n_y, n_zlabels=n_z,
    )

    plotter.camera_position = 'iso'
    plotter.camera.azimuth = -100
    plotter.camera.elevation = 5
    plotter.camera.zoom(0.9)


def _draw_conveyor_belt(plotter: pv.Plotter, buffer_boxes: List[Box],
                        highlight_box_id: Optional[int] = None):
    """Draw conveyor belt with 4 pick-window boxes, no overlaps."""
    BELT_LENGTH = 1200.0
    BELT_WIDTH = 400.0
    MAX_SCENE_Z = 700.0
    DISPLAY_SLOTS = 4
    SLOT_PADDING = 0.85

    frame = pv.Box(bounds=(0, BELT_LENGTH, 0, BELT_WIDTH, 0, MAX_SCENE_Z))
    plotter.add_mesh(frame, style='wireframe', color='lightgray', line_width=0.5, opacity=0.15)

    belt = pv.Box(bounds=(0, BELT_LENGTH, 0, BELT_WIDTH, -10, 0))
    plotter.add_mesh(belt, color='slategray', opacity=0.5, show_edges=True, edge_color='dimgray')

    visible = buffer_boxes[:DISPLAY_SLOTS] if buffer_boxes else []
    if visible:
        slot_width = BELT_LENGTH / DISPLAY_SLOTS
        max_box_w = slot_width * SLOT_PADDING

        for i, box in enumerate(visible):
            scale = min(1.0, max_box_w / max(box.length, 1.0))
            vis_l = box.length * scale
            vis_w = box.width * scale
            vis_h = box.height * scale

            x_center = slot_width * i + slot_width / 2
            x_pos = x_center - vis_l / 2
            y_pos = (BELT_WIDTH - vis_w) / 2

            is_hl = highlight_box_id is not None and box.id == highlight_box_id
            edge = "red" if is_hl else "black"
            lw = 6.0 if is_hl else 1.0
            color = _get_box_color(box.id)

            b = pv.Box(bounds=(
                x_pos, x_pos + vis_l,
                y_pos, y_pos + vis_w,
                0.0, vis_h,
            ))
            plotter.add_mesh(
                b, color=color, show_edges=True, edge_color=edge, line_width=lw,
                specular=0.1, ambient=0.4, diffuse=0.8,
            )

    plotter.camera_position = 'iso'
    plotter.camera.azimuth = -100
    plotter.camera.elevation = 5
    plotter.camera.zoom(0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Info bar (Pillow)
# ─────────────────────────────────────────────────────────────────────────────

def _get_font(size: int = 16):
    """Get a decent font, fall back to default."""
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _draw_info_bar(step: BotkoStep, width: int, bar_height: int = 160) -> Image.Image:
    """Render a text info bar as a PIL image."""
    img = Image.new('RGB', (width, bar_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_lg = _get_font(22)
    font_md = _get_font(17)
    font_sm = _get_font(14)

    y = 8
    left_margin = 20

    # Row 1: Strategy name + step
    strategy_display = step.strategy_name.replace("_", " ").title()
    draw.text((left_margin, y), f"Strategy: {strategy_display}", fill='#1a1a2e', font=font_lg)
    draw.text((width // 2, y), f"Step {step.step}/{step.total_steps}", fill='#333', font=font_lg)
    draw.text((width * 3 // 4, y), f"Remaining: {step.stream_remaining}", fill='#555', font=font_md)
    y += 32

    # Row 2: Box info + action
    box = step.box
    box_dims = f"{box.length:.0f}×{box.width:.0f}×{box.height:.0f}mm"
    draw.text((left_margin, y), f"Box #{box.id} ({box_dims})", fill='#333', font=font_md)

    action_color = '#2d6a4f' if step.placed else '#d62828'
    draw.text((width // 3, y), f"→ {step.action}", fill=action_color, font=font_md)
    y += 28

    # Row 3: Fill rates + close logic
    p0_bar = f"Pallet 0: {step.pallet0_fill:.1%} ({len(step.pallet0_placements)} boxes)"
    p1_bar = f"Pallet 1: {step.pallet1_fill:.1%} ({len(step.pallet1_placements)} boxes)"
    draw.text((left_margin, y), p0_bar, fill='#1a759f', font=font_md)
    draw.text((width // 3, y), p1_bar, fill='#1a759f', font=font_md)

    rejects_color = '#d62828' if step.consecutive_rejects >= 3 else '#555'
    draw.text((width * 2 // 3, y), f"Consec. rejects: {step.consecutive_rejects}/4",
              fill=rejects_color, font=font_md)
    y += 28

    # Row 4: Stats summary
    stats = (f"Placed: {step.total_placed}  |  Rejected: {step.total_rejected}  |  "
             f"Pallets closed: {step.pallets_closed_so_far}")
    draw.text((left_margin, y), stats, fill='#555', font=font_sm)
    draw.text((width * 2 // 3, y), f"Close: {step.close_policy_desc}", fill='#777', font=font_sm)

    # Separator line at top
    draw.line([(0, 0), (width, 0)], fill='#ccc', width=2)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_botko_frame(
    step: BotkoStep,
    bin_config: BinConfig,
    window_size: Tuple[int, int] = (3000, 1000),
) -> Image.Image:
    """Render one complete frame: 3-panel 3D + info bar."""
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, border=False)
    plotter.set_background('white')
    plotter.window_size = window_size
    plotter.enable_lightkit()

    # Left: Pallet 0
    plotter.subplot(0, 0)
    _draw_pallet(plotter, step.pallet0_placements, bin_config)

    # Center: Conveyor
    plotter.subplot(0, 1)
    hl_id = step.box.id if step.placed else None
    _draw_conveyor_belt(plotter, step.buffer_snapshot, highlight_box_id=hl_id)

    # Right: Pallet 1
    plotter.subplot(0, 2)
    _draw_pallet(plotter, step.pallet1_placements, bin_config)

    # Render 3D to PIL
    img_array = plotter.screenshot(None, return_img=True)
    plotter.close()
    panel_img = Image.fromarray(img_array)

    # Add info bar below
    info_bar = _draw_info_bar(step, panel_img.width)
    combined = Image.new('RGB', (panel_img.width, panel_img.height + info_bar.height), (255, 255, 255))
    combined.paste(panel_img, (0, 0))
    combined.paste(info_bar, (0, panel_img.height))

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_botko_gif(
    steps: List[BotkoStep],
    bin_config: BinConfig,
    save_path: str,
    fps: int = 3,
) -> str:
    """Render a full Botko-style animated GIF from a list of BotkoSteps.

    Args:
        steps:      List of BotkoStep records (one per frame).
        bin_config: Pallet configuration (1200×800×2700).
        save_path:  Output GIF path.
        fps:        Frames per second.

    Returns:
        The save_path.
    """
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    frames: List[Image.Image] = []

    for i, step in enumerate(steps):
        try:
            frame = _render_botko_frame(step, bin_config)
            frames.append(frame)
        except Exception as e:
            print(f"  Warning: Could not render frame {i}: {e}")

        if (i + 1) % 10 == 0 or i == len(steps) - 1:
            print(f"    Frame {i + 1}/{len(steps)}")

    if not frames:
        print("  Warning: No frames rendered. GIF not created.")
        return save_path

    ms_per_frame = int(1000 / fps)
    durations = [ms_per_frame] * len(frames)
    durations[-1] = ms_per_frame * 4  # Hold last frame longer

    frames[0].save(
        save_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0,
    )
    print(f"  GIF saved: {save_path}  ({len(frames)} frames)")
    return save_path
