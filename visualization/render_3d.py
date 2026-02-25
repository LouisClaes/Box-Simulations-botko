"""
3D visualization — PyVista-based rendering.

Renders the packing result as a clean 3D scene with:
  - White background
  - Black metric coordinates/text with explicit mm grid
  - Colored solid boxes with distinct qualitative colors
  - Uniform true 3D scaling
  - 200mm grid intervals
"""

import os
import pyvista as pv
import matplotlib.cm as cm
import numpy as np
from typing import List, Tuple, Optional

from config import Placement, BinConfig

VIBRANT_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
]

def _get_box_color(step: int) -> str:
    """Distinct vibrant colors matching Plotly/D3 style."""
    return VIBRANT_COLORS[step % len(VIBRANT_COLORS)]

def _configure_plotter(
    placements: List[Placement],
    bin_config: BinConfig,
    title: str = "",
    off_screen: bool = True
) -> pv.Plotter:
    """
    Build a pyvista Plotter with the 3D packing scene.
    Background: White. Text/Lines: Black.
    Proportional uniform axis scaling.
    """
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background('white')

    p_len, p_wid, p_hei = bin_config.length, bin_config.width, bin_config.height
    z_limit = max(p_hei, 3000) if p_hei > 2000 else p_hei

    # Bounding box wireframe outline
    bounds = (0, p_len, 0, p_wid, 0, z_limit)
    bbox = pv.Box(bounds=bounds)
    plotter.add_mesh(bbox, style='wireframe', color='gray', line_width=1.0, opacity=0.3)

    # Calculate tick counts based on 200mm intervals
    # e.g., length 1200 / 200 = 6 + 1 = 7 ticks
    n_x = int(p_len / 200) + 1
    n_y = int(p_wid / 200) + 1
    n_z = int(z_limit / 200) + 1

    # Plot Boxes
    for p in placements:
        xmin = p.x
        xmax = p.x + p.oriented_l
        ymin = p.y
        ymax = p.y + p.oriented_w
        zmin = p.z
        zmax = p.z + p.oriented_h
        
        box_mesh = pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
        color = _get_box_color(p.step)
        
        plotter.add_mesh(
            box_mesh,
            color=color,
            show_edges=True,
            edge_color='black',
            line_width=1.0,
            specular=0.1,    # bright lighting
            ambient=0.4,
            diffuse=0.8
        )

    # Show grid/axes
    plotter.show_bounds(
        grid='back',
        location='outer',
        ticks='inside',
        axes_ranges=[0, p_len, 0, p_wid, 0, z_limit],
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        xtitle='X (mm)',
        ytitle='Y (mm)',
        ztitle='Z (mm)',
        color='black',
        font_size=12,
        font_family='arial',
        n_xlabels=n_x,
        n_ylabels=n_y,
        n_zlabels=n_z,
    )

    if title:
        plotter.add_text(title, position='upper_edge', color='black', font_size=14, font='arial')

    # Global lighting for brighter scene
    plotter.enable_lightkit()
    
    # Camera instellingen: Isometrisch
    plotter.camera_position = 'iso'
    plotter.camera.azimuth = -100
    plotter.camera.elevation = 5
    plotter.camera.zoom(0.9)

    return plotter

def get_figure(
    placements: List[Placement],
    bin_config: BinConfig,
    title: str = "",
) -> pv.Plotter:
    """
    Backwards compatibility method returning a configured plotter instance instead of plt.Figure.
    """
    return _configure_plotter(placements, bin_config, title=title, off_screen=True)

def render_packing(
    placements: List[Placement],
    bin_config: BinConfig,
    save_path: str,
    title: str = "",
    resolution: Tuple[int, int] = (3840, 2160),
) -> str:
    """Render the packing result as a PNG image using PyVista."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plotter = _configure_plotter(placements, bin_config, title=title, off_screen=True)
    plotter.window_size = resolution
    
    try:
        plotter.screenshot(save_path, return_img=False)
    except Exception as e:
        print(f"  Warning: Could not save image: {e}")
    finally:
        plotter.close()

    return save_path

def render_step_sequence(
    placements: List[Placement],
    bin_config: BinConfig,
    output_dir: str,
    resolution: Tuple[int, int] = (2560, 1440),
) -> str:
    """Render one PNG per placement step using PyVista."""
    os.makedirs(output_dir, exist_ok=True)
    sorted_placements = sorted(placements, key=lambda p: p.step)

    for i in range(1, len(sorted_placements) + 1):
        current = sorted_placements[:i]
        plotter = _configure_plotter(current, bin_config, title="", off_screen=True)
        plotter.window_size = resolution
        
        path = os.path.join(output_dir, f"step_{i - 1:02d}.png")
        try:
            plotter.screenshot(path, return_img=False)
        except Exception as e:
            print(f"  Warning: Could not save step {i}: {e}")
        finally:
            plotter.close()

    return output_dir
