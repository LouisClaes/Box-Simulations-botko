"""

Interactive 3D viewer — Plotly-based.

"""



import plotly.graph_objects as go

import colorsys

from typing import List

from config import Placement, BinConfig



def render_interactive_3d(placements: List[Placement], bin_config: BinConfig, save_path: str):

    """Genereert een interactieve HTML viewer."""

    fig = go.Figure()

    sorted_p = sorted(placements, key=lambda p: p.step)

   

    for p in sorted_p:

        hue = (p.step * 0.13) % 1.0

        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)

        color = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'

       

        # Mesh voor de doos

        fig.add_trace(go.Mesh3d(

            x=[p.x, p.x, p.x+p.oriented_w, p.x+p.oriented_w, p.x, p.x, p.x+p.oriented_w, p.x+p.oriented_w],

            y=[p.y, p.y+p.oriented_l, p.y+p.oriented_l, p.y, p.y, p.y+p.oriented_l, p.y+p.oriented_l, p.y],

            z=[p.z, p.z, p.z, p.z, p.z+p.oriented_h, p.z+p.oriented_h, p.z+p.oriented_h, p.z+p.oriented_h],

            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],

            color=color, opacity=1.0, flatshading=True

        ))

       

        # Lijnen voor de randen

        x_e = [p.x, p.x+p.oriented_w, p.x+p.oriented_w, p.x, p.x, None, p.x, p.x+p.oriented_w, p.x+p.oriented_w, p.x, p.x, None, p.x, p.x, None, p.x+p.oriented_w, p.x+p.oriented_w, None, p.x+p.oriented_w, p.x+p.oriented_w, None, p.x, p.x]

        y_e = [p.y, p.y, p.y+p.oriented_l, p.y+p.oriented_l, p.y, None, p.y, p.y, p.y+p.oriented_l, p.y+p.oriented_l, p.y, None, p.y, p.y, None, p.y, p.y, None, p.y+p.oriented_l, p.y+p.oriented_l, None, p.y+p.oriented_l, p.y+p.oriented_l]

        z_e = [p.z, p.z, p.z, p.z, p.z, None, p.z+p.oriented_h, p.z+p.oriented_h, p.z+p.oriented_h, p.z+p.oriented_h, p.z+p.oriented_h, None, p.z, p.z+p.oriented_h, None, p.z, p.z+p.oriented_h, None, p.z, p.z+p.oriented_h, None, p.z, p.z+p.oriented_h]

        fig.add_trace(go.Scatter3d(x=x_e, y=y_e, z=z_e, mode='lines', line=dict(color='black', width=2), showlegend=False))



    fig.update_layout(

        scene=dict(

            xaxis=dict(range=[0, bin_config.width], title="X (Width)"),

            yaxis=dict(range=[0, bin_config.length], title="Y (Length)"),

            zaxis=dict(range=[0, bin_config.height], title="Z (Height)"),

            aspectmode='data'

        )

    )

    fig.write_html(save_path)