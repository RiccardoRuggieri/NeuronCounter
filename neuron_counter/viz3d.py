"""
viz3d
=====
Interactive 3D QC viewer (standalone HTML, opens in any browser).

Renders the foreground tissue as a faint point cloud coloured by intensity and
the detected neuron centroids as red markers sized by their measured diameter,
so the result can be eye-checked **in 3D** (rotate / zoom) rather than only on a
2D projection. A button toggles between true micron scale (the stack is a thin
~6 µm slab) and a z-exaggerated view that makes the depth separation visible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

VoxelUM = Tuple[float, float, float]


def write_3d_html(points_um: np.ndarray, point_vals: np.ndarray,
                  centroids_um: np.ndarray, diameters: np.ndarray,
                  out_path: Path, title: str = "",
                  max_points: int = 25000, random_state: int = 0) -> None:
    import plotly.graph_objects as go

    pts = np.asarray(points_um, dtype=float)
    vals = np.asarray(point_vals, dtype=float)

    # sub-sample tissue points for a responsive browser scene
    if len(pts) > max_points:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(pts), size=max_points, replace=False)
        pts, vals = pts[sel], vals[sel]

    # columns are (z, y, x) microns
    pz, py, px = pts[:, 0], pts[:, 1], pts[:, 2]

    tissue = go.Scatter3d(
        x=px, y=py, z=pz, mode="markers", name="HuC/D signal",
        marker=dict(size=1.6, color=vals, colorscale="Greys", opacity=0.35,
                    showscale=False, reversescale=True),
        hoverinfo="skip",
    )

    traces = [tissue]
    if len(centroids_um):
        cz, cy, cx = (centroids_um[:, 0], centroids_um[:, 1], centroids_um[:, 2])
        diam = np.asarray(diameters, dtype=float)
        # marker size scaled to diameter (clipped for legibility)
        msize = np.clip(diam * 0.45, 4, 16)
        text = [f"neuron {i + 1}<br>Ø {d:.1f} µm<br>z={z:.1f} µm"
                for i, (d, z) in enumerate(zip(diam, cz))]
        centroids = go.Scatter3d(
            x=cx, y=cy, z=cz, mode="markers", name=f"neurons (n={len(cx)})",
            marker=dict(size=msize, color="#ff3b30", opacity=0.95,
                        line=dict(width=0.5, color="#7a0000")),
            text=text, hoverinfo="text",
        )
        traces.append(centroids)

    # aspect ratios: true scale vs z-exaggerated
    rx = float(np.ptp(px)) or 1.0
    ry = float(np.ptp(py)) or 1.0
    rz = float(np.ptp(pz)) or 1.0
    M = max(rx, ry)
    true_ar = dict(x=rx / M, y=ry / M, z=max(rz / M, 1e-3))
    exag_ar = dict(x=rx / M, y=ry / M, z=0.5)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title="z (µm)",
            aspectmode="manual", aspectratio=exag_ar,
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        ),
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.01, y=0.02, xanchor="left",
            buttons=[
                dict(label="Z exaggerated", method="relayout",
                     args=[{"scene.aspectratio": exag_ar}]),
                dict(label="Z to scale", method="relayout",
                     args=[{"scene.aspectratio": true_ar}]),
            ],
        )],
    )
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
