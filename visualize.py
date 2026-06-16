"""
visualize
=========
Lightweight QC images: a maximum-intensity projection of the marker channel
with the detected mode centroids overlaid, plus a per-z montage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _norm(img: np.ndarray, lo=1.0, hi=99.7) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        b = a + 1.0
    return np.clip((img - a) / (b - a), 0, 1)


def overlay_mip(vol: np.ndarray, centroids_vox: np.ndarray, out_path: Path,
                title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mip = vol.max(axis=0)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(_norm(mip), cmap="gray", interpolation="nearest")
    if len(centroids_vox):
        ax.scatter(centroids_vox[:, 2], centroids_vox[:, 1], s=26,
                   facecolors="none", edgecolors="#ff3b30", linewidths=0.8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def z_montage(vol: np.ndarray, centroids_vox: np.ndarray, out_path: Path,
              title: str = "", cols: int = 3, panel_in: float = 6.0,
              dpi: int = 200) -> None:
    """High-resolution per-z montage. Each centroid is drawn on its nearest
    z-plane (so every neuron appears exactly once)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = vol.shape[0]
    cols = min(Z, cols)
    rows = int(np.ceil(Z / cols))
    nearest = (np.round(centroids_vox[:, 0]).astype(int)
               if len(centroids_vox) else np.array([], dtype=int))
    fig, axes = plt.subplots(rows, cols, figsize=(panel_in * cols, panel_in * rows),
                             squeeze=False)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        ax.axis("off")
        if k >= Z:
            continue
        ax.imshow(_norm(vol[k]), cmap="gray", interpolation="bicubic")
        if len(centroids_vox):
            on = nearest == k
            if on.any():
                ax.scatter(centroids_vox[on, 2], centroids_vox[on, 1],
                           s=70, facecolors="none", edgecolors="#ff3b30",
                           linewidths=1.0)
        ax.set_title(f"z = {k}  ({int((nearest == k).sum())} neurons)",
                     fontsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
