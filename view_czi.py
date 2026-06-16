#!/usr/bin/env python3
"""
view_czi.py
===========
Quickly *look* at a .czi z-stack -- independent of the counting pipeline.

It writes PNGs you can open directly:
  * one maximum-intensity projection (MIP) per channel,
  * a combined grid of all channels,
  * a per-z montage of one channel (the marker by default),

and prints the file's dimensions / voxel size / channel names.

Usage
-----
    python view_czi.py sample.czi
    python view_czi.py sample.czi -o preview --montage-channel 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse the package's robust CZI loader (numpy-only import).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.io_czi import load_czi


def _stretch(plane: np.ndarray, p_lo=1.0, p_hi=99.5) -> np.ndarray:
    """Percentile contrast stretch to [0, 1] for display."""
    plane = plane.astype(np.float32)
    lo, hi = np.percentile(plane, [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((plane - lo) / (hi - lo), 0.0, 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render a .czi z-stack to PNGs.")
    ap.add_argument("input", help="path to the .czi file")
    ap.add_argument("-o", "--output", default="preview", help="output folder")
    ap.add_argument("--montage-channel", type=int, default=0,
                    help="channel index for the per-z montage (default 0 = marker)")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stack = load_czi(args.input)
    print(stack.describe())

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    C = stack.n_channels

    # --- per-channel MIPs ------------------------------------------------- #
    for c in range(C):
        mip = _stretch(stack.channel(c).max(axis=0))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mip, cmap="gray")
        ax.set_title(f"ch{c}: {stack.channel_names[c]}  (MIP)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"mip_ch{c}_{stack.channel_names[c]}.png", dpi=150)
        plt.close(fig)

    # --- all-channels grid ------------------------------------------------ #
    cols = min(C, 4)
    rows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < C:
            ax.imshow(_stretch(stack.channel(i).max(axis=0)), cmap="gray")
            ax.set_title(f"ch{i}: {stack.channel_names[i]}", fontsize=9)
    fig.suptitle(f"{stack.path.name} — channel MIPs")
    fig.tight_layout()
    fig.savefig(out / "mip_all_channels.png", dpi=150)
    plt.close(fig)

    # --- per-z montage of one channel ------------------------------------- #
    mc = max(0, min(args.montage_channel, C - 1))
    vol = stack.channel(mc)
    Z = vol.shape[0]
    cols = min(Z, 6)
    rows = int(np.ceil(Z / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows),
                             squeeze=False)
    for z in range(rows * cols):
        ax = axes[z // cols][z % cols]
        ax.axis("off")
        if z < Z:
            ax.imshow(_stretch(vol[z]), cmap="gray")
            ax.set_title(f"z={z}", fontsize=8)
    fig.suptitle(f"{stack.channel_names[mc]} — per-z planes")
    fig.tight_layout()
    fig.savefig(out / f"zmontage_ch{mc}_{stack.channel_names[mc]}.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote PNGs to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
