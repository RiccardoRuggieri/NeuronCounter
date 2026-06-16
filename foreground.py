"""
foreground
==========
Turn a 3D marker volume into an intensity-weighted **3D point cloud** that the
Gaussian-mixture stage will model.

Each foreground voxel becomes a point, positioned in **physical micron space**
``(z*vz, y*vy, x*vx)`` so the fitted Gaussians live in an isotropic metric and
are not distorted by anisotropic voxels. Points are weighted by intensity (so
brighter soma cores carry more mass) and sub-sampled to keep the EM fit fast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

VoxelUM = Tuple[float, float, float]


@dataclass
class ForegroundParams:
    smooth_sigma_um: float = 1.0        # gaussian denoise sigma (microns); 0 disables
    threshold_method: str = "otsu"      # 'otsu' | 'percentile' | 'triangle' | 'absolute'
    threshold_percentile: float = 98.0  # used when method == 'percentile'
    threshold_absolute: float = 0.0     # used when method == 'absolute'
    min_foreground_frac: float = 0.0005  # safety floor on fraction kept
    max_points: int = 40000             # sub-sample target for the point cloud
    weight_by_intensity: bool = True    # sample/weight points by intensity
    threshold_scale: float = 1.0        # multiply the computed threshold (ensemble jitter)
    random_state: int = 0


@dataclass
class PointCloud:
    coords_um: np.ndarray   # (N, 3) float32, (z, y, x) in microns (sub-sampled)
    coords_vox: np.ndarray  # (N, 3) float32, (z, y, x) in voxel index units
    weights: np.ndarray     # (N,) float32
    threshold: float        # intensity threshold used
    n_foreground: int       # foreground voxels before sub-sampling
    smoothed: np.ndarray    # (Z, Y, X) smoothed volume (for seeding / QC)
    fg_coords_um: np.ndarray   # (M, 3) ALL foreground voxels, microns (for sizing)
    fg_coords_vox: np.ndarray  # (M, 3) ALL foreground voxels, voxel indices
    fg_weights: np.ndarray     # (M,) ALL foreground voxel weights (intensity-thr)


def _threshold(vals: np.ndarray, vol: np.ndarray, p: ForegroundParams) -> float:
    method = p.threshold_method.lower()
    if method == "absolute":
        return float(p.threshold_absolute)
    if method == "percentile":
        return float(np.percentile(vol, p.threshold_percentile))
    try:
        from skimage import filters
        if method == "triangle":
            return float(filters.threshold_triangle(vol))
        return float(filters.threshold_otsu(vol))
    except Exception:
        return float(np.percentile(vol, p.threshold_percentile))


def extract_points(vol: np.ndarray, voxel_um: VoxelUM,
                   params: ForegroundParams) -> PointCloud:
    vz, vy, vx = voxel_um
    vol = vol.astype(np.float32, copy=False)

    # --- denoise --------------------------------------------------------- #
    if params.smooth_sigma_um and params.smooth_sigma_um > 0:
        sigma = (params.smooth_sigma_um / vz,
                 params.smooth_sigma_um / vy,
                 params.smooth_sigma_um / vx)
        smoothed = ndi.gaussian_filter(vol, sigma=sigma)
    else:
        smoothed = vol

    # --- threshold to foreground ---------------------------------------- #
    thr = _threshold(smoothed, smoothed, params)
    # Optional reproducible jitter of the threshold (used by the robust
    # ensemble to propagate threshold uncertainty into the foreground & seeds).
    if params.threshold_scale != 1.0:
        thr = float(thr * params.threshold_scale)
    mask = smoothed > thr

    # Guarantee a minimum amount of signal (handles a too-aggressive Otsu).
    frac = mask.mean()
    if frac < params.min_foreground_frac:
        thr = float(np.percentile(smoothed, 100 * (1 - params.min_foreground_frac)))
        mask = smoothed > thr

    idx = np.argwhere(mask)                      # (M, 3) z,y,x voxel indices
    if idx.size == 0:
        raise RuntimeError("No foreground voxels found; lower the threshold.")
    inten = smoothed[mask].astype(np.float32)
    n_fg = int(idx.shape[0])

    # Keep the FULL foreground (for per-soma size estimation and for the
    # ensemble, which draws a fresh intensity-weighted subsample each run).
    scale = np.array([vz, vy, vx], dtype=np.float32)
    fg_coords_vox = idx.astype(np.float32)
    fg_coords_um = fg_coords_vox * scale
    fg_weights = (inten - thr)
    fg_weights[fg_weights < 0] = 0.0
    if not params.weight_by_intensity or fg_weights.sum() == 0:
        fg_weights = np.ones(n_fg, dtype=np.float32)
    fg_weights = fg_weights.astype(np.float32)

    # --- sub-sample for speed ------------------------------------------- #
    rng = np.random.default_rng(params.random_state)
    if n_fg > params.max_points:
        if params.weight_by_intensity:
            w = inten - thr
            w[w < 0] = 0.0
            s = w.sum()
            prob = (w / s) if s > 0 else None
        else:
            prob = None
        sel = rng.choice(n_fg, size=params.max_points, replace=False, p=prob)
        idx = idx[sel]
        inten = inten[sel]

    coords_vox = idx.astype(np.float32)
    coords_um = coords_vox * np.array([vz, vy, vx], dtype=np.float32)
    weights = (inten - thr)
    weights[weights < 0] = 0.0
    if not params.weight_by_intensity or weights.sum() == 0:
        weights = np.ones(len(inten), dtype=np.float32)

    return PointCloud(coords_um=coords_um, coords_vox=coords_vox,
                      weights=weights.astype(np.float32), threshold=float(thr),
                      n_foreground=n_fg, smoothed=smoothed,
                      fg_coords_um=fg_coords_um, fg_coords_vox=fg_coords_vox,
                      fg_weights=fg_weights)


def estimate_noise_sigma(vol: np.ndarray, voxel_um: VoxelUM,
                         smooth_sigma_um: float = 1.0) -> float:
    """Data-driven estimate of the per-voxel acquisition-noise std.

    Uses the high-frequency residual ``vol - smooth(vol)`` and a robust MAD
    estimator (``sigma ~= 1.4826 * MAD``). This grounds the ensemble's noise
    perturbation in the image's own noise floor instead of an arbitrary knob,
    so the resulting count spread reflects real measurement uncertainty.
    """
    vz, vy, vx = voxel_um
    vol = vol.astype(np.float32, copy=False)
    s = smooth_sigma_um if (smooth_sigma_um and smooth_sigma_um > 0) else 1.0
    sm = ndi.gaussian_filter(vol, sigma=(s / vz, s / vy, s / vx))
    resid = vol - sm
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return 1.4826 * mad
