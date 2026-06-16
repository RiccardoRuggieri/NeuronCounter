"""
gmm_modes
=========
The heart of the counter: model the bright-voxel point cloud as a 3D Gaussian
**kernel-density estimate** (KDE) and **count its modes** (local maxima). One
mode == one soma.

The KDE is the textbook *non-parametric* density estimator -- one isotropic
Gaussian of bandwidth ``h`` per data point -- so there is no "number of
components" to choose; the number of neurons emerges purely as the number of
modes. Modes are found by climbing the density from a set of seeds (the local
intensity maxima) with a Gaussian mean-shift fixed-point iteration, then merging
coincident maxima. The count is the number of distinct, sufficiently-dense modes
that pass the soma-size filter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

VoxelUM = Tuple[float, float, float]


@dataclass
class CountParams:
    # --- KDE mixture --------------------------------------------------------
    bandwidth_um: float = 3.0           # Gaussian kernel bandwidth (~soma radius)

    # --- seeds (local maxima used as mean-shift start points) --------------
    seed_min_distance_um: float = 4.0
    seed_threshold_rel: float = 0.0     # extra relative floor above foreground thr

    # --- mean-shift mode finding -------------------------------------------
    meanshift_max_iter: int = 150
    tol_um: float = 0.02
    merge_radius_um: Optional[float] = None   # None -> 0.5 * bandwidth_um
    min_mode_density_rel: float = 0.0   # drop modes dimmer than this * median density

    # --- soma size filter (in-plane / XY equivalent diameter, microns) ------
    # The stack is thin in z, so a 15-25 um soma is truncated in depth; size is
    # therefore measured as the XY footprint equivalent diameter.
    min_diameter_um: float = 8.0        # exclude objects smaller than this
    max_diameter_um: float = 35.0       # exclude objects larger than this
    size_assign_cap_um: Optional[float] = None  # None -> max_diameter_um

    random_state: int = 0


@dataclass
class ModeResult:
    centroids_um: np.ndarray   # (M, 3)
    centroids_vox: np.ndarray  # (M, 3)
    density: np.ndarray        # (M,) mixture density at each mode (normalised)
    weight_mass: np.ndarray    # (M,) summed seed/component mass merged into mode
    n_components: np.ndarray   # (M,) number of seeds merged into mode
    diameter_um: np.ndarray    # (M,) in-plane (XY) equivalent diameter
    n_modes: int
    n_seeds: int
    n_excluded_small: int
    n_excluded_large: int
    method: str


# --------------------------------------------------------------------------- #
# Seeds: local intensity maxima
# --------------------------------------------------------------------------- #
def _seed_maxima(smoothed: np.ndarray, threshold: float, voxel_um: VoxelUM,
                 p: CountParams) -> np.ndarray:
    from skimage.feature import peak_local_max
    vz, vy, vx = voxel_um
    fp = (max(1, int(round(p.seed_min_distance_um / vz))),
          max(1, int(round(p.seed_min_distance_um / vy))),
          max(1, int(round(p.seed_min_distance_um / vx))))
    thr_abs = threshold * (1.0 + p.seed_threshold_rel)
    peaks = peak_local_max(smoothed, footprint=np.ones(fp, dtype=bool),
                           threshold_abs=thr_abs, exclude_border=False)
    return peaks.astype(np.float64)  # (S, 3) voxel coords


# --------------------------------------------------------------------------- #
# Union-find merge of coincident converged seeds
# --------------------------------------------------------------------------- #
def _merge_labels(points: np.ndarray, radius: float) -> np.ndarray:
    from scipy.spatial import cKDTree
    n = len(points)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n and radius > 0:
        for a, b in cKDTree(points).query_pairs(radius):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


# --------------------------------------------------------------------------- #
# Mixture 1: non-parametric Gaussian KDE
# --------------------------------------------------------------------------- #
class _KDE:
    def __init__(self, points_um, weights, bandwidth):
        from scipy.spatial import cKDTree
        self.P = np.ascontiguousarray(points_um, dtype=np.float64)
        self.w = np.asarray(weights, dtype=np.float64)
        self.w = self.w / self.w.sum()
        self.h = float(bandwidth)
        self.r = 3.0 * self.h
        self.tree = cKDTree(self.P)

    def density(self, X: np.ndarray) -> np.ndarray:
        h2 = self.h * self.h
        nbr = self.tree.query_ball_point(X, self.r, workers=-1)
        out = np.zeros(len(X))
        for i, idx in enumerate(nbr):
            if idx:
                idx = np.asarray(idx)
                d2 = ((self.P[idx] - X[i]) ** 2).sum(1)
                out[i] = (self.w[idx] * np.exp(-d2 / (2 * h2))).sum()
        return out

    def mean_shift(self, seeds, max_iter, tol):
        h2 = self.h * self.h
        Y = np.array(seeds, dtype=np.float64)
        active = np.ones(len(Y), dtype=bool)
        for _ in range(max_iter):
            if not active.any():
                break
            ai = np.where(active)[0]
            nbr = self.tree.query_ball_point(Y[ai], self.r, workers=-1)
            for j, idx in zip(ai, nbr):
                if not idx:
                    active[j] = False
                    continue
                idx = np.asarray(idx)
                d2 = ((self.P[idx] - Y[j]) ** 2).sum(1)
                k = self.w[idx] * np.exp(-d2 / (2 * h2))
                s = k.sum()
                if s <= 0:
                    active[j] = False
                    continue
                ynew = (k[:, None] * self.P[idx]).sum(0) / s
                if np.linalg.norm(ynew - Y[j]) < tol:
                    active[j] = False
                Y[j] = ynew
        return Y


# --------------------------------------------------------------------------- #
# Per-mode soma size: in-plane (XY) equivalent diameter
# --------------------------------------------------------------------------- #
def _mode_diameters(centroids_um, fg_coords_um, fg_coords_vox,
                    voxel_um, cap_um):
    """Assign each foreground voxel to its nearest mode (within ``cap_um``) and
    return the XY-footprint equivalent diameter (microns) of every mode."""
    from scipy.spatial import cKDTree
    M = len(centroids_um)
    diam = np.zeros(M)
    if M == 0 or len(fg_coords_um) == 0:
        return diam
    vz, vy, vx = voxel_um
    pix_area = vy * vx
    dist, assign = cKDTree(centroids_um).query(fg_coords_um, workers=-1)
    within = dist < cap_um
    yx = fg_coords_vox[:, 1:].astype(np.int64)
    # encode (y, x) into one integer for fast unique-per-mode counting
    xmax = int(yx[:, 1].max()) + 1 if len(yx) else 1
    codes = yx[:, 0] * xmax + yx[:, 1]
    for m in range(M):
        sel = within & (assign == m)
        if not sel.any():
            continue
        n_pix = np.unique(codes[sel]).size
        area = n_pix * pix_area
        diam[m] = 2.0 * np.sqrt(area / np.pi)
    return diam


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def count_modes(points_um: np.ndarray, weights: np.ndarray,
                smoothed: np.ndarray, threshold: float, voxel_um: VoxelUM,
                params: CountParams,
                fg_coords_um: Optional[np.ndarray] = None,
                fg_coords_vox: Optional[np.ndarray] = None):
    """Return (ModeResult, info-dict)."""
    vz, vy, vx = voxel_um
    seeds_vox = _seed_maxima(smoothed, threshold, voxel_um, params)
    seeds_um = seeds_vox * np.array([vz, vy, vx])
    n_seeds = len(seeds_um)
    info = {"n_seeds": n_seeds}

    mix = _KDE(points_um, weights, params.bandwidth_um)
    info.update(bandwidth_um=params.bandwidth_um)
    merge = params.merge_radius_um or 0.5 * params.bandwidth_um

    if len(seeds_um) == 0:
        z = np.zeros((0, 3))
        return (ModeResult(z, z, np.zeros(0), np.zeros(0), np.zeros(0, int),
                           np.zeros(0), 0, n_seeds, 0, 0, "kde"), info)

    converged = mix.mean_shift(seeds_um, params.meanshift_max_iter, params.tol_um)
    labels = _merge_labels(converged, merge)
    n_modes = int(labels.max()) + 1

    dens_seeds = mix.density(converged)
    centroids = np.zeros((n_modes, 3))
    dens = np.zeros(n_modes)
    ncomp = np.zeros(n_modes, dtype=int)
    for m in range(n_modes):
        sel = np.where(labels == m)[0]
        best = sel[np.argmax(dens_seeds[sel])]
        centroids[m] = converged[best]
        dens[m] = dens_seeds[best]
        ncomp[m] = len(sel)

    # --- density-based pruning of faint spurious modes ------------------- #
    good = np.ones(n_modes, dtype=bool)
    if params.min_mode_density_rel > 0 and n_modes:
        good &= dens >= params.min_mode_density_rel * np.median(dens)
    centroids, dens, ncomp = centroids[good], dens[good], ncomp[good]

    # --- soma size filter (in-plane equivalent diameter) ---------------- #
    cap = params.size_assign_cap_um or params.max_diameter_um
    if fg_coords_um is not None and fg_coords_vox is not None:
        diam = _mode_diameters(centroids, fg_coords_um, fg_coords_vox,
                               voxel_um, cap)
    else:
        diam = np.full(len(centroids), np.nan)
    too_small = diam < params.min_diameter_um
    too_large = diam > params.max_diameter_um
    n_small = int(np.count_nonzero(too_small))
    n_large = int(np.count_nonzero(too_large))
    keep = ~(too_small | too_large)
    centroids, dens, ncomp, diam = (centroids[keep], dens[keep],
                                    ncomp[keep], diam[keep])

    # normalise density to a 0..1 readout for the metrics table
    dmax = dens.max() if len(dens) else 1.0
    dnorm = dens / dmax if dmax > 0 else dens

    centroids_vox = centroids / np.array([vz, vy, vx])
    res = ModeResult(
        centroids_um=centroids, centroids_vox=centroids_vox,
        density=dnorm, weight_mass=dens, n_components=ncomp, diameter_um=diam,
        n_modes=int(len(centroids)), n_seeds=n_seeds,
        n_excluded_small=n_small, n_excluded_large=n_large, method="kde",
    )
    return res, info


# --------------------------------------------------------------------------- #
# Robust counting: ensemble consensus / stability selection
# --------------------------------------------------------------------------- #
@dataclass
class RobustParams:
    """Controls the uncertainty bootstrap.

    Each run re-derives the whole detection from a perturbed copy of the image,
    so the count spread reflects the genuine sources of fragility rather than a
    single arbitrary knob:

      * ``noise_*``         -- a fresh acquisition-noise realization (its scale
                               estimated from the image itself), re-thresholded
                               and re-seeded, i.e. measurement uncertainty.
      * ``threshold_jitter``-- the foreground threshold (Otsu is not exact), which
                               propagates into both the foreground and the seeds.
      * ``bandwidth_jitter``-- the KDE scale (a modelling choice ~ soma radius).

    The subsample also varies run-to-run (each run reseeds the foreground RNG).
    """
    enabled: bool = True
    n_runs: int = 25                   # ensemble size
    noise_model: str = "gaussian"      # "gaussian" | "poisson" | "none"
    noise_scale: float = 1.0           # inject this * estimated noise-floor each run
    threshold_jitter: float = 0.08     # foreground threshold varied +/- this fraction
    bandwidth_jitter: float = 0.15     # bandwidth varied by +/- this fraction
    consensus_radius_um: float = 4.0   # match detections across runs (< soma spacing)
    selection_threshold: float = 0.5   # keep neurons seen in >= this fraction of runs
    random_state: int = 0


@dataclass
class ConsensusResult:
    centroids_um: np.ndarray
    centroids_vox: np.ndarray
    diameter_um: np.ndarray
    stability: np.ndarray              # selection frequency of each kept neuron
    n_modes: int                       # robust (consensus) count
    per_run_counts: np.ndarray
    count_mean: float
    count_std: float
    count_ci: Tuple[float, float]      # 2.5 / 97.5 percentile of per-run counts
    n_stable: int                      # neurons with frequency >= 0.8
    n_marginal: int                    # threshold <= frequency < 0.8
    n_candidates: int                  # all consensus clusters before selection
    selection_threshold: float
    n_runs: int
    method: str = "kde-consensus"
    # carried so the rest of the pipeline (metrics/summary) is unchanged:
    n_seeds: int = 0
    n_excluded_small: int = 0
    n_excluded_large: int = 0


def consensus_count(marker: np.ndarray, voxel_um: VoxelUM,
                    fparams, base_params: CountParams, robust: RobustParams,
                    on_run=None) -> ConsensusResult:
    """Re-detect modes over an ensemble that bootstraps the *genuine* sources of
    uncertainty, then keep neurons by how often they are re-detected (stability
    selection).

    Each run rebuilds the detection end-to-end from a perturbed image:

        marker + noise realization  (scale estimated from the data)
          -> re-smooth, re-threshold (threshold jittered)  -> seeds & foreground vary
          -> intensity-weighted subsample (reseeded)        -> point cloud varies
          -> KDE with jittered bandwidth                     -> scale varies
          -> count_modes

    so the per-run count spread measures how much the answer moves under
    realistic measurement noise and defensible parameter wobble -- not just a
    single hand-picked bandwidth jitter on an otherwise frozen pipeline.
    """
    from dataclasses import replace as _replace
    from . import foreground as _fg

    vz, vy, vx = voxel_um
    base_bw = base_params.bandwidth_um
    vol = np.asarray(marker, dtype=np.float32)

    # Noise floor estimated from the image itself (data-driven, not a free knob).
    if robust.noise_model != "none" and robust.noise_scale > 0:
        sigma = _fg.estimate_noise_sigma(vol, voxel_um, fparams.smooth_sigma_um)
    else:
        sigma = 0.0

    all_c, all_d, all_run, per_run = [], [], [], []
    for b in range(robust.n_runs):
        rng = np.random.default_rng(robust.random_state + 1 + b)

        # (a) measurement-noise realization
        if sigma > 0 and robust.noise_model == "gaussian":
            vol_b = vol + rng.normal(0.0, sigma * robust.noise_scale,
                                     vol.shape).astype(np.float32)
        elif robust.noise_model == "poisson":
            lam = np.clip(vol, 0.0, None) * robust.noise_scale
            vol_b = (rng.poisson(lam).astype(np.float32)
                     / max(robust.noise_scale, 1e-9))
        else:
            vol_b = vol

        # (b) threshold jitter + (c) fresh intensity-weighted subsample, both via
        # a full re-extraction -> the foreground AND the mean-shift seeds vary.
        ts = 1.0 + robust.threshold_jitter * (2.0 * rng.random() - 1.0)
        fp_b = _replace(fparams, random_state=int(robust.random_state + 1 + b),
                        threshold_scale=float(ts))
        pc = _fg.extract_points(vol_b, voxel_um, fp_b)

        # (d) bandwidth jitter (modelling-scale uncertainty)
        bw = base_bw * (1.0 + robust.bandwidth_jitter * (2.0 * rng.random() - 1.0))
        prun = replace(base_params, bandwidth_um=float(bw))

        res, _ = count_modes(pc.coords_um, pc.weights, pc.smoothed, pc.threshold,
                             voxel_um, prun, fg_coords_um=pc.fg_coords_um,
                             fg_coords_vox=pc.fg_coords_vox)
        per_run.append(res.n_modes)
        if res.n_modes:
            all_c.append(res.centroids_um)
            all_d.append(res.diameter_um)
            all_run.append(np.full(res.n_modes, b))
        if on_run is not None:
            try:
                on_run(b + 1, robust.n_runs)
            except Exception:
                pass

    per_run = np.asarray(per_run, dtype=float)
    if not all_c:
        z = np.zeros((0, 3))
        return ConsensusResult(z, z, np.zeros(0), np.zeros(0), 0, per_run,
                               0.0, 0.0, (0.0, 0.0), 0, 0, 0,
                               robust.selection_threshold, robust.n_runs)

    C = np.vstack(all_c)
    D = np.concatenate(all_d)
    R = np.concatenate(all_run)
    labels = _merge_labels(C, robust.consensus_radius_um)
    n_clusters = int(labels.max()) + 1

    cent = np.zeros((n_clusters, 3))
    diam = np.zeros(n_clusters)
    freq = np.zeros(n_clusters)
    for m in range(n_clusters):
        sel = labels == m
        cent[m] = C[sel].mean(0)
        diam[m] = np.median(D[sel])
        freq[m] = np.unique(R[sel]).size / robust.n_runs

    keep = freq >= robust.selection_threshold
    order = np.argsort(-freq[keep])
    kc, kd, kf = cent[keep][order], diam[keep][order], freq[keep][order]

    return ConsensusResult(
        centroids_um=kc,
        centroids_vox=kc / np.array([vz, vy, vx]),
        diameter_um=kd,
        stability=kf,
        n_modes=int(keep.sum()),
        per_run_counts=per_run,
        count_mean=float(per_run.mean()),
        count_std=float(per_run.std()),
        count_ci=(float(np.percentile(per_run, 2.5)),
                  float(np.percentile(per_run, 97.5))),
        n_stable=int((kf >= 0.8).sum()),
        n_marginal=int(((kf >= robust.selection_threshold) & (kf < 0.8)).sum()),
        n_candidates=n_clusters,
        selection_threshold=robust.selection_threshold,
        n_runs=robust.n_runs,
    )
