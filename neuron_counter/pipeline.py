"""
pipeline
========
Orchestrate the non-parametric counter:

    load CZI
      -> extract an intensity-weighted 3D point cloud (micron space)
      -> model it as a 3D Gaussian-mixture density (KDE by default)
      -> count the modes (local maxima) of that density
      -> write outputs (count, per-neuron centroid table, QC overlays)

The neuron count is the number of density modes -- no per-plane segmentation
and no instance stitching.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import foreground as _fg
from . import gmm_modes as _gmm
from . import visualize
from . import viz3d
from .io_czi import Stack, load_czi


@dataclass
class PipelineResult:
    stack: Stack
    marker_index: int
    n_neurons: int
    modes: "_gmm.ModeResult"
    info: Dict
    metrics: pd.DataFrame
    output_dir: Path
    timings: Dict[str, float]

    def summary(self) -> Dict:
        return {
            "file": str(self.stack.path),
            "voxel_um_zyx": list(self.stack.voxel_um),
            "marker_channel_index": self.marker_index,
            "marker_channel_name": self.stack.channel_names[self.marker_index],
            "method": f"non-parametric 3D Gaussian mixture ({self.modes.method}), "
                      f"mode-counting",
            "n_neurons": self.n_neurons,
            "n_seeds": self.modes.n_seeds,
            "n_excluded_small": self.modes.n_excluded_small,
            "n_excluded_large": self.modes.n_excluded_large,
            "median_diameter_um": (round(float(np.nanmedian(self.modes.diameter_um)), 2)
                                   if self.modes.n_modes else None),
            "fit_info": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in self.info.items()},
            "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
        }


def run_pipeline(path: str, cfg: Dict,
                 output_dir: Optional[str] = None,
                 progress=None) -> PipelineResult:
    """``progress(frac: float, message: str)`` is called at stage boundaries
    (and per ensemble run) so a GUI can show advancement. Optional."""
    def report(frac: float, msg: str) -> None:
        if progress is not None:
            try:
                progress(float(frac), str(msg))
            except Exception:
                pass

    t: Dict[str, float] = {}
    ch = cfg.get("channels", {})
    marker_idx = int(ch.get("neuron_marker_index", 0))
    rng_state = int(cfg.get("random_state", 0))

    # ---- load ---------------------------------------------------------- #
    report(0.04, "Loading image…")
    t0 = time.time()
    fb = cfg.get("voxel_um", {})
    stack = load_czi(
        path,
        channel_names=ch.get("names"),
        voxel_um_fallback=(fb.get("z", 1.0), fb.get("y", 0.69), fb.get("x", 0.69)),
    )
    if marker_idx >= stack.n_channels:
        raise IndexError(
            f"neuron_marker_index={marker_idx} but file has only "
            f"{stack.n_channels} channels"
        )
    marker = stack.channel(marker_idx)
    t["load"] = time.time() - t0

    # ---- foreground point cloud ---------------------------------------- #
    report(0.15, "Extracting bright foreground…")
    t0 = time.time()
    fgp = cfg.get("foreground", {})
    fparams = _fg.ForegroundParams(
        smooth_sigma_um=fgp.get("smooth_sigma_um", 1.0),
        threshold_method=fgp.get("threshold_method", "otsu"),
        threshold_percentile=fgp.get("threshold_percentile", 98.0),
        threshold_absolute=fgp.get("threshold_absolute", 0.0),
        min_foreground_frac=fgp.get("min_foreground_frac", 0.0005),
        max_points=fgp.get("max_points", 20000),
        weight_by_intensity=fgp.get("weight_by_intensity", True),
        random_state=rng_state,
    )
    pc = _fg.extract_points(marker, stack.voxel_um, fparams)
    t["foreground"] = time.time() - t0

    # ---- model density + count modes ----------------------------------- #
    report(0.25, "Counting neurons…")
    t0 = time.time()
    gp = cfg.get("gmm", {})
    mp = cfg.get("modes", {})
    sz = cfg.get("size", {})
    cparams = _gmm.CountParams(
        bandwidth_um=gp.get("bandwidth_um", 3.0),
        seed_min_distance_um=gp.get("seed_min_distance_um", 4.0),
        seed_threshold_rel=gp.get("seed_threshold_rel", 0.0),
        meanshift_max_iter=mp.get("meanshift_max_iter", 150),
        tol_um=mp.get("tol_um", 0.02),
        merge_radius_um=mp.get("merge_radius_um"),
        min_mode_density_rel=mp.get("min_mode_density_rel", 0.0),
        min_diameter_um=sz.get("min_diameter_um", 8.0),
        max_diameter_um=sz.get("max_diameter_um", 35.0),
        size_assign_cap_um=sz.get("size_assign_cap_um"),
        random_state=rng_state,
    )
    rb = cfg.get("robust", {})
    robust_on = rb.get("enabled", True)
    if robust_on:
        rparams = _gmm.RobustParams(
            enabled=True,
            n_runs=rb.get("n_runs", 25),
            noise_model=rb.get("noise_model", "gaussian"),
            noise_scale=rb.get("noise_scale", 1.0),
            threshold_jitter=rb.get("threshold_jitter", 0.08),
            bandwidth_jitter=rb.get("bandwidth_jitter", 0.15),
            consensus_radius_um=rb.get("consensus_radius_um", 4.0),
            selection_threshold=rb.get("selection_threshold", 0.5),
            random_state=rng_state,
        )
        modes = _gmm.consensus_count(
            marker, stack.voxel_um, fparams, cparams, rparams,
            on_run=lambda b, n: report(0.25 + 0.6 * b / n,
                                       f"Counting neurons… ensemble run {b}/{n}"))
        info = {
            "ensemble_runs": rparams.n_runs,
            # The per-run spread is a bootstrap over real sources of variation,
            # not a single hand-tuned jitter -- see uncertainty_sources.
            "uncertainty_sources": ["measurement_noise", "threshold",
                                     "subsample", "bandwidth"],
            "bandwidth_um": cparams.bandwidth_um,
            "noise_model": rparams.noise_model,
            "noise_scale": rparams.noise_scale,
            "threshold_jitter": rparams.threshold_jitter,
            "bandwidth_jitter": rparams.bandwidth_jitter,
            "per_run_count_mean": round(modes.count_mean, 1),
            "per_run_count_std": round(modes.count_std, 1),
            "per_run_count_p2.5_97.5": [round(modes.count_ci[0], 1),
                                        round(modes.count_ci[1], 1)],
            "selection_threshold": modes.selection_threshold,
            "n_candidates": modes.n_candidates,
            "n_stable_ge0.8": modes.n_stable,
            "n_marginal": modes.n_marginal,
        }
    else:
        modes, info = _gmm.count_modes(pc.coords_um, pc.weights, pc.smoothed,
                                       pc.threshold, stack.voxel_um, cparams,
                                       fg_coords_um=pc.fg_coords_um,
                                       fg_coords_vox=pc.fg_coords_vox)
    t["count_modes"] = time.time() - t0
    report(0.9, "Saving results table…")

    # ---- metrics table ------------------------------------------------- #
    metrics = _build_metrics(modes, marker)

    # ---- outputs ------------------------------------------------------- #
    out = cfg.get("output", {})
    out_dir = Path(output_dir or out.get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if out.get("save_metrics_csv", True):
        metrics.to_csv(out_dir / "neurons.csv", index=False)
    if out.get("save_overlays", True):
        report(0.92, "Rendering overlay images…")
        name = stack.channel_names[marker_idx]
        visualize.overlay_mip(marker, modes.centroids_vox, out_dir / "overlay.png",
                              title=f"{name}: {modes.n_modes} neurons (GMM modes)")
        visualize.z_montage(marker, modes.centroids_vox, out_dir / "zmontage.png",
                            title=f"{name} mode centroids per z")
    if out.get("save_html_3d", True):
        report(0.94, "Building 3D viewer…")
        name = stack.channel_names[marker_idx]
        try:
            viz3d.write_3d_html(
                pc.coords_um, pc.weights, modes.centroids_um, modes.diameter_um,
                out_dir / "neurons_3d.html",
                title=f"{name}: {modes.n_modes} neurons (3D GMM modes)",
                random_state=rng_state,
            )
        except ImportError:
            pass  # plotly not installed -> skip the interactive view

    # ---- label-free QC: coverage audit + z-slice viewer ---------------- #
    # Regenerated every run so neurons.csv, the audit, and the viewer stay in
    # sync. The viewer overlays both the detections and the audit's candidate
    # misses. Reuses the already-loaded marker (no extra CZI read).
    name = stack.channel_names[marker_idx]
    if out.get("save_coverage_audit", True):
        report(0.96, "Running coverage audit (QC)…")
        try:
            from . import qc_coverage as _qc
            _qc.run_audit(path, str(out_dir), cfg, _qc.CoverageParams(),
                          marker=marker, voxel_um=stack.voxel_um,
                          channel_name=name,
                          save_png=out.get("save_coverage_png", False))
        except Exception as e:  # never let QC break a successful count
            print(f"[warn] coverage audit skipped: {type(e).__name__}: {e}")
    if out.get("save_zviewer", True):
        report(0.98, "Building z-slice viewer…")
        try:
            from . import viz_zslice as _vz
            _vz.generate_viewer(
                path, results_dir=str(out_dir),
                output=str(out_dir / "zviewer.html"), cfg=cfg,
                marker_index=marker_idx, verbose=False,
                run_audit_if_missing=False,  # audit already ran above
                marker=marker, voxel_um=stack.voxel_um,
                channel_name=name, n_z=stack.n_z)
        except Exception as e:
            print(f"[warn] z-viewer skipped: {type(e).__name__}: {e}")

    result = PipelineResult(stack, marker_idx, modes.n_modes, modes, info,
                            metrics, out_dir, t)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result.summary(), f, indent=2)
    report(1.0, f"Done — {modes.n_modes} neurons")
    return result


def _build_metrics(modes, intensity: np.ndarray) -> pd.DataFrame:
    # consensus result carries a per-neuron `stability`; single-run carries
    # `density` + `n_components`. Build the right columns for each.
    is_consensus = hasattr(modes, "stability")
    if is_consensus:
        cols = ["neuron_id", "centroid_z", "centroid_y", "centroid_x",
                "z_um", "y_um", "x_um", "diameter_um", "stability", "intensity"]
        score = modes.stability
    else:
        cols = ["neuron_id", "centroid_z", "centroid_y", "centroid_x",
                "z_um", "y_um", "x_um", "diameter_um", "rel_density",
                "n_seeds_merged", "intensity"]
        score = modes.density
    if modes.n_modes == 0:
        return pd.DataFrame(columns=cols)

    Z, Y, X = intensity.shape
    cv = modes.centroids_vox
    order = np.argsort(-score)
    rows = []
    for new_id, i in enumerate(order, start=1):
        cz, cy, cx = cv[i]
        zi = int(np.clip(round(cz), 0, Z - 1))
        yi = int(np.clip(round(cy), 0, Y - 1))
        xi = int(np.clip(round(cx), 0, X - 1))
        row = dict(
            neuron_id=new_id,
            centroid_z=round(float(cz), 2),
            centroid_y=round(float(cy), 2),
            centroid_x=round(float(cx), 2),
            z_um=round(float(modes.centroids_um[i, 0]), 2),
            y_um=round(float(modes.centroids_um[i, 1]), 2),
            x_um=round(float(modes.centroids_um[i, 2]), 2),
            diameter_um=round(float(modes.diameter_um[i]), 2),
            intensity=round(float(intensity[zi, yi, xi]), 1),
        )
        if is_consensus:
            row["stability"] = round(float(modes.stability[i]), 3)
        else:
            row["rel_density"] = round(float(modes.density[i]), 4)
            row["n_seeds_merged"] = int(modes.n_components[i])
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)
