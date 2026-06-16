"""
Command-line entry point.

    python -m model INPUT.czi [options]

Examples
--------
    # Count with defaults (KDE mode-counting):
    python -m model sample.czi

    # The main accuracy knob is the bandwidth (~soma radius, microns):
    python -m model sample.czi --bandwidth-um 2.5

    # Trade accuracy for speed (fewer sampled points):
    python -m model sample.czi --max-points 12000

    # Just print file metadata and exit:
    python -m model sample.czi --info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .pipeline import run_pipeline

_DEFAULT_CFG = {
    "random_state": 0,
    "channels": {"neuron_marker_index": 0,
                 "names": ["HuC_D", "Ch1", "Ch2", "Ch3"]},
    "voxel_um": {"x": 0.6905, "y": 0.6905, "z": 1.0},
    "foreground": {"smooth_sigma_um": 1.0, "threshold_method": "otsu",
                   "max_points": 20000, "weight_by_intensity": True},
    "gmm": {"bandwidth_um": 6.0, "seed_min_distance_um": 8.0},
    "modes": {"merge_radius_um": None, "meanshift_max_iter": 150,
              "tol_um": 0.02, "min_mode_density_rel": 0.0},
    "size": {"min_diameter_um": 8.0, "max_diameter_um": 35.0},
    "robust": {"enabled": True, "n_runs": 25, "noise_model": "gaussian",
               "noise_scale": 1.0, "threshold_jitter": 0.08,
               "bandwidth_jitter": 0.15, "consensus_radius_um": 4.0,
               "selection_threshold": 0.5},
    "output": {"dir": "results", "save_overlays": False, "save_metrics_csv": True,
               "save_html_3d": False, "save_coverage_audit": True,
               "save_coverage_png": False, "save_zviewer": True},
}


def _load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    default = Path(__file__).with_name("config.yaml")
    if default.exists():
        with open(default) as f:
            return yaml.safe_load(f) or {}
    return _DEFAULT_CFG


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="model",
        description="Count HuC/D+ neurons in a 3D .czi z-stack by fitting a "
                    "non-parametric Gaussian mixture in 3D and counting its "
                    "density modes.",
    )
    p.add_argument("input", help="path to the .czi file")
    p.add_argument("-c", "--config", help="path to config.yaml")
    p.add_argument("-o", "--output", help="output directory (overrides config)")
    p.add_argument("--marker-index", type=int, help="override marker channel index")
    p.add_argument("--bandwidth-um", type=float,
                   help="KDE bandwidth ~soma radius (larger = fewer neurons)")
    p.add_argument("--seed-min-distance-um", type=float,
                   help="min separation of seed maxima")
    p.add_argument("--merge-radius-um", type=float,
                   help="merge converged modes closer than this (microns)")
    p.add_argument("--min-diameter-um", type=float,
                   help="exclude somata smaller than this (in-plane diameter)")
    p.add_argument("--max-diameter-um", type=float,
                   help="exclude somata larger than this (in-plane diameter)")
    p.add_argument("--max-points", type=int,
                   help="number of foreground points fed to the density")
    p.add_argument("--threshold-method",
                   choices=["otsu", "percentile", "triangle", "absolute"],
                   help="foreground threshold method")
    p.add_argument("--no-robust", action="store_true",
                   help="disable the ensemble-consensus counting (single run)")
    p.add_argument("--no-qc", action="store_true",
                   help="skip the coverage audit and z-slice viewer outputs")
    p.add_argument("--n-runs", type=int,
                   help="ensemble size for robust counting (default 25)")
    p.add_argument("--info", action="store_true",
                   help="print file metadata and exit")
    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    if args.marker_index is not None:
        cfg.setdefault("channels", {})["neuron_marker_index"] = args.marker_index
    if args.max_points is not None:
        cfg.setdefault("foreground", {})["max_points"] = args.max_points
    if args.threshold_method is not None:
        cfg.setdefault("foreground", {})["threshold_method"] = args.threshold_method
    if args.bandwidth_um is not None:
        cfg.setdefault("gmm", {})["bandwidth_um"] = args.bandwidth_um
    if args.seed_min_distance_um is not None:
        cfg.setdefault("gmm", {})["seed_min_distance_um"] = args.seed_min_distance_um
    if args.merge_radius_um is not None:
        cfg.setdefault("modes", {})["merge_radius_um"] = args.merge_radius_um
    if args.no_robust:
        cfg.setdefault("robust", {})["enabled"] = False
    if args.no_qc:
        cfg.setdefault("output", {})["save_coverage_audit"] = False
        cfg.setdefault("output", {})["save_zviewer"] = False
    if args.n_runs is not None:
        cfg.setdefault("robust", {})["n_runs"] = args.n_runs
    if args.min_diameter_um is not None:
        cfg.setdefault("size", {})["min_diameter_um"] = args.min_diameter_um
    if args.max_diameter_um is not None:
        cfg.setdefault("size", {})["max_diameter_um"] = args.max_diameter_um

    if args.info:
        from .io_czi import load_czi
        ch = cfg.get("channels", {})
        fb = cfg.get("voxel_um", {})
        stack = load_czi(args.input, channel_names=ch.get("names"),
                         voxel_um_fallback=(fb.get("z", 1.0), fb.get("y", 0.69),
                                            fb.get("x", 0.69)))
        print(stack.describe())
        return 0

    result = run_pipeline(args.input, cfg, output_dir=args.output)

    s = result.summary()
    print("\n=== Neuron count (GMM mode-counting) ===")
    print(f"File     : {Path(args.input).name}")
    print(f"Marker   : ch{result.marker_index} "
          f"({result.stack.channel_names[result.marker_index]})")
    print(f"Method   : {result.modes.method}")
    if hasattr(result.modes, "stability"):
        m = result.modes
        print(f"Neurons  : {result.n_neurons}   "
              f"(consensus of {m.n_runs} runs; per-run {m.count_mean:.0f}"
              f"±{m.count_std:.0f}, 95% bootstrap band "
              f"{m.count_ci[0]:.0f}-{m.count_ci[1]:.0f})")
        print(f"Stability: {m.n_stable} stable (>=0.8), {m.n_marginal} marginal; "
              f"median diameter {s['median_diameter_um']} um")
    else:
        print(f"Neurons  : {result.n_neurons}")
        print(f"Excluded : {result.modes.n_excluded_small} too small, "
              f"{result.modes.n_excluded_large} too large  "
              f"(median diameter {s['median_diameter_um']} um)")
    print(f"Timings s: {s['timings_s']}")
    print(f"Outputs  : {result.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
