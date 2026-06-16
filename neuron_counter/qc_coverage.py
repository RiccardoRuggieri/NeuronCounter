"""
qc_coverage
===========
A-posteriori accuracy audit of a finished run **without any hand annotation**.

Idea (independent-detector logic)
----------------------------------
The counter detects neurons as *modes of a Gaussian-mixture density*. To check
it for misses we use a deliberately **different** signal -- the raw bright
HuC/D+ foreground itself -- and ask the dual question the user posed:

    "Is there any neuron-looking blob that has NO digital annotation on it?"

Because the two views (density modes vs. raw bright blobs) fail in different
ways, their disagreement is informative and bounds the error *without* ground
truth. Concretely:

1. Rebuild the bright foreground mask (same threshold the pipeline used).
2. Label its 3D connected components.
3. For each accepted centroid (read back from ``neurons.csv``) find which
   component it lands in. A component that owns >= 1 centroid is *covered*.
4. A soma-sized component that owns **zero** centroids is a **candidate miss**
   (false negative). Its area / a typical soma area estimates how many neurons
   it might hold.
5. A centroid that lands on background / a dim sliver of foreground is a
   **candidate false positive**.

Outputs a QC overlay highlighting both, a JSON summary (coverage fraction, the
candidate counts, an implied recall band), and a CSV of the candidate misses to
eyeball. None of this needs a human-labelled mask; it is a triage of suspects
plus an error envelope, not a single trusted accuracy number.

CLI
---
    python -m neuron_counter.qc_coverage sample.czi --results results
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import foreground as _fg
from .io_czi import load_czi

VoxelUM = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
@dataclass
class CoverageParams:
    # how far (microns) a centroid may sit from a foreground voxel and still be
    # counted as "on" that component (tolerance for sub-voxel / rounding drift)
    centroid_snap_um: float = 3.0
    # an uncovered foreground component must be at least this XY-diameter to be a
    # candidate missed soma (smaller = debris). Defaults to size.min_diameter_um.
    min_miss_diameter_um: Optional[float] = None
    # and no larger than this many times a typical soma (guards against one huge
    # tissue-spanning component being read as a single miss); only used for the
    # per-component miss-count estimate.
    typical_soma_diameter_um: Optional[float] = None  # default: median detected


@dataclass
class CoverageResult:
    n_detected: int
    n_fg_voxels: int
    coverage_fraction: float            # fraction of fg voxels in covered comps
    coverage_fraction_intensity: float  # same, weighted by intensity mass
    n_components: int
    n_covered_components: int
    n_uncovered_soma_components: int
    n_candidate_misses: int             # estimated missed neurons (area-based)
    n_candidate_false_positives: int
    recall_lower: float                 # detected / (detected + candidate misses)
    miss_table: pd.DataFrame
    fp_centroids_vox: np.ndarray
    detected_centroids_vox: np.ndarray
    miss_centroids_vox: np.ndarray


# --------------------------------------------------------------------------- #
def _foreground_mask(marker: np.ndarray, voxel_um: VoxelUM,
                     fgp: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (boolean fg mask, smoothed volume, threshold) using the SAME
    foreground settings the pipeline ran with."""
    params = _fg.ForegroundParams(
        smooth_sigma_um=fgp.get("smooth_sigma_um", 1.0),
        threshold_method=fgp.get("threshold_method", "otsu"),
        threshold_percentile=fgp.get("threshold_percentile", 98.0),
        threshold_absolute=fgp.get("threshold_absolute", 0.0),
        min_foreground_frac=fgp.get("min_foreground_frac", 0.0005),
        # we only need the mask/smoothed/threshold, so sub-sampling is irrelevant
        max_points=10**9,
        weight_by_intensity=fgp.get("weight_by_intensity", True),
        random_state=int(fgp.get("random_state", 0)),
    )
    pc = _fg.extract_points(marker, voxel_um, params)
    mask = np.zeros(marker.shape, dtype=bool)
    vox = pc.fg_coords_vox.astype(np.int64)
    mask[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    return mask, pc.smoothed, pc.threshold


def coverage_audit(marker: np.ndarray, voxel_um: VoxelUM,
                   centroids_vox: np.ndarray, fgp: Dict,
                   params: CoverageParams,
                   median_detected_diam_um: Optional[float] = None
                   ) -> CoverageResult:
    vz, vy, vx = voxel_um
    Z, Y, X = marker.shape
    pix_area = vy * vx

    mask, smoothed, thr = _foreground_mask(marker, voxel_um, fgp)
    n_fg = int(mask.sum())

    # 3D connected components (26-connectivity).
    labels, n_comp = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))

    # --- snap each centroid onto a component --------------------------------
    # A centroid is "on" component L if its voxel is in L, or the nearest
    # foreground voxel within `centroid_snap_um` is in L.
    cvox = np.asarray(centroids_vox, dtype=float)
    n_det = len(cvox)
    comp_of_centroid = np.zeros(n_det, dtype=np.int64)  # 0 == background / none

    # distance transform gives, for every background voxel, the index of the
    # nearest foreground voxel -> lets us snap centroids that land just off-mask.
    snap_vox = max(params.centroid_snap_um / vz,
                   params.centroid_snap_um / vy,
                   params.centroid_snap_um / vx)
    inds = ndi.distance_transform_edt(~mask, sampling=(vz, vy, vx),
                                      return_distances=True,
                                      return_indices=True)
    dist_bg, nearest_idx = inds
    for i, (cz, cy, cx) in enumerate(cvox):
        zi = int(np.clip(round(cz), 0, Z - 1))
        yi = int(np.clip(round(cy), 0, Y - 1))
        xi = int(np.clip(round(cx), 0, X - 1))
        lab = labels[zi, yi, xi]
        if lab == 0:
            # not on foreground: snap to nearest fg voxel if close enough
            if dist_bg[zi, yi, xi] <= params.centroid_snap_um:
                nz = nearest_idx[0][zi, yi, xi]
                ny = nearest_idx[1][zi, yi, xi]
                nx = nearest_idx[2][zi, yi, xi]
                lab = labels[nz, ny, nx]
        comp_of_centroid[i] = lab

    centroids_per_comp = np.bincount(comp_of_centroid, minlength=n_comp + 1)
    centroids_per_comp[0] = 0  # ignore background bucket

    # --- per-component geometry --------------------------------------------
    # voxel counts and XY footprint (unique y,x) per label
    flat_lab = labels.ravel()
    vox_count = np.bincount(flat_lab, minlength=n_comp + 1)
    # intensity mass per component (for intensity-weighted coverage)
    inten_mass = ndi.sum(smoothed, labels, index=np.arange(n_comp + 1))

    # XY footprint per component: encode (y,x) and count uniques per label
    zz, yy, xx = np.nonzero(mask)
    lab_fg = labels[zz, yy, xx]
    code = yy.astype(np.int64) * (X + 1) + xx.astype(np.int64)
    # unique (label, yx) pairs
    order = np.lexsort((code, lab_fg))
    lab_s, code_s = lab_fg[order], code[order]
    new_pair = np.ones(len(lab_s), dtype=bool)
    new_pair[1:] = (lab_s[1:] != lab_s[:-1]) | (code_s[1:] != code_s[:-1])
    xy_count = np.bincount(lab_s[new_pair], minlength=n_comp + 1)
    comp_diam_um = 2.0 * np.sqrt((xy_count * pix_area) / np.pi)

    # --- coverage fraction --------------------------------------------------
    covered = centroids_per_comp > 0          # per-label boolean
    covered[0] = False
    covered_voxels = int(vox_count[1:][covered[1:]].sum())
    coverage_fraction = covered_voxels / n_fg if n_fg else 0.0
    total_mass = float(inten_mass[1:].sum())
    covered_mass = float(inten_mass[1:][covered[1:]].sum())
    coverage_fraction_int = covered_mass / total_mass if total_mass else 0.0

    # --- candidate misses: uncovered, soma-sized components -----------------
    min_miss_d = (params.min_miss_diameter_um
                  if params.min_miss_diameter_um is not None else 8.0)
    typ_d = (params.typical_soma_diameter_um
             or median_detected_diam_um or 19.0)
    typ_area = np.pi * (typ_d / 2.0) ** 2

    miss_rows: List[dict] = []
    n_miss_estimate = 0
    n_uncov_soma = 0
    for L in range(1, n_comp + 1):
        if covered[L]:
            continue
        d = comp_diam_um[L]
        if d < min_miss_d:
            continue
        n_uncov_soma += 1
        area = xy_count[L] * pix_area
        est = max(1, int(round(area / typ_area)))
        n_miss_estimate += est
        # component centroid (voxel) for plotting/CSV
        sel = labels == L
        cz, cy, cx = (np.array(np.nonzero(sel)).mean(axis=1))
        miss_rows.append(dict(
            component_id=int(L),
            centroid_z=round(float(cz), 1),
            centroid_y=round(float(cy), 1),
            centroid_x=round(float(cx), 1),
            xy_diameter_um=round(float(d), 1),
            n_voxels=int(vox_count[L]),
            est_neurons=est,
            peak_intensity=round(float(smoothed[sel].max()), 1),
        ))
    miss_table = pd.DataFrame(
        miss_rows,
        columns=["component_id", "centroid_z", "centroid_y", "centroid_x",
                 "xy_diameter_um", "n_voxels", "est_neurons", "peak_intensity"],
    )
    miss_centroids = (miss_table[["centroid_z", "centroid_y", "centroid_x"]]
                      .to_numpy() if len(miss_table) else np.zeros((0, 3)))

    # --- candidate false positives: centroid on background / tiny sliver ----
    fp_mask = np.zeros(n_det, dtype=bool)
    for i in range(n_det):
        L = comp_of_centroid[i]
        if L == 0:
            fp_mask[i] = True                      # not on any foreground
        elif comp_diam_um[L] < min_miss_d and centroids_per_comp[L] == 1:
            fp_mask[i] = True                      # alone on a sub-soma sliver
    fp_centroids = cvox[fp_mask] if n_det else np.zeros((0, 3))

    recall_lower = (n_det / (n_det + n_miss_estimate)
                    if (n_det + n_miss_estimate) else 1.0)

    return CoverageResult(
        n_detected=n_det,
        n_fg_voxels=n_fg,
        coverage_fraction=coverage_fraction,
        coverage_fraction_intensity=coverage_fraction_int,
        n_components=int(n_comp),
        n_covered_components=int(covered[1:].sum()),
        n_uncovered_soma_components=n_uncov_soma,
        n_candidate_misses=n_miss_estimate,
        n_candidate_false_positives=int(fp_mask.sum()),
        recall_lower=recall_lower,
        miss_table=miss_table,
        fp_centroids_vox=fp_centroids,
        detected_centroids_vox=cvox,
        miss_centroids_vox=miss_centroids,
    )


# --------------------------------------------------------------------------- #
def overlay_audit(marker: np.ndarray, res: CoverageResult, out_path: Path,
                  title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .visualize import _norm

    mip = marker.max(axis=0)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(_norm(mip), cmap="gray", interpolation="nearest")

    if len(res.detected_centroids_vox):
        d = res.detected_centroids_vox
        ax.scatter(d[:, 2], d[:, 1], s=24, facecolors="none",
                   edgecolors="#34c759", linewidths=0.7, label="detected")
    if len(res.miss_centroids_vox):
        m = res.miss_centroids_vox
        ax.scatter(m[:, 2], m[:, 1], s=130, facecolors="none",
                   edgecolors="#ffd60a", linewidths=2.0,
                   label="candidate miss")
    if len(res.fp_centroids_vox):
        f = res.fp_centroids_vox
        ax.scatter(f[:, 2], f[:, 1], s=90, marker="x",
                   color="#ff375f", linewidths=1.6,
                   label="candidate false +")
    ax.legend(loc="upper right", framealpha=0.6, fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _read_centroids(results_dir: Path) -> Tuple[np.ndarray, Optional[float]]:
    csv = results_dir / "neurons.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found -- run `python -m neuron_counter <file.czi>` first."
        )
    df = pd.read_csv(csv)
    cols = ["centroid_z", "centroid_y", "centroid_x"]
    if not all(c in df.columns for c in cols):
        raise ValueError(f"{csv} missing columns {cols}")
    cvox = df[cols].to_numpy(dtype=float)
    median_d = float(np.nanmedian(df["diameter_um"])) if "diameter_um" in df else None
    return cvox, median_d


def _load_config(path: Optional[str]) -> dict:
    import yaml
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    default = Path(__file__).with_name("config.yaml")
    if default.exists():
        with open(default) as f:
            return yaml.safe_load(f) or {}
    return {}


def run_audit(czi_path: str, results_dir: str, cfg: dict,
              params: CoverageParams,
              marker: Optional[np.ndarray] = None,
              voxel_um: Optional[VoxelUM] = None,
              channel_name: Optional[str] = None,
              save_png: bool = True) -> CoverageResult:
    ch = cfg.get("channels", {})
    fb = cfg.get("voxel_um", {})
    marker_idx = int(ch.get("neuron_marker_index", 0))
    # Reuse an already-loaded marker (pipeline) or load the CZI ourselves.
    if marker is None or voxel_um is None or channel_name is None:
        stack = load_czi(czi_path, channel_names=ch.get("names"),
                         voxel_um_fallback=(fb.get("z", 1.0), fb.get("y", 0.69),
                                            fb.get("x", 0.69)))
        marker = stack.channel(marker_idx)
        voxel_um = stack.voxel_um
        channel_name = stack.channel_names[marker_idx]
    rdir = Path(results_dir)
    cvox, median_d = _read_centroids(rdir)

    fgp = dict(cfg.get("foreground", {}))
    fgp.setdefault("random_state", int(cfg.get("random_state", 0)))
    if params.min_miss_diameter_um is None:
        params.min_miss_diameter_um = cfg.get("size", {}).get("min_diameter_um", 8.0)

    res = coverage_audit(marker, voxel_um, cvox, fgp, params,
                         median_detected_diam_um=median_d)

    # --- write outputs ------------------------------------------------------
    name = channel_name
    if save_png:
        overlay_audit(
            marker, res, rdir / "qc_coverage.png",
            title=(f"{name} coverage audit -- {res.n_detected} detected, "
                   f"{res.n_candidate_misses} candidate miss(es), "
                   f"{res.n_candidate_false_positives} candidate FP"),
        )
    res.miss_table.to_csv(rdir / "qc_candidate_misses.csv", index=False)
    summary = {
        "file": Path(czi_path).name,
        "results_dir": str(rdir),
        "n_detected": res.n_detected,
        "foreground_voxels": res.n_fg_voxels,
        "n_components": res.n_components,
        "n_covered_components": res.n_covered_components,
        "coverage_fraction_voxels": round(res.coverage_fraction, 4),
        "coverage_fraction_intensity": round(res.coverage_fraction_intensity, 4),
        "n_uncovered_soma_components": res.n_uncovered_soma_components,
        "n_candidate_misses": res.n_candidate_misses,
        "n_candidate_false_positives": res.n_candidate_false_positives,
        "implied_recall_lower_bound": round(res.recall_lower, 4),
        "params": {
            "centroid_snap_um": params.centroid_snap_um,
            "min_miss_diameter_um": params.min_miss_diameter_um,
            "typical_soma_diameter_um": params.typical_soma_diameter_um or median_d,
        },
        "note": ("Label-free audit. coverage_fraction = share of bright HuC/D "
                 "foreground that sits in a component owning >=1 detection. "
                 "Candidate misses/FPs are SUSPECTS to eyeball (amber rings in "
                 "zviewer.html), not a measured error rate."),
    }
    with open(rdir / "qc_coverage.json", "w") as f:
        json.dump(summary, f, indent=2)
    return res


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="neuron_counter.qc_coverage",
        description="Label-free a-posteriori audit: find bright HuC/D blobs that "
                    "carry no digital annotation (candidate misses) and "
                    "annotations on empty regions (candidate false positives).",
    )
    p.add_argument("input", help="path to the .czi file that was counted")
    p.add_argument("--results", default="results",
                   help="directory holding neurons.csv (default: results)")
    p.add_argument("-c", "--config", help="path to config.yaml used for the run")
    p.add_argument("--centroid-snap-um", type=float, default=3.0,
                   help="max distance a centroid may sit off-mask and still "
                        "count as covering a component (default 3)")
    p.add_argument("--min-miss-diameter-um", type=float, default=None,
                   help="min XY diameter for an uncovered blob to be a candidate "
                        "miss (default: size.min_diameter_um from config)")
    p.add_argument("--typical-soma-diameter-um", type=float, default=None,
                   help="typical soma diameter for the area->neuron-count "
                        "estimate (default: median detected diameter)")
    args = p.parse_args(argv)

    cfg = _load_config(args.config)
    params = CoverageParams(
        centroid_snap_um=args.centroid_snap_um,
        min_miss_diameter_um=args.min_miss_diameter_um,
        typical_soma_diameter_um=args.typical_soma_diameter_um,
    )
    res = run_audit(args.input, args.results, cfg, params)

    print("\n=== Coverage audit (no hand annotation) ===")
    print(f"Detected neurons     : {res.n_detected}")
    print(f"Foreground voxels    : {res.n_fg_voxels:,} in {res.n_components} components")
    print(f"Coverage (voxels)    : {100*res.coverage_fraction:.1f}%  "
          f"(intensity {100*res.coverage_fraction_intensity:.1f}%)")
    print(f"Candidate misses     : {res.n_candidate_misses} "
          f"(from {res.n_uncovered_soma_components} uncovered soma-sized blobs)")
    print(f"Candidate false +    : {res.n_candidate_false_positives}")
    print(f"Implied recall >=    : {100*res.recall_lower:.1f}% "
          f"(detected / (detected + candidate misses))")
    print(f"Outputs              : {Path(args.results)}/qc_coverage.png, "
          f"qc_coverage.json, qc_candidate_misses.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
