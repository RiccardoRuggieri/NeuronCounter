# neuron_counter — count HuC/D⁺ neurons in 3D `.czi` confocal stacks

Counts neurons in Carl Zeiss `.czi` z-stacks **without any deep network**. The
marker channel (default HuC/D) is modelled as a non-parametric **3D Gaussian
kernel-density estimate (KDE)**, and each soma is one **mode** (local maximum)
of that density. The neuron count is the number of modes. It runs on a CPU in
about a second per single pass.

## Method

```
.czi  ─▶  marker channel (Z,Y,X)
      ─▶  foreground voxels ─▶ intensity-weighted 3D point cloud (in microns)
      ─▶  3D Gaussian KDE density
      ─▶  mean-shift to the modes ─▶ merge coincident modes
      ─▶  size filter (exclude < 8 µm and > 35 µm somata)
      ─▶  count = number of surviving modes
```

1. **Point cloud.** Bright voxels (above an Otsu threshold of a lightly smoothed
   volume) become points in **physical micron space** `(z·vz, y·vy, x·vx)`, so the
   Gaussians live in an isotropic metric despite anisotropic voxels. Points are
   sub-sampled proportional to intensity, so bright soma cores carry more mass.

2. **KDE density.** The cloud is a sum of one isotropic Gaussian of bandwidth `h`
   per point. There is no "number of components" to choose — the count *emerges*
   as the number of density modes. The bandwidth `h` is essentially the soma
   radius and is the single most important knob.

3. **Mode counting.** From each local intensity maximum we climb the density with
   a Gaussian **mean-shift** until it lands on a maximum, then merge coincident
   maxima. Over-segmented twin peaks collapse into one mode; somata farther apart
   than `~h` stay separate.

4. **Size filter.** Each mode is assigned its nearest foreground voxels and its
   **in-plane (XY) equivalent diameter** is measured. Modes smaller than **8 µm**
   (debris) or larger than **35 µm** (fused blobs) are excluded. Size is judged
   in-plane because the stack is only ≈ 6 µm deep.

## Robust counting & uncertainty (default)

A single run depends on acquisition noise, the foreground threshold, the random
sub-sample, and the bandwidth. To measure how much the count actually depends on
those, the detection is **rebuilt end-to-end** over an **ensemble** (default 25
runs), each run perturbing the *genuine* sources of variation:

- a **fresh measurement-noise realization** added to the image (σ **estimated
  from the image's own noise floor**, not a free dial), then re-smoothed,
  re-thresholded and **re-seeded**;
- the **foreground threshold jittered** ±8 % (Otsu is not exact), which moves
  both the foreground and the mean-shift **seeds**;
- a fresh intensity-weighted **sub-sample**; and
- the KDE **bandwidth jittered** ±15 % (the modelling-scale choice).

A neuron is kept only if re-detected in at least half the runs (**stability
selection**). The reported count is this consensus, with a per-neuron
**stability** score (fraction of runs it appeared in) and a **95 % bootstrap
band** (2.5–97.5th percentile of the per-run counts). The ensemble is
reproducible (fixed `random_state`); disable it with `--no-robust` for a single
fast pass.

> This is a *sensitivity* estimate over the imaging and the pipeline's own
> choices — not a substitute for validation against ground-truth counts, which
> is the only way to know whether the consensus is actually *accurate*. In
> practice the spread is dominated by the bandwidth/scale choice; noise and
> threshold contribute little.

### Reading the viewer: green / blue / amber

Each detection is coloured by its stability score:

- **Green = stable** (re-detected in ≥ 80 % of runs): robust to every
  perturbation — almost certainly a real neuron.
- **Blue = marginal** (re-detected in 50–80 % of runs): it *flickered* — present
  in some runs, absent in others. These are the **uncertain** ones, almost always
  a borderline case where the answer depends on scale: two somata right at the
  merge/split boundary, a cell near the 8/35 µm size cutoff, or a faint
  near-threshold peak. Exactly the population worth eyeballing.
- **Amber = candidate miss**: not a low-stability detection at all, but a place
  the separate coverage-audit pass flags as possibly skipped by the main counter.

## Install

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

No GPU, no model weights, no network. CZI reading uses the pure-python `czifile`
(no compiler needed); the faster `aicspylibczi` is used automatically if present.

## Easy mode (GUI, for operators)

1. **Launch** — double-click **`Launch Neuron Counter.command`** (macOS) or
   **`Launch Neuron Counter.bat`** (Windows), or run `python -m neuron_counter.app`.
2. **Select image…** — pick a `.czi`. A **progress bar** shows the algorithm
   running (~20–30 s); the interactive viewer then opens in your browser. Outputs
   land next to the image in `<image>_results/`.
3. **Review** — zoom/pan, scrub depth, add / remove neurons by eye
   (green = stable, blue = marginal, amber = candidate miss).
4. **✓ Accept & save count** — saves a one-line `<image>_count.csv` next to the
   image and closes the app.

The GUI needs Python's Tkinter (bundled with python.org builds; on Debian/Ubuntu
`sudo apt install python3-tk`).

## Usage (CLI)

```bash
# Count with defaults:
python -m neuron_counter sample.czi

# The one knob that matters — bandwidth (~soma radius, microns):
python -m neuron_counter sample.czi --bandwidth-um 8   # larger -> fewer, merged cells

# Single fast pass without the ensemble:
python -m neuron_counter sample.czi --no-robust

# Inspect channels / voxel size and exit:
python -m neuron_counter sample.czi --info
```

## Outputs (in `results/` by default)

By default only the z-slice viewer and the data it needs are written:

- `zviewer.html` — the z-slice review viewer (green/blue/amber rings).
- `neurons.csv` — one row per neuron: centroid (voxel & micron), in-plane
  `diameter_um`, `stability` score, intensity.
- `qc_candidate_misses.csv` / `qc_coverage.json` — coverage-audit data (the amber
  candidate-miss rings shown in the viewer).
- `summary.json` — count, per-run mean/std and 95 % band, uncertainty sources,
  per-stage timings.

The static PNGs (`overlay.png`, `zmontage.png`, `qc_coverage.png`) and the 3D
viewer (`neurons_3d.html`) are **off by default**; enable them in `config.yaml`
via `output.save_overlays`, `output.save_coverage_png`, `output.save_html_3d`.

## Package layout

```
neuron_counter/
  io_czi.py      # CZI -> (C,Z,Y,X) + voxel metadata
  foreground.py  # bright voxels -> intensity-weighted 3D point cloud + noise estimate
  gmm_modes.py   # KDE density, mean-shift mode counting, robust ensemble
  visualize.py   # MIP overlay + z montage
  viz3d.py       # interactive 3D HTML viewer (plotly)
  viz_zslice.py  # z-slice review viewer
  qc_coverage.py # label-free coverage audit (candidate misses)
  pipeline.py    # orchestration
  app.py         # GUI launcher
  __main__.py    # CLI
  config.yaml    # all parameters
```

All parameters live in `neuron_counter/config.yaml`; the defaults are the
standard KDE settings and need no tuning for typical HuC/D stacks. The main knob,
if cells are split or merged, is `gmm.bandwidth_um` — calibrate it so the
reported `diameter_um` median sits in the expected 15–25 µm range.
