"""
viz_zslice
==========
Self-contained **z-slice viewer** for a counted ``.czi`` stack -- the 2D view
familiar from ZEN / Fiji / napari, where a cursor scrubs through depth and the
image **cross-fades (interpolates) between neighbouring z-planes** so the pass
through z looks continuous instead of jumping plane-to-plane.

Shows the HuC/D marker channel. Detected neuron centroids (read back from
``neurons.csv``) can be overlaid as rings that fade in around their own z, so
they appear exactly where they live in depth as you scrub.

The output is a single standalone ``.html`` file (slices embedded as base64
PNGs) -- no server, opens in any browser, shareable like ``neurons_3d.html``.

CLI
---
    python -m neuron_counter.viz_zslice sample.czi --results results
    python -m neuron_counter.viz_zslice sample.czi -o results/zviewer.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

VoxelUM = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
def _global_norm_uint8(vol: np.ndarray, lo=1.0, hi=99.7) -> np.ndarray:
    """Normalise the WHOLE stack with one window so planes don't flicker in
    brightness as you scrub, then map to 8-bit."""
    a, b = np.percentile(vol, [lo, hi])
    if b <= a:
        b = a + 1.0
    out = np.clip((vol.astype(np.float32) - a) / (b - a), 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _png_data_url(plane_u8: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(plane_u8, mode="L").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _read_centroids(results_dir: Path) -> List[dict]:
    csv = results_dir / "neurons.csv"
    if not csv.exists():
        return []
    import pandas as pd
    df = pd.read_csv(csv)
    need = ["centroid_z", "centroid_y", "centroid_x"]
    if not all(c in df.columns for c in need):
        return []
    has_stab = "stability" in df.columns
    out = []
    for _, r in df.iterrows():
        out.append({
            "z": round(float(r["centroid_z"]), 3),
            "y": round(float(r["centroid_y"]), 2),
            "x": round(float(r["centroid_x"]), 2),
            "d": round(float(r["diameter_um"]), 2) if "diameter_um" in df else 18.0,
            # stability (fraction of ensemble runs the neuron appeared in); a
            # single-run result has no ensemble, so treat it as fully stable.
            "s": round(float(r["stability"]), 3) if has_stab else 1.0,
        })
    return out


def _read_misses(results_dir: Path,
                 csv_name: str = "qc_candidate_misses.csv") -> List[dict]:
    """Candidate missed somata found by the coverage audit (qc_coverage)."""
    csv = results_dir / csv_name
    if not csv.exists():
        return []
    import pandas as pd
    df = pd.read_csv(csv)
    if df.empty or "centroid_z" not in df.columns:
        return []
    out = []
    for _, r in df.iterrows():
        out.append({
            "z": round(float(r["centroid_z"]), 3),
            "y": round(float(r["centroid_y"]), 2),
            "x": round(float(r["centroid_x"]), 2),
            "d": (round(float(r["xy_diameter_um"]), 2)
                  if "xy_diameter_um" in df else 12.0),
            "n": int(r["est_neurons"]) if "est_neurons" in df else 1,
        })
    return out


# --------------------------------------------------------------------------- #
_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{ color-scheme: dark; }
  *{ box-sizing:border-box; }
  html,body{ margin:0; height:100%; background:#000; color:#e7e7ea;
    font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .app{ display:flex; height:100vh; width:100vw; overflow:hidden; }
  .stage{ position:relative; flex:1 1 auto; min-width:0; background:#000; overflow:hidden; }
  #cv{ position:absolute; inset:0; width:100%; height:100%; display:block;
    touch-action:none; cursor:grab; }
  #cv.editing{ cursor:crosshair; }
  #cv.panning{ cursor:grabbing; }
  /* floating overlays on the image */
  .hud{ position:absolute; top:10px; left:12px; background:rgba(0,0,0,.5);
    border:1px solid #2a2a31; border-radius:7px; padding:5px 9px; font-size:12px;
    font-variant-numeric:tabular-nums; pointer-events:none; }
  .fab{ position:absolute; top:10px; right:12px; width:34px; height:34px; border-radius:9px;
    display:flex; align-items:center; justify-content:center; font-size:17px; z-index:5;
    background:rgba(20,20,26,.85); border:1px solid #2e2e36; cursor:pointer; color:#e7e7ea; }
  .fab:hover{ border-color:#3a3a45; }
  .zoomgrp{ position:absolute; right:12px; bottom:14px; display:flex; flex-direction:column;
    gap:6px; z-index:5; }
  .zoomgrp button{ width:34px; height:34px; padding:0; font-size:17px; border-radius:9px;
    background:rgba(20,20,26,.85); }
  /* side panel */
  .panel{ flex:0 0 340px; width:340px; height:100%; overflow-y:auto;
    background:#0b0b0d; border-left:1px solid #1f1f25; padding:14px 16px 28px;
    display:flex; flex-direction:column; gap:14px; }
  .app.collapsed .panel{ display:none; }
  .phead{ display:flex; align-items:flex-start; justify-content:space-between; gap:8px; }
  .phead h1{ font-size:14px; margin:0; font-weight:600; line-height:1.3; }
  .sub{ color:#8a8a92; font-size:11.5px; margin:0; }
  .sec{ display:flex; flex-direction:column; gap:8px;
    border-top:1px solid #1c1c22; padding-top:12px; }
  .sec.first{ border-top:none; padding-top:0; }
  .sec h2{ font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
    color:#6f6f78; margin:0 0 2px; }
  .ctl{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  input[type=range]{ width:100%; accent-color:#34c759; }
  .badge{ font-variant-numeric:tabular-nums; background:#16161a; border:1px solid #2a2a31;
    border-radius:6px; padding:3px 8px; font-size:12px; text-align:center; }
  label.sm{ color:#aeaeb6; font-size:12px; display:flex; align-items:center; gap:6px; }
  button,.filebtn{ background:#1c1c22; color:#e7e7ea; border:1px solid #2e2e36;
    border-radius:7px; padding:6px 11px; cursor:pointer; font-size:13px; }
  button:hover,.filebtn:hover{ border-color:#3a3a45; }
  button.on{ background:#0f3d22; border-color:#1f7a43; color:#7bf3a6; }
  button.warn{ background:#3a2f08; border-color:#7a5e1f; color:#ffd60a; }
  button.active{ background:#172f4d; border-color:#2f6db0; color:#8cc2ff; }
  button.accept{ background:#0f3d22; border-color:#2f9e58; color:#7bf3a6;
    font-weight:600; flex:1; }
  button.accept:hover{ border-color:#3fcf74; }
  .approved{ position:absolute; left:50%; top:16px; transform:translateX(-50%) translateY(-12px);
    background:rgba(15,61,34,.95); border:1px solid #2f9e58; color:#a7f3c4;
    padding:10px 16px; border-radius:9px; font-size:13px; font-weight:600;
    opacity:0; pointer-events:none; transition:opacity .25s, transform .25s; z-index:8; }
  .approved.show{ opacity:1; transform:translateX(-50%) translateY(0); }
  .hint{ color:#7a7a82; font-size:11px; }
  #cur{ color:#cfcfd6; font-size:12px; }
  .legend{ display:flex; flex-direction:column; gap:6px; color:#b9b9c0; font-size:12px; }
  .legend i{ display:inline-block; width:11px; height:11px; border-radius:50%;
    margin-right:7px; vertical-align:-1px; }
  .legend i.open{ background:transparent !important; border:2px solid; }
</style></head>
<body><div class="app" id="app">
  <div class="stage" id="stage">
    <canvas id="cv"></canvas>
    <div class="hud" id="hud">z = 0.00</div>
    <div class="approved" id="approved"></div>
    <button id="panelOpen" class="fab" title="Show / hide panel">≡</button>
    <div class="zoomgrp">
      <button id="zin" title="Zoom in">＋</button>
      <button id="zfit" title="Fit to view">⤢</button>
      <button id="zout" title="Zoom out">－</button>
    </div>
  </div>
  <aside class="panel" id="panel">
    <div class="phead">
      <h1>__TITLE__</h1>
      <button id="panelClose" title="Hide panel">✕</button>
    </div>
    <p class="sub">Scroll = zoom · drag = pan · scrub = depth.
      Curate: pick a mode, then click the image.</p>
    <p class="sub" id="genline" style="color:#6f6f78"></p>

    <div class="sec first">
      <h2>Depth</h2>
      <div class="ctl"><button id="play">▶ Play</button>
        <span class="badge" id="zlabel">z = 0.00</span></div>
      <input id="z" type="range" min="0" max="__ZMAX__" step="0.02" value="0">
      <label class="sm">Speed <span class="badge" id="spdlabel">0.035/f</span></label>
      <input id="spd" type="range" min="2" max="120" value="35">
    </div>

    <div class="sec">
      <h2>Image</h2>
      <label class="sm">Brightness</label>
      <input id="bri" type="range" min="50" max="250" value="100">
      <label class="sm">Contrast</label>
      <input id="con" type="range" min="50" max="300" value="100">
      <label class="sm"><input id="interp" type="checkbox" checked> interpolate between z-planes</label>
    </div>

    <div class="sec">
      <h2>Overlays</h2>
      <button id="ovl" class="on">◎ Neurons: on</button>
      <button id="miss" class="warn">△ Candidate misses: on</button>
      <div class="hint" id="cnt"></div>
    </div>

    <div class="sec">
      <h2>Curate (expert eye)</h2>
      <div class="ctl">
        <button id="m_off" class="active">✋ Off</button>
        <button id="m_sel">◎ Toggle</button>
        <button id="m_add">＋ Add</button>
      </div>
      <div class="ctl">
        <button id="accept" class="accept">✓ Accept &amp; save count</button>
        <button id="rst">↺ Reset</button>
      </div>
      <div class="hint" id="cur"></div>
    </div>

    <div class="sec">
      <h2>Legend</h2>
      <div class="legend">
        <span><i style="background:#34c759"></i>stable (≥ __STABLE_THR__)</span>
        <span><i style="background:#0a84ff"></i>marginal</span>
        <span><i class="open" style="border-color:#ffd60a"></i>candidate miss</span>
        <span><i style="background:#ffd60a"></i>accepted miss</span>
        <span><i style="background:#bf5af2"></i>manual add</span>
        <span><i style="background:#ff453a"></i>rejected</span>
      </div>
    </div>
  </aside>
</div>
<script>
const IMAGES = __IMAGES__;
const CENTROIDS = __CENTROIDS__;
const MISSES = __MISSES__;
const FILENAME = __FILENAME__;
const RUN_ID = __RUNID__;             // unique per generation -> fresh state each run
// When launched by the desktop app, the page URL carries ?accept=<localhost url>.
// Clicking Accept then pings the app (which saves the count CSV and closes).
const ACCEPT_URL = (function(){ try { return new URLSearchParams(location.search).get('accept'); }
  catch(e){ return null; } })();
const VZ = __VZ__, VY = __VY__, VX = __VX__;
const W = __W__, H = __H__, NZ = IMAGES.length;
const Z_SIGMA = 0.9;            // depth (planes) over which a neuron ring fades
const STABLE_THR = __STABLE_THR__;   // stability >= this -> "stable" neuron
const C_STABLE = '52,199,89';        // green  : stable detections
const C_MARGINAL = '10,132,255';     // blue   : marginal (low-stability) detections
const C_MISS = '255,214,10';         // amber  : candidate misses (2nd algorithm)

const C_MANUAL = '191,90,242';       // violet : manually added neurons
const C_REJECT = '255,69,58';        // red    : rejected detections
const HIT_R = 16;                    // click hit radius (image px)

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const zEl = document.getElementById('z'), zlab = document.getElementById('zlabel');
const ovlBtn = document.getElementById('ovl'), playBtn = document.getElementById('play');
const missBtn = document.getElementById('miss');
const briEl = document.getElementById('bri'), conEl = document.getElementById('con');
const interpEl = document.getElementById('interp'), cntEl = document.getElementById('cnt');
const spdEl = document.getElementById('spd'), spdLab = document.getElementById('spdlabel');
const mOff = document.getElementById('m_off'), mSel = document.getElementById('m_sel');
const mAdd = document.getElementById('m_add'), curEl = document.getElementById('cur');
const acceptBtn = document.getElementById('accept');
const rstBtn = document.getElementById('rst');
const approvedEl = document.getElementById('approved');
const app = document.getElementById('app'), stage = document.getElementById('stage');
const hud = document.getElementById('hud');
const zinBtn = document.getElementById('zin'), zoutBtn = document.getElementById('zout');
const zfitBtn = document.getElementById('zfit');
const panelOpenBtn = document.getElementById('panelOpen'), panelCloseBtn = document.getElementById('panelClose');

const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
let view = { k: 1, tx: 0, ty: 0 };   // image->canvas transform (zoom + pan)
let fitK = 1;                        // scale at which the image fits the canvas
let INVK = 1;                        // 1/view.k, for screen-constant line widths

let imgs = [], ready = 0, showOvl = true, showMiss = true, playing = false, raf = null, dir = 1;
let editMode = 'off';                // 'off' | 'sel' | 'add'
let speed = 0.035;                   // z-planes advanced per animation frame
const detRejected = new Set();       // indices into CENTROIDS marked NOT retained
const missAccepted = new Set();      // indices into MISSES promoted to retained
let manualPts = [];                  // {x,y,z,d} expert-added neurons (voxel coords)
const N_STABLE = CENTROIDS.filter(c => c.s >= STABLE_THR).length;
const N_MARG = CENTROIDS.length - N_STABLE;
const MED_D = (function(){ const a = CENTROIDS.map(c => c.d).sort((p,q)=>p-q);
  return a.length ? a[a.length>>1] : 18.0; })();   // default diameter for manual pts
// Key the saved curation by the run id, so edits from a PREVIOUS run never
// reload on top of freshly recomputed neurons (that made re-runs look stale).
const LS_KEY = 'neuron_curation::' + FILENAME + '::' + RUN_ID;

function fit(){                       // size the canvas backing store to the stage
  const r = stage.getBoundingClientRect();
  cv.style.width = r.width + 'px'; cv.style.height = r.height + 'px';
  cv.width = Math.max(1, Math.round(r.width * DPR));
  cv.height = Math.max(1, Math.round(r.height * DPR));
}
function fitView(){                   // reset zoom/pan so the image fills the canvas
  fitK = Math.min(cv.width / W, cv.height / H);
  view.k = fitK;
  view.tx = (cv.width - W * fitK) / 2;
  view.ty = (cv.height - H * fitK) / 2;
}

function draw(){
  const zf = parseFloat(zEl.value);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (ready < NZ) return;
  ctx.setTransform(view.k, 0, 0, view.k, view.tx, view.ty);   // zoom + pan
  ctx.imageSmoothingEnabled = true;
  const lo = Math.floor(zf), hi = Math.min(NZ - 1, Math.ceil(zf));
  const frac = zf - lo;
  if (!interpEl.checked){
    ctx.globalAlpha = 1; ctx.drawImage(imgs[Math.round(zf)], 0, 0, W, H);
  } else {
    ctx.globalAlpha = 1; ctx.drawImage(imgs[lo], 0, 0, W, H);
    if (hi !== lo){ ctx.globalAlpha = frac; ctx.drawImage(imgs[hi], 0, 0, W, H); }
    ctx.globalAlpha = 1;
  }
  INVK = 1 / view.k;                  // keep ring strokes ~constant on screen
  const det = showOvl ? drawRings(zf) : [0, 0];
  const nMiss = showMiss ? drawMisses(zf) : 0;
  drawManual(zf);
  updateCount(det[0], det[1], nMiss);
  updateCuration();
  const zt = 'z = ' + zf.toFixed(2) + '  (' + (zf * VZ).toFixed(2) + ' µm)';
  zlab.textContent = zt;
  hud.textContent = zt + '   ·   ' + Math.round(view.k / fitK * 100) + '%';
}

function ringFade(zf, z){ const dz = zf - z; return Math.exp(-(dz*dz)/(2*Z_SIGMA*Z_SIGMA)); }

function drawRings(zf){            // detected: green=stable, blue=marginal, red=rejected
  ctx.setLineDash([]);
  let nStable = 0, nMarg = 0;
  for (let i = 0; i < CENTROIDS.length; i++){
    const c = CENTROIDS[i];
    const w = ringFade(zf, c.z);
    if (w < 0.06) continue;
    const r = Math.max(4, (c.d / 2) / VY);     // soma radius in pixels
    if (detRejected.has(i)){                   // rejected by the expert -> red + ✕
      ctx.lineWidth = 1.6 * INVK;
      ctx.strokeStyle = 'rgba(' + C_REJECT + ',' + w.toFixed(3) + ')';
      ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, 6.2832); ctx.stroke();
      const k = r * 0.7;
      ctx.beginPath();
      ctx.moveTo(c.x - k, c.y - k); ctx.lineTo(c.x + k, c.y + k);
      ctx.moveTo(c.x - k, c.y + k); ctx.lineTo(c.x + k, c.y - k); ctx.stroke();
      continue;
    }
    const stable = (c.s >= STABLE_THR);
    if (stable) nStable++; else nMarg++;
    ctx.lineWidth = (stable ? 2.0 : 1.4) * INVK;  // stable drawn a touch bolder
    ctx.strokeStyle = 'rgba(' + (stable ? C_STABLE : C_MARGINAL) + ',' + w.toFixed(3) + ')';
    ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, 6.2832); ctx.stroke();
  }
  return [nStable, nMarg];
}

function drawMisses(zf){           // candidate misses: dashed amber; accepted=solid + ✓
  let visible = 0;
  for (let i = 0; i < MISSES.length; i++){
    const c = MISSES[i];
    const w = ringFade(zf, c.z);
    if (w < 0.06) continue;
    visible++;
    const r = Math.max(6, (c.d / 2) / VY);
    ctx.strokeStyle = 'rgba(' + C_MISS + ',' + w.toFixed(3) + ')';
    if (missAccepted.has(i)){
      ctx.setLineDash([]); ctx.lineWidth = 2.6 * INVK;
      ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, 6.2832); ctx.stroke();
      ctx.beginPath();                          // check-mark
      ctx.moveTo(c.x - r*0.4, c.y); ctx.lineTo(c.x - r*0.05, c.y + r*0.35);
      ctx.lineTo(c.x + r*0.45, c.y - r*0.3); ctx.stroke();
    } else {
      ctx.setLineDash([5 * INVK, 4 * INVK]); ctx.lineWidth = 2.2 * INVK;
      ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, 6.2832); ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  return visible;
}

function drawManual(zf){           // expert-added neurons: violet ring + ＋
  ctx.setLineDash([]); ctx.lineWidth = 2.0 * INVK;
  for (const c of manualPts){
    const w = ringFade(zf, c.z);
    if (w < 0.06) continue;
    const r = Math.max(5, (c.d / 2) / VY);
    ctx.strokeStyle = 'rgba(' + C_MANUAL + ',' + w.toFixed(3) + ')';
    ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, 6.2832); ctx.stroke();
    const k = r * 0.6;
    ctx.beginPath();
    ctx.moveTo(c.x - k, c.y); ctx.lineTo(c.x + k, c.y);
    ctx.moveTo(c.x, c.y - k); ctx.lineTo(c.x, c.y + k); ctx.stroke();
  }
}

function updateCuration(){
  const keptDet = CENTROIDS.length - detRejected.size;
  const total = keptDet + missAccepted.size + manualPts.length;
  curEl.textContent = 'Retained ' + total + '  =  ' + keptDet + ' detected (−'
    + detRejected.size + ' rejected)  +  ' + missAccepted.size + ' accepted miss  +  '
    + manualPts.length + ' manual';
}

function updateCount(nStable, nMarg, nMiss){
  const parts = [];
  if (showOvl) parts.push('● ' + N_STABLE + ' stable + ● ' + N_MARG + ' marginal · '
                          + (nStable + nMarg) + ' near depth');
  if (showMiss && MISSES.length) parts.push('▢ ' + MISSES.length + ' candidate miss · '
                          + nMiss + ' near depth');
  cntEl.textContent = parts.join('     |     ');
}

function applyFilter(){
  cv.style.filter = 'brightness(' + (briEl.value/100) + ') contrast(' + (conEl.value/100) + ')';
}

function loop(){
  if (!playing) return;
  let v = parseFloat(zEl.value) + dir * speed;
  if (v >= (NZ - 1)){ v = NZ - 1; dir = -1; }
  else if (v <= 0){ v = 0; dir = 1; }
  zEl.value = v; draw();
  raf = requestAnimationFrame(loop);
}

// ---- playback speed ------------------------------------------------------ //
function applySpeed(){ speed = spdEl.value / 1000;          // 0.002 .. 0.120 /frame
  spdLab.textContent = speed.toFixed(3) + '/f'; }
spdEl.addEventListener('input', applySpeed);

// ---- curation: modes, hit-testing, clicks -------------------------------- //
function setMode(m){
  editMode = m;
  mOff.classList.toggle('active', m === 'off');
  mSel.classList.toggle('active', m === 'sel');
  mAdd.classList.toggle('active', m === 'add');
  cv.classList.toggle('editing', m !== 'off');
}
mOff.addEventListener('click', () => setMode('off'));
mSel.addEventListener('click', () => setMode('sel'));
mAdd.addEventListener('click', () => setMode('add'));

function canvasXY(e){                                // event -> canvas backing px
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * (cv.width / r.width),
          (e.clientY - r.top) * (cv.height / r.height)];
}
function toImg(e){                                   // event -> image px (undo zoom/pan)
  const [sx, sy] = canvasXY(e);
  return [(sx - view.tx) / view.k, (sy - view.ty) / view.k];
}
function nearest(zf, ix, iy){                        // closest visible marker, if any
  const hitR = HIT_R / view.k;                       // screen px -> image px
  let best = null;
  const consider = (type, idx, x, y, z) => {
    if (ringFade(zf, z) < 0.06) return;
    const d = Math.hypot(x - ix, y - iy);
    if (d <= hitR && (!best || d < best.d)) best = { type, idx, d };
  };
  for (let i = 0; i < CENTROIDS.length; i++){ const c = CENTROIDS[i]; consider('det', i, c.x, c.y, c.z); }
  for (let i = 0; i < MISSES.length; i++){ const c = MISSES[i]; consider('miss', i, c.x, c.y, c.z); }
  for (let i = 0; i < manualPts.length; i++){ const c = manualPts[i]; consider('man', i, c.x, c.y, c.z); }
  return best;
}
function doCurate(e){
  if (editMode === 'off') return;
  const zf = parseFloat(zEl.value);
  const [ix, iy] = toImg(e);
  const hit = nearest(zf, ix, iy);
  if (editMode === 'sel'){                           // flip keep/reject on a marker
    if (!hit) return;
    if (hit.type === 'det'){ detRejected.has(hit.idx) ? detRejected.delete(hit.idx) : detRejected.add(hit.idx); }
    else if (hit.type === 'miss'){ missAccepted.has(hit.idx) ? missAccepted.delete(hit.idx) : missAccepted.add(hit.idx); }
    else if (hit.type === 'man'){ manualPts.splice(hit.idx, 1); }
  } else {                                           // 'add'
    if (hit && hit.type === 'miss'){ missAccepted.add(hit.idx); }       // accept a candidate miss
    else if (hit && hit.type === 'det'){ detRejected.delete(hit.idx); } // ensure a detection is kept
    else { manualPts.push({ x: ix, y: iy, z: zf, d: MED_D }); }          // brand-new neuron
  }
  saveState(); draw();
}

// ---- pan (drag) + curate (click) ----------------------------------------- //
let dragging = false, moved = false, lastX = 0, lastY = 0;
cv.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY;
  cv.classList.add('panning');
});
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
  const r = cv.getBoundingClientRect();
  view.tx += dx * (cv.width / r.width);
  view.ty += dy * (cv.height / r.height);
  lastX = e.clientX; lastY = e.clientY; draw();
});
window.addEventListener('mouseup', (e) => {
  if (!dragging) return;
  dragging = false; cv.classList.remove('panning');
  if (!moved) doCurate(e);                            // a click (no drag) curates
});

// ---- zoom (wheel / buttons / double-click) ------------------------------- //
function zoomAt(sx, sy, f){
  const k2 = Math.max(fitK * 0.8, Math.min(fitK * 30, view.k * f));
  const px = (sx - view.tx) / view.k, py = (sy - view.ty) / view.k;
  view.k = k2; view.tx = sx - px * k2; view.ty = sy - py * k2;
  draw();
}
cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  const [sx, sy] = canvasXY(e);
  zoomAt(sx, sy, Math.exp(-e.deltaY * 0.0015));
}, { passive: false });
cv.addEventListener('dblclick', (e) => {
  if (editMode !== 'off') return;                     // don't fight Add-mode clicks
  const [sx, sy] = canvasXY(e); zoomAt(sx, sy, 1.6);
});
zinBtn.addEventListener('click', () => zoomAt(cv.width / 2, cv.height / 2, 1.3));
zoutBtn.addEventListener('click', () => zoomAt(cv.width / 2, cv.height / 2, 1 / 1.3));
zfitBtn.addEventListener('click', () => { fitView(); draw(); });

// ---- collapsible side panel ---------------------------------------------- //
function setPanel(open){
  const ow = cv.width, oh = cv.height;
  app.classList.toggle('collapsed', !open);
  requestAnimationFrame(() => {                       // keep the view centred
    fit(); view.tx += (cv.width - ow) / 2; view.ty += (cv.height - oh) / 2; draw();
  });
}
panelOpenBtn.addEventListener('click', () => setPanel(app.classList.contains('collapsed')));
panelCloseBtn.addEventListener('click', () => setPanel(false));

// ---- persistence (localStorage; works from a file on disk) --------------- //
function saveState(){
  try { localStorage.setItem(LS_KEY, JSON.stringify(
    { rej: [...detRejected], acc: [...missAccepted], man: manualPts })); } catch(e){}
}
function loadState(){
  try {
    const s = localStorage.getItem(LS_KEY); if (!s) return;
    const d = JSON.parse(s);
    (d.rej || []).forEach(i => { if (i < CENTROIDS.length) detRejected.add(i); });
    (d.acc || []).forEach(i => { if (i < MISSES.length) missAccepted.add(i); });
    manualPts = Array.isArray(d.man) ? d.man : [];
  } catch(e){}
}

// ---- accept (finalise) + reset ------------------------------------------- //
function curatedCount(){
  return (CENTROIDS.length - detRejected.size) + missAccepted.size + manualPts.length;
}
function downloadFile(name, text, type){
  const b = new Blob([text], { type }); const u = URL.createObjectURL(b);
  const a = document.createElement('a'); a.href = u; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1500);
}
function showApproved(n, viaApp){
  approvedEl.textContent = viaApp
    ? '✓ Approved — ' + n + ' neurons. Count saved; you can close this tab.'
    : '✓ Approved — ' + n + ' neurons · neuron_count.csv saved';
  approvedEl.classList.add('show');
  if (!viaApp) setTimeout(() => approvedEl.classList.remove('show'), 5000);
}
function doAccept(){                                   // approval -> save the count
  const n = curatedCount();
  if (ACCEPT_URL){                                     // launched by the desktop app
    try {
      new Image().src = ACCEPT_URL + (ACCEPT_URL.indexOf('?') >= 0 ? '&' : '?')
        + 'count=' + n + '&image=' + encodeURIComponent(FILENAME) + '&t=' + Date.now();
    } catch(e){}
    showApproved(n, true);
  } else {                                             // standalone -> browser download
    downloadFile('neuron_count.csv', 'image,neuron_count\n' + FILENAME + ',' + n + '\n', 'text/csv');
    showApproved(n, false);
  }
}
acceptBtn.addEventListener('click', doAccept);
rstBtn.addEventListener('click', () => {
  if (!confirm('Clear all curation edits (rejections, accepted misses, manual additions)?')) return;
  detRejected.clear(); missAccepted.clear(); manualPts = [];
  saveState(); draw();
});

zEl.addEventListener('input', draw);
briEl.addEventListener('input', applyFilter);
conEl.addEventListener('input', applyFilter);
interpEl.addEventListener('change', draw);
ovlBtn.addEventListener('click', () => {
  showOvl = !showOvl;
  ovlBtn.classList.toggle('on', showOvl);
  ovlBtn.textContent = '◎ Neurons: ' + (showOvl ? 'on' : 'off');
  draw();
});
missBtn.addEventListener('click', () => {
  showMiss = !showMiss;
  missBtn.classList.toggle('warn', showMiss);
  missBtn.textContent = '△ Candidate misses: ' + (showMiss ? 'on' : 'off');
  draw();
});
playBtn.addEventListener('click', () => {
  playing = !playing;
  playBtn.textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) loop(); else if (raf) cancelAnimationFrame(raf);
});
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight'){ zEl.value = Math.min(NZ-1, parseFloat(zEl.value)+1); draw(); }
  else if (e.key === 'ArrowLeft'){ zEl.value = Math.max(0, parseFloat(zEl.value)-1); draw(); }
  else if (e.code === 'Space'){ e.preventDefault(); playBtn.click(); }
});
window.addEventListener('resize', () => {
  const ow = cv.width, oh = cv.height;
  fit(); view.tx += (cv.width - ow) / 2; view.ty += (cv.height - oh) / 2; draw();
});

// preload embedded slices
(function(){ const g = document.getElementById('genline');
  if (g) g.textContent = 'Generated ' + RUN_ID; })();
loadState();
applySpeed();
fit();
IMAGES.forEach((src, i) => {
  const im = new Image();
  im.onload = () => { ready++; if (ready === NZ){ fit(); fitView(); applyFilter(); draw(); } };
  im.src = src; imgs[i] = im;
});
</script></body></html>
"""


def write_zslice_html(marker: np.ndarray, voxel_um: VoxelUM,
                      centroids: List[dict], out_path: Path,
                      title: str = "z-slice viewer",
                      misses: Optional[List[dict]] = None,
                      stable_threshold: float = 0.8,
                      source_name: str = "", run_id: str = "") -> None:
    import datetime as _dt
    if not run_id:
        _now = _dt.datetime.now()
        run_id = _now.strftime("%Y-%m-%d %H:%M:%S.") + f"{_now.microsecond // 1000:03d}"
    Z, Y, X = marker.shape
    u8 = _global_norm_uint8(marker)
    images = [_png_data_url(u8[z]) for z in range(Z)]
    html = (_HTML
            .replace("__TITLE__", title)
            .replace("__ZMAX__", str(Z - 1))
            .replace("__IMAGES__", json.dumps(images))
            .replace("__CENTROIDS__", json.dumps(centroids))
            .replace("__MISSES__", json.dumps(misses or []))
            .replace("__FILENAME__", json.dumps(source_name))
            .replace("__RUNID__", json.dumps(run_id))
            .replace("__STABLE_THR__", repr(float(stable_threshold)))
            .replace("__VZ__", repr(float(voxel_um[0])))
            .replace("__VY__", repr(float(voxel_um[1])))
            .replace("__VX__", repr(float(voxel_um[2])))
            .replace("__W__", str(X))
            .replace("__H__", str(Y)))
    out_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
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


def generate_viewer(input_path: str, results_dir: str = "results",
                    output: Optional[str] = None, cfg: Optional[dict] = None,
                    config_path: Optional[str] = None,
                    marker_index: Optional[int] = None,
                    no_overlay: bool = False, no_misses: bool = False,
                    stable_threshold: float = 0.8,
                    run_audit_if_missing: bool = True, verbose: bool = True,
                    marker: Optional[np.ndarray] = None,
                    voxel_um: Optional[VoxelUM] = None,
                    channel_name: Optional[str] = None, n_z: Optional[int] = None
                    ) -> Path:
    """Build the standalone z-slice viewer HTML.

    Can run standalone (loads the CZI itself) or be called by the pipeline with
    an already-loaded ``marker`` / ``voxel_um`` / ``channel_name`` to avoid a
    redundant reload. ``cfg`` is used as-is when given (so CLI overrides flow
    through); otherwise it is read from ``config_path`` / the packaged default.
    """
    from .io_czi import load_czi
    if cfg is None:
        cfg = _load_config(config_path)
    ch = cfg.get("channels", {})
    fb = cfg.get("voxel_um", {})
    marker_idx = (marker_index if marker_index is not None
                  else int(ch.get("neuron_marker_index", 0)))

    if marker is None or voxel_um is None or channel_name is None:
        stack = load_czi(input_path, channel_names=ch.get("names"),
                         voxel_um_fallback=(fb.get("z", 1.0), fb.get("y", 0.69),
                                            fb.get("x", 0.69)))
        marker = stack.channel(marker_idx)
        voxel_um = stack.voxel_um
        channel_name = stack.channel_names[marker_idx]
        n_z = stack.n_z
    if n_z is None:
        n_z = marker.shape[0]

    rdir = Path(results_dir)
    centroids = [] if no_overlay else _read_centroids(rdir)

    misses: List[dict] = []
    if not no_misses:
        misses = _read_misses(rdir)
        if (not misses and run_audit_if_missing
                and (rdir / "neurons.csv").exists()):
            try:
                from . import qc_coverage as _qc
                if verbose:
                    print("  (running coverage audit to find candidate misses…)")
                _qc.run_audit(input_path, str(rdir), cfg, _qc.CoverageParams(),
                              marker=marker, voxel_um=voxel_um)
                misses = _read_misses(rdir)
            except Exception as e:
                if verbose:
                    print(f"  (could not auto-run coverage audit: {e})")

    out = Path(output) if output else rdir / "zviewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_zslice_html(marker, voxel_um, centroids, out,
                      title=f"{channel_name} — z-slice viewer "
                            f"({n_z} planes, {len(centroids)} neurons)",
                      misses=misses, stable_threshold=stable_threshold,
                      source_name=Path(input_path).name)
    if verbose:
        print(f"z-slice viewer written: {out}")
        print(f"  channel : ch{marker_idx} ({channel_name})")
        print(f"  planes  : {n_z}  ({marker.shape[2]}×{marker.shape[1]} px, "
              f"z-step {voxel_um[0]:.3g} µm)")
        print(f"  overlay : {len(centroids)} neuron centroids"
              + (" (disabled)" if no_overlay else ""))
        print(f"  misses  : {len(misses)} candidate-miss markers"
              + (" (disabled)" if no_misses else ""))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="neuron_counter.viz_zslice",
        description="Build a standalone HTML z-slice viewer that smoothly "
                    "interpolates between z-planes (ZEN/Fiji-style), with an "
                    "optional overlay of detected neuron centroids.",
    )
    p.add_argument("input", help="path to the .czi file")
    p.add_argument("--results", default="results",
                   help="directory holding neurons.csv for the overlay (default: results)")
    p.add_argument("-o", "--output", default=None,
                   help="output html path (default: <results>/zviewer.html)")
    p.add_argument("-c", "--config", help="path to config.yaml")
    p.add_argument("--marker-index", type=int, default=None,
                   help="override marker channel index")
    p.add_argument("--no-overlay", action="store_true",
                   help="do not embed detected centroids")
    p.add_argument("--no-misses", action="store_true",
                   help="do not embed the coverage-audit candidate misses")
    args = p.parse_args(argv)

    generate_viewer(args.input, results_dir=args.results, output=args.output,
                    config_path=args.config, marker_index=args.marker_index,
                    no_overlay=args.no_overlay, no_misses=args.no_misses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
