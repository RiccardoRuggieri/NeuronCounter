"""
io_czi
======
Minimal, dependency-light loader for Carl Zeiss ``.czi`` z-stacks.

Returns a :class:`Stack`: the image collapsed to ``(C, Z, Y, X)`` plus the
physical voxel size in microns ``(z, y, x)`` and the channel names.

Readers are tried in order until one works:

1. ``aicspylibczi`` -- fast C-backed reader (if installed / compiled).
2. ``czifile``      -- pure-python reader (no compiler needed).

Only the reading is shared with the old code base; everything downstream
(the counting algorithm) is new.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

VoxelUM = Tuple[float, float, float]  # (z, y, x) microns


@dataclass
class Stack:
    data: np.ndarray            # (C, Z, Y, X)
    voxel_um: VoxelUM           # (z, y, x) microns
    channel_names: List[str]
    path: Path

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_z(self) -> int:
        return self.data.shape[1]

    def channel(self, index: int) -> np.ndarray:
        if not 0 <= index < self.n_channels:
            raise IndexError(
                f"channel index {index} out of range (0..{self.n_channels - 1})"
            )
        return self.data[index]

    def describe(self) -> str:
        zyx = " x ".join(f"{v:.4g}" for v in self.voxel_um)
        return (
            f"{self.path.name}\n"
            f"  channels : {self.n_channels} -> {self.channel_names}\n"
            f"  z-planes : {self.n_z}\n"
            f"  Y x X    : {self.data.shape[2]} x {self.data.shape[3]}\n"
            f"  dtype    : {self.data.dtype}\n"
            f"  voxel um : (z,y,x) = {zyx}"
        )


def load_czi(
    path: str | Path,
    channel_names: Optional[List[str]] = None,
    voxel_um_fallback: VoxelUM = (1.0, 0.69, 0.69),
) -> Stack:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CZI file not found: {path}")

    errors = []
    for reader in (_load_pylibczi, _load_czifile):
        try:
            return reader(path, channel_names, voxel_um_fallback)
        except ImportError as e:
            errors.append(f"{reader.__name__}: not available ({e})")
        except Exception as e:
            errors.append(f"{reader.__name__}: {type(e).__name__}: {e}")
    raise RuntimeError(
        "Could not read CZI with any reader:\n  " + "\n  ".join(errors)
        + "\nInstall a reader, e.g.:  pip install czifile"
    )


# --------------------------------------------------------------------------- #
def _load_pylibczi(path, channel_names, fallback) -> Stack:
    from aicspylibczi import CziFile

    czi = CziFile(str(path))
    img, shp = czi.read_image()
    axes = "".join(str(d) for d, _ in shp)
    czyx = _reduce_to_czyx(img, axes)
    meta = ""
    try:
        from lxml import etree
        meta = etree.tostring(czi.meta).decode("utf-8", "ignore")
    except Exception:
        pass
    voxel = _voxel_from_metadata(meta, fallback)
    names = channel_names or _names_from_metadata(meta, czyx.shape[0])
    return Stack(czyx, voxel, _pad_names(names, czyx.shape[0]), path)


def _load_czifile(path, channel_names, fallback) -> Stack:
    import czifile

    with czifile.CziFile(str(path)) as czi:
        arr = czi.asarray()
        axes = czi.axes
        meta = czi.metadata()
    czyx = _reduce_to_czyx(arr, axes)
    voxel = _voxel_from_metadata(meta, fallback)
    names = channel_names or _names_from_metadata(meta, czyx.shape[0])
    return Stack(czyx, voxel, _pad_names(names, czyx.shape[0]), path)


# --------------------------------------------------------------------------- #
def _reduce_to_czyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """Collapse an arbitrary CZI axis order to ``(C, Z, Y, X)``."""
    axes = axes.upper()
    keep = {"C", "Z", "Y", "X"}
    for a in [ax for ax in axes if ax not in keep]:
        pos = axes.index(a)
        arr = np.take(arr, 0, axis=pos)
        axes = axes.replace(a, "", 1)
    for needed in ("Z", "C"):
        if needed not in axes:
            arr = arr[np.newaxis, ...]
            axes = needed + axes
    order = [axes.index(a) for a in "CZYX"]
    return np.ascontiguousarray(np.transpose(arr, order))


def _voxel_from_metadata(meta: str, fallback: VoxelUM) -> VoxelUM:
    vz, vy, vx = fallback
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(meta)
        for dist in root.iter("Distance"):
            axis = (dist.get("Id") or "").upper()
            val = dist.findtext("Value")
            if val is None:
                continue
            microns = float(val) * 1e6  # metres -> microns
            if axis == "X":
                vx = microns
            elif axis == "Y":
                vy = microns
            elif axis == "Z":
                vz = microns
    except Exception:
        pass
    return (float(vz), float(vy), float(vx))


def _names_from_metadata(meta: str, n: int) -> List[str]:
    names: List[str] = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(meta)
        for ch in root.iter("Channel"):
            nm = ch.findtext("Fluor") or ch.findtext("DyeName") or ch.get("Name")
            if nm and nm not in names:
                names.append(nm)
        names = names[:n]
    except Exception:
        pass
    return names


def _pad_names(names: List[str], n: int) -> List[str]:
    names = list(names)[:n]
    while len(names) < n:
        names.append(f"Ch{len(names)}")
    return names
