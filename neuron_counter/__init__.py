"""
neuron_counter
==============
Count neurons (HuC/D+ cell bodies) in 3D confocal z-stacks stored as Carl
Zeiss ``.czi`` files.

Method: model the 3D point cloud of bright voxels as a *non-parametric*
Gaussian kernel-density estimate (KDE), then count the **modes** (local maxima)
of that density. No deep network, no per-plane segmentation, no instance
stitching.
"""

from .io_czi import Stack, load_czi
from .pipeline import run_pipeline, PipelineResult

__all__ = ["load_czi", "Stack", "run_pipeline", "PipelineResult"]
__version__ = "1.0.0"
