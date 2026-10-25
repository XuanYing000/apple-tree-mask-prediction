"""EXPLORATORY: lift masked depth pixels to 3D.

WARNING: the dataset provides neither camera intrinsics nor the physical unit
of the depth values. ``hfov_deg`` and ``depth_unit_m`` are therefore
ASSUMPTIONS, and nothing produced here can be validated.

What the assumptions do (see tests):

* ``depth_unit_m`` multiplies every coordinate, so it only changes the overall
  SIZE of the cloud, never its shape.
* ``hfov_deg`` changes the sideways / vertical extent relative to depth, so it
  changes the proportions of the cloud.
* Depth is assumed linear in distance (it could be scaled or inverse).
* In summer, the depth at a branch pixel is probably the depth of the leaves in
  front of the branch, so lifted points may lie on foliage, not on wood.
"""
from __future__ import annotations

import numpy as np


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float):
    """Pinhole intrinsics ``(fx, fy, cx, cy)`` from an assumed horizontal field of view."""
    if not 1.0 <= hfov_deg <= 179.0:
        raise ValueError("hfov_deg must be between 1 and 179")
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    return fx, fx, width / 2.0, height / 2.0       # square pixels: fy = fx


def depth_to_points(depth, mask=None, hfov_deg: float = 60.0, depth_unit_m: float = 0.02):
    """Back-project depth pixels to 3D points (ASSUMED intrinsics and scale).

    Parameters
    ----------
    depth : (H, W) array of grey values; 0 means "no measurement" and is dropped.
    mask : (H, W) bool, optional
        Only pixels inside the mask are lifted.
    hfov_deg, depth_unit_m : float
        ASSUMED horizontal field of view and metres per grey level.

    Returns
    -------
    (n, 3) float array of ``(x right, y up, z away from camera)``.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {depth.shape}")
    if not depth_unit_m > 0:
        raise ValueError("depth_unit_m must be > 0")
    ok = depth > 0
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != depth.shape:
            raise ValueError(f"mask {mask.shape} and depth {depth.shape} differ")
        ok &= mask
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics_from_hfov(w, h, hfov_deg)
    v, u = np.nonzero(ok)
    z = depth[v, u] * depth_unit_m
    return np.column_stack([(u - cx) * z / fx, -(v - cy) * z / fy, z])


def quantisation_bound_m(depth_unit_m: float) -> float:
    """Largest depth error caused by the 8-bit grey levels alone: half a level."""
    return depth_unit_m / 2.0
