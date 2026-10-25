"""Feature extraction and mask resampling."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def pool_features(X, channels, grid: int = 16) -> np.ndarray:
    """Average-pool selected channels onto a ``grid x grid`` lattice.

    Parameters
    ----------
    X : (N, S, S, C) uint8
        Inputs from :func:`appleskel.data.prepare_input` (channels R, G, B, D).
    channels : sequence of int
        Channel indices to keep, e.g. ``[0, 1, 2, 3]`` for RGB+D.
    grid : int
        Cells per side; ``S`` must be divisible by ``grid``.

    Returns
    -------
    (N, grid * grid * len(channels)) float32 in [0, 1]
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"X must be (N, S, S, C), got shape {X.shape}")
    n, h, w, c = X.shape
    if h != w or grid < 1 or h % grid:
        raise ValueError(f"image size {h}x{w} must be square and divisible by grid={grid}")
    channels = list(channels)
    if not channels or any(ch < 0 or ch >= c for ch in channels):
        raise ValueError(f"channels {channels} invalid for {c} channels")
    a = X[..., channels].astype(np.float32) / 255.0
    f = h // grid
    return a.reshape(n, grid, f, grid, f, len(channels)).mean(axis=(2, 4)).reshape(n, -1)


def downsample_masks(Y, size: int = 64) -> np.ndarray:
    """Block-average masks to ``size x size`` and threshold at 0.5 (majority vote).

    Parameters
    ----------
    Y : (N, H, H) array of {0, 1}; ``H`` must be divisible by ``size``.
    """
    Y = np.asarray(Y)
    n, h, w = Y.shape
    if h != w or size < 1 or h % size:
        raise ValueError(f"mask size {h}x{w} must be square and divisible by size={size}")
    f = h // size
    return Y.reshape(n, size, f, size, f).mean(axis=(2, 4)) > 0.5


def upsample_probs(P, size: int, out: int = 256) -> np.ndarray:
    """Bilinearly enlarge ``(n, size*size)`` or ``(n, size, size)`` maps to ``(n, out, out)``."""
    P = np.asarray(P, dtype=float)
    P = P.reshape(len(P), size, size)
    return np.stack([zoom(q, out / size, order=1) for q in P])
