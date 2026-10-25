"""Synthetic stand-in for the dataset (NOT real orchard data).

Used by the tests and by ``scripts/make_fake_dataset.py`` so that the whole
pipeline can be run without the real images. Each fake tree has a Y-shaped
mask whose lean is encoded in the depth image (a brighter vertical stripe), so
a model that reads depth can genuinely beat the mean-mask baseline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_y_mask(lean: float, size: int = 256) -> np.ndarray:
    """Boolean Y-shaped mask: a trunk plus two arms, leaning sideways by ``lean``."""
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1.0)
    cx = 0.5 + lean * (1.0 - yy)                       # lean grows towards the top
    trunk = np.abs(xx - cx) < 0.06
    arms = (yy < 0.5) & (np.abs(np.abs(xx - cx) - (0.5 - yy) * 0.8) < 0.04)
    return trunk | arms


def _blocky(rng, shape_small, scale, channels=None):
    """Compressible random image: random blocks enlarged by ``scale``."""
    small = rng.integers(0, 255, shape_small + ((channels,) if channels else ()), dtype=np.uint8)
    reps = (scale, scale) + ((1,) if channels else ())
    return np.kron(small, np.ones(reps, np.uint8))


def make_fake_dataset(root, rows=range(1, 9), per_row: int = 6, seed: int = 0, n_odd_winter: int = 2):
    """Write a fake dataset using the real file-naming scheme.

    Creates, for every tree, summer RGB / depth / ground-truth mask and winter
    RGB / depth. The first ``n_odd_winter`` trees get winter files whose names
    embed a different tree ID (mimicking the real data problem).

    Returns
    -------
    list of str
        Tree IDs written, in sorted order.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"R{r:02d}N{n:02d}" for r in rows for n in range(1, per_row + 1)]
    for i, tid in enumerate(sorted(ids)):
        lean = float(rng.uniform(-0.12, 0.12))
        stamp = f"12-{i // 60:02d}-{i % 60:02d}"
        rgb = _blocky(rng, (12, 16), 40, 3)                           # 480 x 640 x 3
        depth = np.full((480, 640), 100, np.uint8)
        col = int(320 + lean * 640 * 0.5)
        depth[:, max(col - 40, 0):col + 40] = 150                     # stripe encodes the lean
        depth[_blocky(rng, (12, 16), 40) < 25] = 0                    # missing-depth blocks
        depth[0, 0] = 100                                             # never all-zero
        mask = make_y_mask(lean)
        Image.fromarray(rgb).save(root / f"{tid}_Summer_RGB-{stamp}.png")
        Image.fromarray(np.dstack([depth] * 3)).save(root / f"{tid}_Summer_D-{stamp}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(root / f"{tid}_Summer_GT-{stamp}.png")
        tag = ""
        if i < n_odd_winter:
            r, n = int(tid[1:3]), int(tid[4:6])
            tag = f"R{r:02d}N{n + 10:02d}_RGB_"
        w_rgb = _blocky(rng, (12, 16), 40, 3)
        w_d = np.full((480, 640), 120, np.uint8)
        Image.fromarray(w_rgb).save(root / f"{tid}_Winter_RGB-{tag}{stamp}.png")
        Image.fromarray(np.dstack([w_d] * 3)).save(root / f"{tid}_Winter_D-{tag}{stamp}.png")
    return sorted(ids)
