"""Thin a mask to a skeleton and place it on the full-size image grid (for the 3D lift)."""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

IMAGE_SHAPE = (480, 640)


def thin(mask) -> np.ndarray:
    """Morphological skeleton (1-pixel-wide centre line) of a boolean mask."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    return skeletonize(mask)


def mask_to_image_grid(mask, mode: str, image_shape=IMAGE_SHAPE) -> np.ndarray:
    """Map a 256x256 mask onto the 480x640 image grid (nearest neighbour).

    ``stretch``: the mask covers the whole image. ``centre``: it covers the central
    480x480 square and the side strips are False. Which mapping is right is
    UNCONFIRMED for this dataset.
    """
    h, w = image_shape
    m = Image.fromarray(np.asarray(mask, bool).astype(np.uint8) * 255)
    if mode == "stretch":
        return np.array(m.resize((w, h), Image.NEAREST)) > 0
    if mode == "centre":
        out = np.zeros((h, w), bool)
        left = (w - h) // 2
        out[:, left:left + h] = np.array(m.resize((h, h), Image.NEAREST)) > 0
        return out
    raise ValueError("mode must be 'stretch' or 'centre'")
