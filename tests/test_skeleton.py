"""Skeleton thinning and mask-to-image mapping."""
import numpy as np
import pytest

from appleskel.skeleton import mask_to_image_grid, thin


def test_thin_reduces_thick_bar_to_a_line():
    m = np.zeros((40, 40), bool); m[5:35, 18:23] = True      # 5 px wide vertical bar
    s = thin(m)
    assert 0 < s.sum() < m.sum() / 3 and (s & ~m).sum() == 0  # fewer pixels, all inside the mask
    assert s[10:30].any(axis=1).all()                        # still spans the bar's length


def test_mapping_modes():
    m = np.ones((256, 256), bool)
    assert mask_to_image_grid(m, "stretch").all()
    c = mask_to_image_grid(m, "centre")
    assert c.shape == (480, 640) and c[:, 80:560].all() and not c[:, :80].any() and not c[:, 560:].any()
    with pytest.raises(ValueError):
        mask_to_image_grid(m, "zoom")
    with pytest.raises(ValueError):
        thin(np.zeros((3, 3, 3), bool))
