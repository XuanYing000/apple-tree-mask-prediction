"""3D lift invariants and analytic cost formulas."""
import numpy as np
import pytest

from appleskel.lift3d import depth_to_points, intrinsics_from_hfov, quantisation_bound_m
from appleskel.profiling import peak_memory_mb, ridge_flops, time_call, unet_flops, unet_params


def test_intrinsics_from_hfov():
    fx, fy, cx, cy = intrinsics_from_hfov(640, 480, 90.0)
    assert fx == pytest.approx(320.0) and fx == fy
    assert (cx, cy) == (320.0, 240.0)       # principal point at the image centre


def test_centre_pixel_lies_on_optical_axis_and_zero_depth_dropped():
    d = np.zeros((6, 6)); d[3, 3] = 10                # pixel (3, 3) sits on the principal point (3, 3)
    pts = depth_to_points(d, hfov_deg=60, depth_unit_m=0.1)
    assert pts.shape == (1, 3)                        # zero-depth pixels were dropped
    assert np.allclose(pts[0], [0, 0, 1.0])


def test_depth_unit_scales_cloud_uniformly():
    rng = np.random.default_rng(0)
    d = rng.integers(1, 255, (20, 30)).astype(float)
    a, b = depth_to_points(d, depth_unit_m=0.01), depth_to_points(d, depth_unit_m=0.03)
    assert np.allclose(b, 3 * a)


def test_mask_selects_points_and_validation():
    d = np.full((6, 6), 5.0); m = np.zeros((6, 6), bool); m[1:3, 1:3] = True
    assert len(depth_to_points(d, m)) == 4
    with pytest.raises(ValueError):
        depth_to_points(d, np.zeros((3, 3), bool))
    with pytest.raises(ValueError):
        depth_to_points(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        depth_to_points(d, depth_unit_m=0)
    assert quantisation_bound_m(0.02) == pytest.approx(0.01)


def test_unet_param_count_matches_hand_calculation():
    assert unet_params(4) == 1_227_665
    assert unet_params(3) < unet_params(4)


def test_flops_formulas():
    assert ridge_flops(10, 20, 3) == pytest.approx(2 * 100 * 20 + (2 / 3) * 1000 + 2 * 100 * 3 + 2 * 10 * 20 * 3)
    assert ridge_flops(10, 20, 3, m=5) > ridge_flops(10, 20, 3)
    assert unet_flops(128, 128, 4) == pytest.approx(4 * unet_flops(64, 64, 4), rel=1e-9)
    assert 1.0e9 < unet_flops(128, 128, 4) < 3.0e9


def test_timing_and_memory_helpers():
    t = time_call(lambda: sum(range(1000)), repeats=3)
    assert 0 <= t["best_s"] <= t["mean_s"]
    res, mb = peak_memory_mb(lambda: np.ones((1000, 1000)))
    assert res.shape == (1000, 1000) and mb > 5
