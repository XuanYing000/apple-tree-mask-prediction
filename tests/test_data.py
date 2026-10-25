"""Dataset scanning, loading and invalid-input handling."""
import numpy as np
import pytest
from PIL import Image

from appleskel.data import (DatasetError, build_arrays, load_cache, load_depth, load_mask,
                            parse_filename, parse_tree_id, prepare_input, save_cache,
                            scan_dataset, summer_complete_ids)
from appleskel.synthetic import make_fake_dataset


def test_parse_tree_id_and_filename():
    assert parse_tree_id("R03N12") == (3, 12)
    info = parse_filename("R03N12_Summer_GT-12-00-01.png")
    assert (info["tree_id"], info["season"], info["kind"], info["embedded_id"]) == ("R03N12", "Summer", "GT", None)
    assert parse_filename("notes.txt") is None
    with pytest.raises(ValueError):
        parse_tree_id("tree7")


def test_scan_reports_embedded_id_mismatch(tmp_path):
    make_fake_dataset(tmp_path, rows=range(1, 3), per_row=3, n_odd_winter=2)
    records, issues = scan_dataset(tmp_path)
    assert len(records) == 6
    assert len(issues) == 4                      # 2 odd trees x (RGB + D)
    assert all("differs" in s for s in issues)
    assert len(summer_complete_ids(records)) == 6


def test_scan_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path / "missing")
    with pytest.raises(DatasetError):
        scan_dataset(tmp_path)                   # empty folder


def test_load_mask_rejects_non_binary_and_wrong_shape(tmp_path):
    ok = np.zeros((256, 256), np.uint8); ok[10:20] = 255
    Image.fromarray(ok).save(tmp_path / "ok.png")
    assert load_mask(tmp_path / "ok.png").dtype == bool
    grey = ok.copy(); grey[0, 0] = 77
    Image.fromarray(grey).save(tmp_path / "grey.png")
    with pytest.raises(DatasetError):
        load_mask(tmp_path / "grey.png")
    Image.fromarray(ok[:100, :100]).save(tmp_path / "small.png")
    with pytest.raises(DatasetError):
        load_mask(tmp_path / "small.png")


def test_load_depth_all_zero_is_error(tmp_path):
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(tmp_path / "d.png")
    with pytest.raises(DatasetError):
        load_depth(tmp_path / "d.png")


def test_prepare_input_modes_and_errors():
    rgb = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 50, np.uint8)
    for mode in ("stretch", "centre"):
        out = prepare_input(rgb, depth, mode, size=128)
        assert out.shape == (128, 128, 4) and out.dtype == np.uint8
    with pytest.raises(ValueError):
        prepare_input(rgb, depth, "zoom")


def test_centre_crop_ignores_side_strips():
    rgb = np.zeros((480, 640, 3), np.uint8); rgb[:, :80] = 255   # left strip is outside the crop
    depth = np.full((480, 640), 9, np.uint8)
    out = prepare_input(rgb, depth, "centre", size=64)
    assert out[..., :3].max() == 0


def test_build_arrays_skips_or_raises_on_bad_tree(tmp_path):
    ids = make_fake_dataset(tmp_path, rows=range(1, 3), per_row=3)
    bad = tmp_path / f"{ids[0]}_Summer_GT-{next(tmp_path.glob(ids[0] + '_Summer_GT-*')).name.split('GT-')[1]}"
    arr = np.zeros((256, 256), np.uint8); arr[5, 5] = 100; arr[6, 6] = 200   # 3 grey levels
    Image.fromarray(arr).save(bad)
    records, _ = scan_dataset(tmp_path)
    out = build_arrays(records, summer_complete_ids(records), on_error="skip")
    assert ids[0] in [s[0] for s in out["skipped"]] and len(out["trees"]) == 5
    with pytest.raises(DatasetError):
        build_arrays(records, summer_complete_ids(records), on_error="raise")


def test_cache_roundtrip(tmp_path):
    make_fake_dataset(tmp_path / "d", rows=range(1, 3), per_row=3)
    records, _ = scan_dataset(tmp_path / "d")
    out = build_arrays(records, summer_complete_ids(records))
    save_cache(tmp_path / "c.npz", out["trees"], out["X"], out["Y"])
    back = load_cache(tmp_path / "c.npz")
    assert list(back["trees"]) == list(out["trees"])
    assert np.array_equal(back["Y"], out["Y"])
