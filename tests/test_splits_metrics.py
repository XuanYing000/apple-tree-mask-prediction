"""Row-wise splitting (leakage) and metric hand-checks."""
import numpy as np
import pytest

from appleskel.metrics import boundary_f1, dice, iou, tune_threshold
from appleskel.splits import split_by_rows


def _ids(rows=8, per=6):
    return [f"R{r:02d}N{n:02d}" for r in range(1, rows + 1) for n in range(1, per + 1)]


def test_no_row_shared_between_splits_and_partition_complete():
    ids = _ids()
    tr, va, te = split_by_rows(ids, seed=1)
    row = lambda idx: {ids[i][:3] for i in idx}
    assert not (row(tr) & row(va)) and not (row(tr) & row(te)) and not (row(va) & row(te))
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(len(ids)))


def test_split_is_deterministic_and_seed_dependent():
    ids = _ids()
    a, b = split_by_rows(ids, seed=3), split_by_rows(ids, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = split_by_rows(ids, seed=4)
    assert not np.array_equal(a[2], c[2])


def test_regression_known_test_rows_seed0():
    # Value recorded from the verified notebook run (20 rows R01..R20, seed 0).
    ids = [f"R{r:02d}N01" for r in range(1, 21)]
    _, _, te = split_by_rows(ids, seed=0)
    assert sorted(int(ids[i][1:3]) for i in te) == [3, 5, 7, 20]


def test_split_bad_inputs():
    with pytest.raises(ValueError):
        split_by_rows(_ids(), test_frac=0.9, val_frac=0.5)
    with pytest.raises(ValueError):
        split_by_rows(["R01N01", "R01N02"])          # only one row


def test_dice_iou_hand_cases():
    a = np.zeros((4, 4), bool); a[:2] = True          # 8 px
    b = np.zeros((4, 4), bool); b[1:3] = True         # 8 px, overlap 4
    assert dice(a, b) == pytest.approx(2 * 4 / 16)
    assert iou(a, b) == pytest.approx(4 / 12)
    assert dice(a, a) == 1.0 and dice(a, ~a) == 0.0
    z = np.zeros((4, 4), bool)
    assert dice(z, z) == 1.0 and iou(z, z) == 1.0     # both empty
    assert dice(a, z) == 0.0


def test_metric_shape_mismatch():
    with pytest.raises(ValueError):
        dice(np.zeros((3, 3), bool), np.zeros((4, 4), bool))


def test_boundary_f1_tolerance():
    gt = np.zeros((64, 64), bool); gt[:, 30:33] = True
    shifted = np.zeros_like(gt); shifted[:, 32:35] = True   # 2 px off
    assert dice(shifted, gt) < 0.5
    assert boundary_f1(shifted, gt, tol_px=3) == pytest.approx(1.0)
    assert boundary_f1(shifted, gt, tol_px=0.5) < 0.6
    assert boundary_f1(gt, gt, tol_px=1) == pytest.approx(1.0)
    far = np.zeros_like(gt); far[:, 55:58] = True
    assert boundary_f1(far, gt, tol_px=3) == 0.0


def test_tune_threshold_picks_best():
    gt = np.zeros((2, 8, 8), bool); gt[:, :4] = True
    prob = np.where(gt, 0.7, 0.2)
    thr, d = tune_threshold(prob, gt, [0.1, 0.5, 0.9])
    assert thr == 0.5 and d == 1.0
