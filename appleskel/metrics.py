"""Segmentation metrics and threshold tuning.

All masks are boolean arrays of identical shape. The ground-truth masks are
thick shapes (about 6-10 % of the pixels), so Dice and IoU punish small
sideways shifts hard; ``boundary_f1`` adds a tolerance of a few pixels.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _check(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    return pred, gt


def dice(pred, gt) -> float:
    """Dice coefficient ``2|A and B| / (|A| + |B|)``; two empty masks give 1.0."""
    pred, gt = _check(pred, gt)
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2.0 * (pred & gt).sum() / total)


def iou(pred, gt) -> float:
    """Intersection over union; two empty masks give 1.0."""
    pred, gt = _check(pred, gt)
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


def _boundary(mask):
    """One-pixel inner boundary; the image border does not count as boundary."""
    return mask & ~binary_erosion(mask, border_value=1)


def boundary_f1(pred, gt, tol_px: float = 3.0) -> float:
    """Boundary F1 with a pixel tolerance.

    Precision is the share of predicted boundary pixels within ``tol_px`` of a
    ground-truth boundary pixel; recall is the reverse. Returns 1.0 for two
    empty masks and 0.0 if only one is empty.
    """
    pred, gt = _check(pred, gt)
    if tol_px < 0:
        raise ValueError("tol_px must be >= 0")
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        return 1.0
    if not bp.any() or not bg.any():
        return 0.0
    d_to_gt = distance_transform_edt(~bg)
    d_to_pred = distance_transform_edt(~bp)
    precision = float((d_to_gt[bp] <= tol_px).mean())
    recall = float((d_to_pred[bg] <= tol_px).mean())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def tune_threshold(prob, gt, grid):
    """Pick the probability threshold with the best mean Dice (first on ties).

    Use VALIDATION data only. Returns ``(threshold, mean_dice)``.
    """
    best_thr, best = None, -1.0
    for th in grid:
        d = float(np.mean([dice(prob[i] >= th, gt[i]) for i in range(len(gt))]))
        if d > best:
            best_thr, best = float(th), d
    return best_thr, best
