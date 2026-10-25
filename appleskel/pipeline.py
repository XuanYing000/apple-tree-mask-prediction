"""The evaluation protocol used throughout the project.

For every method: fit on the training rows, choose the probability threshold on
the validation rows, then score once on the test rows. Scores are per-tree
Dice / IoU (and optionally boundary F1) at 256x256.
"""
from __future__ import annotations

import numpy as np

from .features import downsample_masks, upsample_probs
from .metrics import boundary_f1, dice, iou, tune_threshold
from .ridge import RidgeRegressor
from .shape_model import ShapeModel

THRESHOLD_GRID = np.linspace(0.05, 0.9, 18)   # steps of 0.05; same grid for every model


def score_split(prob_val, gt_val, prob_test, gt_test, grid=THRESHOLD_GRID, tol_px=None):
    """Tune the threshold on validation, then score the test set.

    Parameters
    ----------
    prob_val, prob_test : (n, 256, 256) predicted probabilities (or any scores)
    gt_val, gt_test : (n, 256, 256) boolean ground truth
    tol_px : float or None
        If given, also compute boundary F1 with this pixel tolerance.

    Returns
    -------
    dict with ``threshold``, ``val_dice`` (mean), and per-tree arrays
    ``test_dice``, ``test_iou`` and ``test_boundary_f1`` (None if ``tol_px`` is None).
    """
    thr, val_dice = tune_threshold(prob_val, gt_val, grid)
    pred = [prob_test[i] >= thr for i in range(len(gt_test))]
    td = np.array([dice(p, g) for p, g in zip(pred, gt_test)])
    ti = np.array([iou(p, g) for p, g in zip(pred, gt_test)])
    bf = (np.array([boundary_f1(p, g, tol_px) for p, g in zip(pred, gt_test)])
          if tol_px is not None else None)
    return {"threshold": thr, "val_dice": val_dice, "test_dice": td,
            "test_iou": ti, "test_boundary_f1": bf}


def run_mean_mask(Y, tr, va, te, grid=THRESHOLD_GRID, tol_px=None):
    """Baseline that ignores the image: threshold the mean training mask.

    Parameters
    ----------
    Y : (N, 256, 256) array of {0, 1}
    tr, va, te : index arrays from :func:`appleskel.splits.split_by_rows`
    """
    Yb = np.asarray(Y) > 0
    mean_mask = Yb[tr].mean(axis=0)
    prob = lambda ids: np.broadcast_to(mean_mask, (len(ids),) + mean_mask.shape)
    return score_split(prob(va), Yb[va], prob(te), Yb[te], grid, tol_px)


def run_shape_ridge(F, Y, tr, va, te, k=20, alpha=1e4, size=64, grid=THRESHOLD_GRID, tol_px=None):
    """SVD shape model + ridge regression from image features to shape coefficients.

    Steps: (1) block-average masks to ``size x size``; (2) fit a ``k``-component
    shape model on the training masks; (3) fit ridge from features ``F`` to the
    training coefficients; (4) predict validation/test masks, enlarge to
    256x256 and score with :func:`score_split`.

    Parameters
    ----------
    F : (N, p) features, e.g. from :func:`appleskel.features.pool_features`
    Y : (N, 256, 256) array of {0, 1}
    """
    Y = np.asarray(Y)
    Yb = Y > 0
    Y_small = downsample_masks(Y, size).reshape(len(Y), -1).astype(np.float64)
    shape = ShapeModel(k).fit(Y_small[tr])
    ridge = RidgeRegressor(alpha).fit(F[tr], shape.project(Y_small[tr]))
    to_prob = lambda ids: upsample_probs(shape.reconstruct(ridge.predict(F[ids])), size, Y.shape[1])
    return score_split(to_prob(va), Yb[va], to_prob(te), Yb[te], grid, tol_px)
