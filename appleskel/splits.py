"""Row-wise train / validation / test splits.

Trees in the same orchard row are planted, trained and pruned alike, so they
look very similar. Splitting by *whole rows* keeps near-copies out of the
training set when a tree is in the test set (a random tree-level split would
leak). Thresholds and model choices must be made on the validation rows only.
"""
from __future__ import annotations

import numpy as np

from .data import parse_tree_id


def split_by_rows(tree_ids, seed: int = 0, test_frac: float = 0.20, val_frac: float = 0.15):
    """Assign whole rows to train / validation / test.

    Parameters
    ----------
    tree_ids : sequence of str
        IDs such as ``"R05N34"``; the row is the number after ``R``.
    seed : int
        Seed of the row shuffle. Seed 0 reproduces the notebook's split.
    test_frac, val_frac : float
        Approximate fractions of *rows* (not trees) for test and validation.

    Returns
    -------
    tr, va, te : ndarray of int
        Indices into ``tree_ids``. The three sets never share a row.

    Raises
    ------
    ValueError
        Fractions are invalid, or there are too few rows for three non-empty sets.
    """
    if not (0 < test_frac < 1 and 0 < val_frac < 1 and test_frac + val_frac < 1):
        raise ValueError("need 0 < test_frac, val_frac and test_frac + val_frac < 1")
    row_of = np.array([parse_tree_id(t)[0] for t in tree_ids])
    rows = sorted(set(row_of.tolist()))
    n = len(rows)
    # floor with a tiny tolerance so e.g. 0.35 * 20 gives 7, not 6, despite float error
    a = int(np.floor(test_frac * n + 1e-9))
    b = int(np.floor((test_frac + val_frac) * n + 1e-9))
    if a < 1 or b - a < 1 or b >= n:
        raise ValueError(f"{n} rows are too few for test_frac={test_frac}, val_frac={val_frac}")
    shuffled = np.random.default_rng(seed).permutation(rows)
    te = np.where(np.isin(row_of, shuffled[:a]))[0]
    va = np.where(np.isin(row_of, shuffled[a:b]))[0]
    tr = np.setdiff1d(np.arange(len(row_of)), np.concatenate([te, va]))
    return tr, va, te
