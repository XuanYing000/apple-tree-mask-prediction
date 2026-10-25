"""SVD (principal-component) shape model of the ground-truth masks.

The masks are all variations of one shape (a trunk splitting into two arms),
so each mask can be summarised by a few coefficients along the principal
components of the training masks. The ``k``-component reconstruction of the
*true* test masks is a ceiling: no method that predicts only ``k`` numbers can
do better.
"""
from __future__ import annotations

import numpy as np


class ShapeModel:
    """Mean mask plus ``k`` principal components, computed with a thin SVD.

    Parameters
    ----------
    k : int
        Number of components to keep.

    Attributes
    ----------
    mean_ : (D,) mean flattened mask
    components_ : (k, D) orthonormal rows (principal directions; sign arbitrary)
    explained_variance_ratio_ : (k,) share of total variance per component
    """

    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def fit(self, M):
        """Fit on ``M`` of shape (N, D); needs ``N >= 2`` and ``k <= min(N, D)``."""
        M = np.asarray(M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] < 2:
            raise ValueError("M must be 2-D with at least 2 rows")
        if self.k > min(M.shape):
            raise ValueError(f"k={self.k} exceeds min(N, D)={min(M.shape)}")
        self.mean_ = M.mean(axis=0)
        _, s, vt = np.linalg.svd(M - self.mean_, full_matrices=False)
        energy = s ** 2
        total = energy.sum()
        self.components_ = vt[: self.k]
        self.explained_variance_ratio_ = (energy[: self.k] / total if total > 0
                                          else np.zeros(self.k))
        return self

    def project(self, M):
        """Coefficients ``(M - mean) @ components.T`` with shape (N, k)."""
        return (np.asarray(M, dtype=np.float64) - self.mean_) @ self.components_.T

    def reconstruct(self, C):
        """Masks (as continuous values) from coefficients: ``mean + C @ components``."""
        C = np.asarray(C, dtype=np.float64)
        if C.ndim != 2 or C.shape[1] != self.k:
            raise ValueError(f"C must have shape (N, {self.k}), got {C.shape}")
        return self.mean_ + C @ self.components_
