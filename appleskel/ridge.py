"""Ridge regression in dual (kernel) form, implemented from scratch.

With ``n`` training samples and ``p`` features (here ``n`` ~ 400, ``p`` ~ 1000)
the dual form solves an ``n x n`` system instead of ``p x p``:

    W = A^T (A A^T + alpha I)^{-1} (C - c0)      (dual)
      = (A^T A + alpha I)^{-1} A^T (C - c0)      (primal; identical result)

where ``A`` is the standardised feature matrix and ``c0`` the mean target.
Everything is computed in float64. ``alpha > 0`` keeps ``A A^T + alpha I``
positive definite, so the solve is stable even when features are collinear.
"""
from __future__ import annotations

import numpy as np


class RidgeRegressor:
    """Standardise features, then fit regularised least squares to ``C``.

    Parameters
    ----------
    alpha : float
        Regularisation strength (> 0). Larger values shrink predictions
        towards the mean target.
    """

    def __init__(self, alpha: float):
        if not alpha > 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)

    def fit(self, F, C):
        """Fit on features ``F`` (n, p) and targets ``C`` (n, k)."""
        F, C = np.asarray(F, np.float64), np.asarray(C, np.float64)
        if F.ndim != 2 or C.ndim != 2 or len(F) != len(C):
            raise ValueError(f"F {F.shape} and C {C.shape} must be 2-D with equal rows")
        if len(F) < 2:
            raise ValueError("need at least 2 samples")
        self.f_mean_ = F.mean(axis=0)
        self.f_std_ = F.std(axis=0) + 1e-6          # avoid division by zero for constant features
        A = (F - self.f_mean_) / self.f_std_
        self.c_mean_ = C.mean(axis=0)
        gram = A @ A.T                              # (n, n)
        coef = np.linalg.solve(gram + self.alpha * np.eye(len(A)), C - self.c_mean_)
        self.weights_ = A.T @ coef                  # (p, k)
        return self

    def predict(self, F):
        """Predict targets for ``F`` (m, p); ``p`` must match the training data."""
        F = np.asarray(F, np.float64)
        if F.ndim != 2 or F.shape[1] != len(self.f_mean_):
            raise ValueError(f"F must have {len(self.f_mean_)} columns, got shape {F.shape}")
        return ((F - self.f_mean_) / self.f_std_) @ self.weights_ + self.c_mean_
