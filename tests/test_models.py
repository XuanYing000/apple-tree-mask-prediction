"""Shape model, ridge regression and pipeline checked against independent references."""
import numpy as np
import pytest

from appleskel.features import downsample_masks, pool_features, upsample_probs
from appleskel.pipeline import run_mean_mask, run_shape_ridge
from appleskel.ridge import RidgeRegressor
from appleskel.shape_model import ShapeModel
from appleskel.splits import split_by_rows
from appleskel.synthetic import make_y_mask


def test_ridge_dual_equals_primal():
    rng = np.random.default_rng(0)
    F, C = rng.normal(size=(30, 12)), rng.normal(size=(30, 3))
    m = RidgeRegressor(alpha=2.5).fit(F, C)
    A = (F - F.mean(0)) / (F.std(0) + 1e-6)
    primal = np.linalg.solve(A.T @ A + 2.5 * np.eye(12), A.T @ (C - C.mean(0)))
    assert np.allclose(m.weights_, primal, atol=1e-9)


def test_ridge_matches_sklearn():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    F, C = rng.normal(size=(40, 9)), rng.normal(size=(40, 2))
    A = (F - F.mean(0)) / (F.std(0) + 1e-6)
    ref = sk.Ridge(alpha=3.0).fit(A, C)
    T = rng.normal(size=(5, 9))
    mine = RidgeRegressor(3.0).fit(F, C).predict(T)
    assert np.allclose(mine, ref.predict((T - F.mean(0)) / (F.std(0) + 1e-6)), atol=1e-8)


def test_ridge_constant_feature_and_bad_inputs():
    F = np.ones((10, 3)); F[:, 1] = np.arange(10)
    C = np.arange(10.0)[:, None]
    assert np.isfinite(RidgeRegressor(1.0).fit(F, C).predict(F)).all()
    with pytest.raises(ValueError):
        RidgeRegressor(0.0)
    with pytest.raises(ValueError):
        RidgeRegressor(1.0).fit(F, C[:5])
    m = RidgeRegressor(1.0).fit(F, C)
    with pytest.raises(ValueError):
        m.predict(np.ones((2, 4)))


def test_large_alpha_predicts_mean():
    rng = np.random.default_rng(2)
    F, C = rng.normal(size=(20, 5)), rng.normal(size=(20, 2))
    p = RidgeRegressor(1e12).fit(F, C).predict(F)
    assert np.allclose(p, C.mean(0), atol=1e-6)


def test_shape_model_matches_sklearn_pca():
    dec = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(3)
    M = rng.normal(size=(25, 40))
    mine = ShapeModel(5).fit(M)
    ref = dec.PCA(n_components=5, svd_solver="full").fit(M)
    assert np.allclose(mine.explained_variance_ratio_, ref.explained_variance_ratio_, atol=1e-10)
    for a, b in zip(mine.components_, ref.components_):      # equal up to sign
        assert abs(abs(a @ b) - 1) < 1e-8


def test_shape_model_full_rank_reconstructs_exactly():
    M = np.random.default_rng(4).normal(size=(10, 30))
    sm = ShapeModel(9).fit(M)                                # centred data has rank <= 9
    assert np.allclose(sm.reconstruct(sm.project(M)), M, atol=1e-9)


def test_feature_helpers():
    X = np.zeros((2, 8, 8, 4), np.uint8); X[0, :4, :4, 0] = 255
    f = pool_features(X, [0], grid=2)
    assert f.shape == (2, 4) and f[0, 0] == pytest.approx(1.0) and f[0, 1:].sum() == 0
    with pytest.raises(ValueError):
        pool_features(X, [0], grid=3)
    with pytest.raises(ValueError):
        pool_features(X, [7], grid=2)
    Y = np.zeros((1, 256, 256), bool); Y[0, :128] = True
    assert downsample_masks(Y, 64)[0, :32].all() and not downsample_masks(Y, 64)[0, 32:].any()
    assert upsample_probs(np.ones((1, 8, 8)), 8, out=256).shape == (1, 256, 256)


def test_pipeline_ridge_beats_mean_mask_on_synthetic_leaning_trees():
    rng = np.random.default_rng(5)
    n = 60
    ids = [f"R{r:02d}N{k:02d}" for r in range(1, 11) for k in range(1, 7)]
    lean = rng.uniform(-0.12, 0.12, n)
    Y = np.stack([make_y_mask(l) for l in lean])
    X = np.zeros((n, 32, 32, 4), np.uint8)
    for i, l in enumerate(lean):                              # depth stripe encodes the lean
        c = int(16 + l * 32 * 0.5 / 0.5)
        X[i, :, max(c - 3, 0):c + 3, 3] = 200
    F = pool_features(X, [3], grid=16)
    tr, va, te = split_by_rows(ids, seed=0)
    base = run_mean_mask(Y, tr, va, te)
    fit = run_shape_ridge(F, Y, tr, va, te, k=5, alpha=10.0, size=64)
    assert fit["test_dice"].mean() > base["test_dice"].mean()
    assert 0 <= fit["threshold"] <= 1
