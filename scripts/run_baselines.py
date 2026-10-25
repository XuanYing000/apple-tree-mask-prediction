"""Mean-mask baseline vs shape-model ridge over repeated row-wise splits.

    python scripts/run_baselines.py data/apple_cache.npz --mode centre --splits 10
Writes results/baselines.csv (one line per split and method).
"""
import argparse

import numpy as np

import _common
from appleskel.features import pool_features
from appleskel.pipeline import run_mean_mask, run_shape_ridge
from appleskel.splits import split_by_rows

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("cache"); ap.add_argument("--mode", default="centre", choices=["stretch", "centre"])
ap.add_argument("--splits", type=int, default=10); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--alpha", type=float, default=1e4); ap.add_argument("--tol", type=float, default=3.0)
ap.add_argument("--out", default="results/baselines.csv")
a = ap.parse_args()
trees, X, Y = _common.load(a.cache)
F = pool_features(X[a.mode], [0, 1, 2, 3], grid=16)
rows, gain = [], []
for seed in range(a.splits):
    tr, va, te = split_by_rows(trees, seed=seed)
    res = {"mean_mask": run_mean_mask(Y, tr, va, te, tol_px=a.tol),
           "shape_ridge": run_shape_ridge(F, Y, tr, va, te, k=a.k, alpha=a.alpha, tol_px=a.tol)}
    for name, r in res.items():
        rows.append({"seed": seed, "method": name, "n_test": len(te), "threshold": r["threshold"],
                     "dice": np.mean(r["test_dice"]), "iou": np.mean(r["test_iou"]),
                     "boundary_f1": np.mean(r["test_boundary_f1"])})
    gain.append(rows[-1]["dice"] - rows[-2]["dice"])
_common.write_csv(a.out, rows)
g = np.array(gain)
print(f"ridge - mean_mask Dice over {a.splits} splits: mean {g.mean():+.3f}, std {g.std():.3f}, "
      f"ridge better in {(g > 0).sum()}/{len(g)}")
for m in ("mean_mask", "shape_ridge"):
    d = [r["dice"] for r in rows if r["method"] == m]; b = [r["boundary_f1"] for r in rows if r["method"] == m]
    print(f"{m:12s} Dice {np.mean(d):.3f}+/-{np.std(d):.3f}   boundary-F1 {np.mean(b):.3f}")
