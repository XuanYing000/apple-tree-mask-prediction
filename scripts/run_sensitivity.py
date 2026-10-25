"""Sensitivity of the ridge model to alpha, K, feature grid and training size; timing and FLOPs.

    python scripts/run_sensitivity.py data/apple_cache.npz --mode centre
Writes results/sensitivity.csv and results/performance.csv.
"""
import argparse

import numpy as np

import _common
from appleskel.features import pool_features
from appleskel.pipeline import run_shape_ridge
from appleskel.profiling import peak_memory_mb, ridge_flops, time_call
from appleskel.splits import split_by_rows

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("cache"); ap.add_argument("--mode", default="centre", choices=["stretch", "centre"])
ap.add_argument("--splits", type=int, default=3); ap.add_argument("--outdir", default="results")
a = ap.parse_args()
trees, X, Y = _common.load(a.cache)
Xm = X[a.mode]
F16 = pool_features(Xm, [0, 1, 2, 3], grid=16)
splits = [split_by_rows(trees, seed=s) for s in range(a.splits)]


def mean_dice(F, k=20, alpha=1e4, frac=1.0):
    d = []
    for i, (tr, va, te) in enumerate(splits):
        if frac < 1:                                   # random subset of training trees (all rows stay in train)
            tr = np.sort(np.random.default_rng(i).permutation(tr)[: max(int(len(tr) * frac), 5)])
        k_eff = min(k, len(tr) - 1)                    # k cannot exceed the number of training trees
        d.append(np.mean(run_shape_ridge(F, Y, tr, va, te, k=k_eff, alpha=alpha)["test_dice"]))
    return float(np.mean(d)), float(np.std(d))


rows = []
def add(factor, value, F=F16, **kw):
    m, s = mean_dice(F, **kw); rows.append({"factor": factor, "value": value, "dice_mean": m, "dice_std_over_splits": s})
    print(f"{factor:10s} {value!s:8s} Dice {m:.3f} +/- {s:.3f}")

for al in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6): add("alpha", al, alpha=al)
min_train = min(len(sp[0]) for sp in splits)
for k in (2, 5, 10, 20, 40, 80):
    if k < min_train: add("K", k, k=k)
    else: print(f"K        {k}  skipped (needs fewer components than the {min_train} training trees)")
for g in (4, 8, 16, 32): add("grid", g, F=pool_features(Xm, [0, 1, 2, 3], grid=g))
for name, ch in (("rgb", [0, 1, 2]), ("depth", [3]), ("rgbd", [0, 1, 2, 3])): add("channels", name, F=pool_features(Xm, ch, 16))
for fr in (0.1, 0.25, 0.5, 1.0): add("train_frac", fr, frac=fr)
_common.write_csv(f"{a.outdir}/sensitivity.csv", rows)

tr, va, te = splits[0]
perf = []
n, p, k = len(tr), F16.shape[1], 20
t = time_call(lambda: run_shape_ridge(F16, Y, tr, va, te, k=k), repeats=3)
_, mb = peak_memory_mb(lambda: run_shape_ridge(F16, Y, tr, va, te, k=k))
fl = ridge_flops(n, p, k, m=len(va) + len(te))
perf.append({"item": "shape_ridge fit+predict", "n_train": n, "p": p, "k": k, "best_s": t["best_s"],
             "peak_MB": mb, "flops_model": fl, "gflops_per_s": fl / t["best_s"] / 1e9})
_common.write_csv(f"{a.outdir}/performance.csv", perf)
print(perf[0])
