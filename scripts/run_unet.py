"""Train the U-Net on one input configuration and report test metrics (needs PyTorch + a GPU is helpful).

    python scripts/run_unet.py data/apple_cache.npz --mode centre --channels 0 1 2 3 --epochs 40 --seeds 3
Appends one line per seed to results/unet.csv.
"""
import argparse

import numpy as np

import _common
from appleskel.metrics import boundary_f1
from appleskel.pipeline import score_split
from appleskel.profiling import unet_flops
from appleskel.splits import split_by_rows
from appleskel.unet import make_inputs, masks_to_size, predict_probs, train_unet

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("cache"); ap.add_argument("--mode", default="centre", choices=["stretch", "centre"])
ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--tol", type=float, default=3.0); ap.add_argument("--out", default="results/unet.csv")
a = ap.parse_args()
trees, X, Y = _common.load(a.cache)
tr, va, te = split_by_rows(trees, seed=0)                 # same fixed split for every seed
inp, tgt, gt = make_inputs(X[a.mode], a.channels), masks_to_size(Y, 128), Y > 0
n_rgb = sum(c < 3 for c in a.channels)
rows = []
for seed in range(a.seeds):
    model, info = train_unet(inp, tgt, gt, tr, va, n_rgb, epochs=a.epochs, seed=seed)
    dev = next(model.parameters()).device
    r = score_split(predict_probs(model, inp, va, dev), gt[va], predict_probs(model, inp, te, dev), gt[te], tol_px=a.tol)
    rows.append({"mode": a.mode, "channels": "".join(map(str, a.channels)), "seed": seed,
                 "best_epoch": info["best_epoch"], "train_s": info["train_seconds"], "params": info["n_params"],
                 "gflops_fwd": unet_flops(128, 128, len(a.channels)) / 1e9, "threshold": r["threshold"],
                 "dice": np.mean(r["test_dice"]), "iou": np.mean(r["test_iou"]), "boundary_f1": np.mean(r["test_boundary_f1"])})
    print(rows[-1])
_common.write_csv(a.out, rows)
print(f"Dice over seeds: {np.mean([r['dice'] for r in rows]):.3f} +/- {np.std([r['dice'] for r in rows]):.3f}")
