"""Exploratory 3D skeleton for one tree: mask -> skeleton -> lifted points (PLY + 3-view figure).

Uses the ground-truth mask by default, or a predicted mask saved as .npy (256x256 bool).
ALL geometry is ASSUMED (field of view, depth unit, mask-to-image mapping): the shape is
indicative only and is NOT validated against any 3D ground truth.

    python scripts/lift_to_3d.py "/path/to/Dataset (With Summer GT)" R01N01 --mode centre
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from appleskel.data import load_depth, load_mask, scan_dataset
from appleskel.lift3d import depth_to_points, quantisation_bound_m
from appleskel.skeleton import mask_to_image_grid, thin

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("dataset"); ap.add_argument("tree_id")
ap.add_argument("--mode", default="centre", choices=["stretch", "centre"])
ap.add_argument("--mask-npy", help="predicted 256x256 mask instead of ground truth")
ap.add_argument("--hfov", type=float, default=60.0); ap.add_argument("--depth-unit", type=float, default=0.02)
ap.add_argument("--outdir", default="results/3d")
a = ap.parse_args()

records, _ = scan_dataset(a.dataset)
rec = records[a.tree_id]
depth = load_depth(rec.files[("Summer", "D")])
mask = np.load(a.mask_npy).astype(bool) if a.mask_npy else load_mask(rec.files[("Summer", "GT")])
skel_img = mask_to_image_grid(thin(mask), a.mode)
mask_img = mask_to_image_grid(mask, a.mode)
# a 1-px skeleton can fall on a depth hole after resizing: use depth > 0 within a 1-px dilation of the mask
pts_skel = depth_to_points(depth, skel_img, a.hfov, a.depth_unit)
pts_mask = depth_to_points(depth, mask_img, a.hfov, a.depth_unit)
print(f"{a.tree_id}: mask pixels {int(mask_img.sum())}, with valid depth {len(pts_mask)} "
      f"({100 * len(pts_mask) / max(int(mask_img.sum()), 1):.0f}%); skeleton points {len(pts_skel)}")
print(f"ASSUMED hfov={a.hfov} deg, depth unit={a.depth_unit} m/level; quantisation bound +/-"
      f"{quantisation_bound_m(a.depth_unit) * 100:.1f} cm")
if len(pts_skel) == 0:
    raise SystemExit("no skeleton pixel has valid depth: nothing to export")

out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
ply = out / f"{a.tree_id}_{a.mode}_skeleton.ply"
with open(ply, "w") as f:
    f.write(f"ply\nformat ascii 1.0\ncomment ASSUMED hfov={a.hfov} depth_unit_m={a.depth_unit}\n"
            f"element vertex {len(pts_skel)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    np.savetxt(f, pts_skel, fmt="%.4f")
fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
for axi, (i, j, nx, ny) in zip(ax, [(0, 1, "x (right)", "y (up)"), (2, 1, "z (depth)", "y (up)"), (0, 2, "x (right)", "z (depth)")]):
    axi.scatter(pts_mask[:, i], pts_mask[:, j], s=1, c="0.8", label="masked depth")
    axi.scatter(pts_skel[:, i], pts_skel[:, j], s=3, c="crimson", label="skeleton")
    axi.set_xlabel(nx + " [m, assumed]"); axi.set_ylabel(ny); axi.set_aspect("equal")
ax[0].legend(markerscale=4)
fig.suptitle(f"{a.tree_id} ({a.mode}): exploratory, unvalidated, assumed camera")
fig.tight_layout(); png = out / f"{a.tree_id}_{a.mode}_3views.png"; fig.savefig(png, dpi=120)
print("wrote", ply, "and", png)
