# appleskel: trunk/branch mask prediction from single RGB-D apple-tree images

MMA3001 project, Dataset 1 (RGB-D apple-tree reconstruction and orchard mapping).

## What this does

Given **one summer RGB-D image** of a tree (640x480; depth as an 8-bit grey image,
0 = missing), predict the 256x256 **binary trunk/branch mask** supplied as summer ground truth.
Predictions are scored against that ground truth on trees from **rows never used for fitting**.

* Baseline: the mean training mask, thresholded (uses no image at all).
* Method A: SVD shape model of the masks + ridge regression from pooled image features (written from scratch, NumPy).
* Method B: small U-Net (PyTorch), BCE + soft Dice, flip and brightness augmentation.
* Exploratory only (not validated): lifting masked depth pixels to 3D with **assumed** camera and depth scale (`appleskel/lift3d.py`).

### Status of the results

All recorded numbers come from the notebook in `notebooks/`; regenerate them with the scripts below.

| Result (notebook run) | Value |
|---|---|
| Mean-mask baseline, test Dice | 0.371 +/- 0.087 (106 test trees, 4 test rows) |
| Ridge minus baseline, 10 random row splits | +0.020 mean, better in 10/10 splits |
| U-Net (one split, one seed), test Dice | 0.408 +/- 0.124 |
| Shape-model ceiling (best possible with K components) | about 0.72 |

Honest reading: every learned model sits close to the baseline and far from the ceiling. The U-Net has
**not** been shown to beat ridge (single split, single seed). `scripts/run_unet.py --seeds 3` is there to test that.

## Known data issues (verified on the supplied files)

* 42 winter files carry a different tree ID inside their file name; winter data is therefore not used.
* The brief mentions three seasons; the supplied data contains two (summer, winter).
* The mapping of the 256x256 masks onto the 640x480 images is not documented. Two candidates are
  implemented, `stretch` (whole image) and `centre` (central 480x480 crop); neither is confirmed.
* The depth unit is unknown and there are no camera intrinsics, so any 3D output uses assumed values.

## Install and run

```bash
pip install -r requirements.txt            # torch only needed for the U-Net
python -m pytest                            # verification tests (U-Net tests skip without torch)

# real data
python scripts/build_cache.py "/path/to/Dataset (With Summer GT)" data/apple_cache.npz
python scripts/run_baselines.py   data/apple_cache.npz --mode centre --splits 10
python scripts/run_sensitivity.py data/apple_cache.npz --mode centre
python scripts/run_unet.py        data/apple_cache.npz --mode centre --channels 0 1 2 3 --seeds 3

# exploratory 3D skeleton for one tree (assumed camera; writes PLY + 3-view PNG to results/3d)
python scripts/lift_to_3d.py "/path/to/Dataset (With Summer GT)" R01N01 --mode centre

# no data? try the pipeline on a synthetic stand-in (numbers are meaningless)
python scripts/make_fake_dataset.py data/fake
python scripts/build_cache.py data/fake data/fake_cache.npz
python scripts/run_baselines.py data/fake_cache.npz --splits 3
```

Results are written to `results/*.csv`. API documentation: `pdoc appleskel -o docs`, then open `docs/index.html`.

## Invalid input handling

`scan_dataset` raises `FileNotFoundError` for a missing folder and `DatasetError` for an empty folder or
duplicate files; unrecognised names and ID mismatches are reported as issues. `load_mask` rejects wrong-sized or
non-binary masks, `load_depth` rejects all-zero depth, and `build_arrays(on_error="skip")` records the failing
tree and continues (`"raise"` stops). Model code validates shapes, `alpha > 0`, `k` and fractions.

## Layout

```
appleskel/   data, splits, metrics, features, shape_model, ridge, pipeline, unet, skeleton, lift3d, profiling, synthetic
tests/       pytest verification tests (see below)
scripts/     one script per experiment
notebooks/   exploratory notebook (source of the numbers above)
AI_LOG.md    record of AI assistance
```

## What the tests check

Ridge dual form equals the primal form and scikit-learn Ridge; the SVD shape model equals scikit-learn PCA; Dice/IoU on
hand-computed cases; boundary F1 tolerance behaviour; splits never share a row and reproduce the notebook's
test rows; loader errors; 3D lift invariants (principal point on the axis, uniform scaling with depth unit);
the analytic U-Net parameter count (1,227,665) against the real module (torch only).
Test report: `python -m pytest --junitxml=results/test_report.xml`.

## Licence

MIT, see `LICENSE`. Set your name in the licence before submitting.
