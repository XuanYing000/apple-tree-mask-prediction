"""appleskel: predict apple-tree trunk/branch masks from single RGB-D images.

The package supports an MMA3001 project on the "Apple-tree reconstruction"
dataset. For each tree there is one summer RGB image, one depth image and a
256x256 binary ground-truth mask of the trunk and main branches (the structure
may be hidden behind leaves). The package provides:

``data``         scan / validate / load the dataset and build cached arrays
``splits``       row-wise train/validation/test splits (no neighbour leakage)
``metrics``      Dice, IoU, boundary F1 with pixel tolerance, threshold tuning
``features``     pooled image features, mask down/up-sampling
``shape_model``  SVD (PCA) shape model of the masks
``ridge``        dual-form ridge regression (own implementation)
``pipeline``     the evaluation protocol: mean-mask baseline and shape+ridge
``unet``         small U-Net (requires PyTorch, imported separately)
``lift3d``       exploratory depth lifting to 3D with ASSUMED intrinsics
``profiling``    timing, memory and analytic FLOP / parameter counts
``synthetic``    fake dataset generator for tests and smoke runs

Importing ``appleskel`` does not import PyTorch.
"""
__version__ = "0.1.0"
