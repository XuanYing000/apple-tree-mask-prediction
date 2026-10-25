"""U-Net tests; skipped automatically when PyTorch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from appleskel.profiling import unet_params
from appleskel.unet import (UNet, augment_batch, count_parameters, make_inputs, masks_to_size,
                            soft_dice_bce_loss, train_unet)


def test_parameter_count_matches_analytic_formula():
    assert count_parameters(UNet(4)) == unet_params(4) == 1_227_665


def test_forward_shape():
    out = UNet(4)(torch.zeros(2, 4, 128, 128))
    assert out.shape == (2, 1, 128, 128)


def test_loss_is_small_for_perfect_prediction_and_large_for_wrong():
    y = torch.zeros(2, 1, 16, 16); y[:, :, :8] = 1
    good = soft_dice_bce_loss((y * 2 - 1) * 12, y)
    bad = soft_dice_bce_loss((1 - y * 2) * 12, y)
    assert good < 0.05 < bad


def test_augment_flips_image_and_mask_together_and_leaves_depth():
    x = torch.rand(8, 4, 16, 16); y = torch.rand(8, 1, 16, 16)
    x2, y2 = augment_batch(x, y, n_rgb=3)
    for i in range(8):
        same = torch.equal(y2[i], y[i]); flipped = torch.equal(y2[i], y[i].flip(-1))
        assert same or flipped
        d_same = torch.equal(x2[i, 3], x[i, 3]); d_flip = torch.equal(x2[i, 3], x[i, 3].flip(-1))
        assert (d_same and same) or (d_flip and flipped)


def test_masks_to_size_and_inputs():
    Y = np.zeros((2, 256, 256), bool); Y[:, :128] = True
    t = masks_to_size(Y, 128)
    assert t.shape == (2, 1, 128, 128) and t[0, 0, :64].min() == 1 and t[0, 0, 64:].max() == 0
    X = np.full((2, 128, 128, 4), 255, np.uint8)
    assert make_inputs(X, [0, 1, 2, 3]).shape == (2, 4, 128, 128)


def test_training_runs_and_returns_history():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, (12, 128, 128, 4), dtype=np.uint8)
    Y = np.zeros((12, 256, 256), bool); Y[:, :, 100:140] = True
    inp, tgt = make_inputs(X, [0, 1, 2, 3]), masks_to_size(Y, 128)
    model, info = train_unet(inp, tgt, Y, np.arange(8), np.arange(8, 12), n_rgb=3, epochs=2, device="cpu")
    assert len(info["history"]) == 2 and info["n_params"] == 1_227_665
