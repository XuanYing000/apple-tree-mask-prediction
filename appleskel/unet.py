"""Small U-Net for mask prediction (requires PyTorch).

Input: ``(N, c, 128, 128)`` float tensors in [0, 1] (R, G, B, D channels).
Output: one logit map per image at the same resolution; probabilities are
enlarged bilinearly to 256x256 before scoring against the ground truth.

Architecture: four 2x poolings (128 -> 8), channels 16-32-64-128-128, two 3x3
convolutions + BatchNorm + ReLU per block, bilinear upsampling with skip
connections. With 4 input channels it has 1,227,665 trainable parameters
(see :func:`appleskel.profiling.unet_params`).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import dice


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net with four down/up levels; input height and width must be divisible by 16."""

    def __init__(self, cin: int, w: int = 16):
        super().__init__()
        ch = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        self.enc = nn.ModuleList([_block(cin, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(4)])
        self.dec = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in range(4)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        feats = []
        for i, enc in enumerate(self.enc):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = enc(x)
            feats.append(x)
        for i in range(3, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.dec[i](torch.cat([x, feats[i]], 1))
        return self.out(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def soft_dice_bce_loss(logits, target):
    """Binary cross-entropy plus ``1 - soft Dice`` (both averaged over the batch)."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - (2 * inter + 1) / (den + 1)).mean()


def make_inputs(X, channels) -> torch.Tensor:
    """``(N, S, S, 4)`` uint8 -> ``(N, len(channels), S, S)`` float32 in [0, 1]."""
    a = np.asarray(X)[..., list(channels)].astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def masks_to_size(Y, size: int = 128) -> torch.Tensor:
    """Block-average ``(N, 256, 256)`` masks to ``size`` and threshold at 0.5; shape (N, 1, size, size)."""
    Y = np.asarray(Y)
    n, h, _ = Y.shape
    f = h // size
    small = Y.reshape(n, size, f, size, f).mean(axis=(2, 4)) > 0.5
    return torch.from_numpy(small.astype(np.float32)).unsqueeze(1)


def augment_batch(xb, yb, n_rgb: int):
    """Random horizontal flip (image and mask together) and brightness jitter of the RGB channels only."""
    flip = (torch.rand(len(xb), device=xb.device) < 0.5)[:, None, None, None]
    xb, yb = torch.where(flip, xb.flip(3), xb), torch.where(flip, yb.flip(3), yb)
    if n_rgb:
        xb = xb.clone()
        xb[:, :n_rgb] *= 1 + 0.2 * (torch.rand(len(xb), 1, 1, 1, device=xb.device) - 0.5)
    return xb, yb


@torch.no_grad()
def predict_probs(model, inputs, ids, device, out: int = 256, batch_size: int = 32) -> np.ndarray:
    """Probabilities ``(len(ids), out, out)`` as float32 NumPy."""
    model.eval()
    chunks = []
    for i in range(0, len(ids), batch_size):
        xb = inputs[torch.as_tensor(ids[i:i + batch_size])].to(device)
        chunks.append(torch.sigmoid(model(xb)).cpu())
    p = F.interpolate(torch.cat(chunks), size=out, mode="bilinear", align_corners=False)
    return p[:, 0].numpy()


def train_unet(inputs, targets, gt256, train_idx, val_idx, n_rgb, epochs=40, seed=0,
               batch_size=16, lr=1e-3, device=None):
    """Train a :class:`UNet`; keep the weights of the epoch with the best validation Dice.

    Parameters
    ----------
    inputs : (N, c, 128, 128) tensor from :func:`make_inputs`
    targets : (N, 1, 128, 128) tensor from :func:`masks_to_size`
    gt256 : (N, 256, 256) boolean ground truth (for validation Dice at 0.5)
    train_idx, val_idx : index arrays; the epoch is chosen on ``val_idx`` only
    n_rgb : int
        Number of leading RGB channels in ``inputs`` (brightness jitter applies to them only).

    Returns
    -------
    model : UNet with the best-validation weights loaded (on ``device``)
    info : dict with ``train_seconds``, ``best_val_dice``, ``best_epoch``,
        ``history`` (validation Dice per epoch) and ``n_params``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet(inputs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), batch_size):
            b = torch.as_tensor(perm[i:i + batch_size])
            xb, yb = augment_batch(inputs[b].to(device), targets[b].to(device), n_rgb)
            opt.zero_grad()
            soft_dice_bce_loss(model(xb), yb).backward()
            opt.step()
        sched.step()
        pv = predict_probs(model, inputs, val_idx, device)
        vd = float(np.mean([dice(pv[k] >= 0.5, gt256[val_idx][k]) for k in range(len(val_idx))]))
        history.append(vd)
        if vd > best_val:
            best_val, best_epoch, best_state = vd, ep, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"train_seconds": time.perf_counter() - t0, "best_val_dice": best_val,
                   "best_epoch": best_epoch, "history": history, "n_params": count_parameters(model)}
