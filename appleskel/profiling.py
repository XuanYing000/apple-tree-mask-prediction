"""Timing, memory and analytic cost estimates.

The FLOP counts are leading-order estimates of arithmetic work (one multiply
plus one add = 2 FLOPs); they ignore memory traffic, so they indicate scaling,
not exact runtime.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np


def time_call(fn, repeats: int = 3) -> dict:
    """Run ``fn()`` ``repeats`` times; return ``best_s`` and ``mean_s`` (wall clock)."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"best_s": min(times), "mean_s": float(np.mean(times))}


def peak_memory_mb(fn):
    """Run ``fn()``; return ``(result, peak_MB)`` of Python/NumPy allocations.

    GPU memory and memory held by compiled libraries outside NumPy are not counted.
    """
    tracemalloc.start()
    try:
        result = fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, peak / 1e6


def ridge_flops(n: int, p: int, k: int, m: int = 0) -> float:
    """Leading-order FLOPs of :class:`appleskel.ridge.RidgeRegressor`.

    Gram matrix ``2 n^2 p``; LU solve ``(2/3) n^3 + 2 n^2 k``; weights
    ``2 n p k``; predicting ``m`` samples ``2 m p k``.
    """
    return 2.0 * n * n * p + (2.0 / 3.0) * n ** 3 + 2.0 * n * n * k + 2.0 * n * p * k + 2.0 * m * p * k


def _block_params(cin: int, cout: int) -> int:
    """Two 3x3 convolutions (with bias) and two BatchNorms (weight + bias each)."""
    return (9 * cin * cout + cout) + 2 * cout + (9 * cout * cout + cout) + 2 * cout


def _unet_channels(w0: int):
    return [w0, 2 * w0, 4 * w0, 8 * w0, 8 * w0]


def unet_params(cin: int, w0: int = 16) -> int:
    """Trainable parameters of :class:`appleskel.unet.UNet` computed analytically.

    ``unet_params(4) == 1_227_665``, the count PyTorch reported in the notebook.
    """
    ch = _unet_channels(w0)
    total = _block_params(cin, ch[0]) + sum(_block_params(ch[i], ch[i + 1]) for i in range(4))
    total += sum(_block_params(ch[i + 1] + ch[i], ch[i]) for i in range(4))
    total += ch[0] * 1 + 1                          # final 1x1 convolution with bias
    return total


def unet_flops(h: int, w: int, cin: int, w0: int = 16) -> float:
    """FLOPs of one forward pass (convolutions only) for an ``h x w`` input.

    Each 3x3 convolution costs ``2 * 9 * c_in * c_out * H * W``. BatchNorm,
    ReLU, pooling and upsampling are ignored (a few percent at most).
    ``h`` and ``w`` must be divisible by 16 (four 2x poolings).
    """
    if h % 16 or w % 16:
        raise ValueError("h and w must be divisible by 16")
    ch = _unet_channels(w0)

    def block(ci, co, hh, ww):
        return 2.0 * 9 * (ci * co + co * co) * hh * ww

    total = block(cin, ch[0], h, w)
    for i in range(4):
        total += block(ch[i], ch[i + 1], h // 2 ** (i + 1), w // 2 ** (i + 1))
    for i in range(4):
        total += block(ch[i + 1] + ch[i], ch[i], h // 2 ** i, w // 2 ** i)
    total += 2.0 * ch[0] * 1 * h * w                # final 1x1 convolution
    return total
