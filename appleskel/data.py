"""Dataset scanning, validation, loading and caching.

File names follow ``R<row>N<number>_<Season>_<Kind>-<HH>-<MM>-<SS>.png`` with
Season in {Summer, Winter} and Kind in {RGB, D, GT}. A tree is identified by
its ``R<row>N<number>`` ID. Some winter files carry a *different* tree's ID
inside the name (e.g. ``R04N07_Winter_D-R04N22_RGB_...``); these are flagged
because the winter data for such trees cannot be trusted.

Error policy (see README, "Handling invalid input"):

* Problems that make processing unsafe raise :class:`DatasetError`
  (missing folder contents, duplicate files, a mask that is not binary,
  wrong image size, a depth image with no valid pixels).
* Recoverable problems (unrecognised file names, trees that fail to load) are
  reported to the caller: ``scan_dataset`` returns them as ``issues`` and
  ``build_arrays(..., on_error="skip")`` returns them as ``skipped``.

Conventions: images are 480x640 (height x width); masks are 256x256; depth is
an 8-bit grey value whose physical unit is NOT known (0 means "no
measurement").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SHAPE = (480, 640)          # (height, width) of RGB and depth images
MASK_SHAPE = (256, 256)
MODES = ("stretch", "centre")
MODE_KEYS = {"stretch": "Xs", "centre": "Xc"}   # keys used inside the cache file
FILENAME_RE = re.compile(r"^(R\d+N\d+)_(Summer|Winter)_(RGB|D|GT)-(.*)\.png$")
EMBEDDED_ID_RE = re.compile(r"^(R\d+N\d+)_RGB_")


class DatasetError(ValueError):
    """The dataset content cannot be used safely; processing must stop."""


@dataclass
class TreeRecord:
    """All files found for one tree.

    Attributes
    ----------
    tree_id : str
        e.g. ``"R05N34"``.
    files : dict
        Maps ``(season, kind)`` to the file path.
    id_mismatch : set
        ``(season, kind)`` keys whose file name embeds another tree's ID.
    """
    tree_id: str
    files: dict = field(default_factory=dict)
    id_mismatch: set = field(default_factory=set)


def parse_tree_id(tree_id: str) -> tuple[int, int]:
    """Split ``"R05N34"`` into ``(row, number) = (5, 34)``.

    Raises
    ------
    ValueError
        If ``tree_id`` does not look like ``R<digits>N<digits>``.
    """
    m = re.fullmatch(r"R(\d+)N(\d+)", tree_id)
    if not m:
        raise ValueError(f"Not a tree ID like 'R05N34': {tree_id!r}")
    return int(m.group(1)), int(m.group(2))


def parse_filename(name: str):
    """Parse a dataset file name.

    Returns
    -------
    dict or None
        Keys ``tree_id, season, kind, embedded_id`` (``embedded_id`` is the
        other tree ID found inside the name, or ``None``); ``None`` if the
        name does not follow the dataset convention.
    """
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    tree_id, season, kind, rest = m.groups()
    e = EMBEDDED_ID_RE.match(rest)
    return {"tree_id": tree_id, "season": season, "kind": kind,
            "embedded_id": e.group(1) if e else None}


def scan_dataset(root):
    """Index every recognised PNG under ``root`` by tree.

    Returns
    -------
    records : dict[str, TreeRecord]
    issues : list[str]
        Human-readable recoverable problems (unrecognised names, ID mismatches).

    Raises
    ------
    FileNotFoundError
        ``root`` does not exist.
    DatasetError
        ``root`` is not a folder, contains no recognised files, or contains
        two files for the same (tree, season, kind).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    if not root.is_dir():
        raise DatasetError(f"Not a folder: {root}")
    records, issues = {}, []
    for p in sorted(root.rglob("*.png")):
        info = parse_filename(p.name)
        if info is None:
            issues.append(f"skipped (unrecognised file name): {p.name}")
            continue
        rec = records.setdefault(info["tree_id"], TreeRecord(info["tree_id"]))
        key = (info["season"], info["kind"])
        if key in rec.files:
            raise DatasetError(f"duplicate {key} for {info['tree_id']}: "
                               f"{rec.files[key].name} and {p.name}")
        rec.files[key] = p
        if info["embedded_id"] and info["embedded_id"] != info["tree_id"]:
            rec.id_mismatch.add(key)
            issues.append(f"{p.name}: embedded ID {info['embedded_id']} differs from "
                          f"its own ID {info['tree_id']} (winter data unreliable)")
    if not records:
        raise DatasetError(f"No recognised dataset PNG files under {root}")
    return records, issues


def summer_complete_ids(records) -> list[str]:
    """Sorted IDs of trees that have a summer RGB, depth and ground-truth image."""
    need = (("Summer", "RGB"), ("Summer", "D"), ("Summer", "GT"))
    return sorted(t for t, r in records.items() if all(k in r.files for k in need))


def load_mask(path, expected_shape=MASK_SHAPE) -> np.ndarray:
    """Load a ground-truth mask as a boolean array.

    Raises
    ------
    DatasetError
        Wrong size, or more than two distinct pixel values (not binary).
    """
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != tuple(expected_shape):
        raise DatasetError(f"{Path(path).name}: mask shape {a.shape}, expected {tuple(expected_shape)}")
    if len(np.unique(a)) > 2:
        raise DatasetError(f"{Path(path).name}: mask is not binary")
    return a > 0


def load_rgb(path) -> np.ndarray:
    """Load an RGB image as ``uint8`` of shape (480, 640, 3)."""
    a = np.array(Image.open(path).convert("RGB"))
    if a.shape != IMAGE_SHAPE + (3,):
        raise DatasetError(f"{Path(path).name}: RGB shape {a.shape}, expected {IMAGE_SHAPE + (3,)}")
    return a


def load_depth(path) -> np.ndarray:
    """Load a depth image as 2-D ``uint8`` (first channel; 0 = no measurement).

    The depth files are stored as 3 identical grey channels; only the first is
    used. The physical unit of the grey values is unknown.

    Raises
    ------
    DatasetError
        Wrong size, or no valid (non-zero) pixel at all.
    """
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != IMAGE_SHAPE:
        raise DatasetError(f"{Path(path).name}: depth shape {a.shape}, expected {IMAGE_SHAPE}")
    if not a.any():
        raise DatasetError(f"{Path(path).name}: depth image has no valid (non-zero) pixels")
    return a.astype(np.uint8)


def prepare_input(rgb, depth, mode: str, size: int = 128) -> np.ndarray:
    """Resize RGB + depth to a ``(size, size, 4)`` ``uint8`` network input.

    Parameters
    ----------
    rgb : (H, W, 3) uint8
    depth : (H, W) uint8
    mode : {"stretch", "centre"}
        ``"stretch"`` squeezes the whole image into a square (aspect ratio
        changes). ``"centre"`` first crops the central H x H square. Which one
        matches the 256x256 masks is NOT confirmed by the dataset authors.
    size : int
        Output side length in pixels.

    Notes
    -----
    RGB uses bilinear resampling; depth uses nearest-neighbour so that
    "0 = missing" is never blended with valid values.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if size < 1:
        raise ValueError("size must be >= 1")
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.ndim != 3 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise DatasetError(f"rgb {rgb.shape} and depth {depth.shape} do not match")
    if mode == "centre":
        h, w = depth.shape
        left = (w - h) // 2                      # 80 for a 480x640 image
        rgb, depth = rgb[:, left:left + h], depth[:, left:left + h]
    rgb_r = np.array(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    d_r = np.array(Image.fromarray(depth).resize((size, size), Image.NEAREST))
    return np.dstack([rgb_r, d_r]).astype(np.uint8)


def build_arrays(records, tree_ids, modes=MODES, size: int = 128, on_error: str = "skip"):
    """Load, validate and resize all summer data for ``tree_ids``.

    Parameters
    ----------
    on_error : {"skip", "raise"}
        ``"skip"`` records a failing tree in ``skipped`` and continues;
        ``"raise"`` stops with :class:`DatasetError`.

    Returns
    -------
    dict with ``trees`` (kept IDs), ``X`` (mode -> (N, size, size, 4) uint8),
    ``Y`` ((N, 256, 256) uint8 in {0, 1}) and ``skipped`` ((id, reason) pairs).

    Raises
    ------
    DatasetError
        No tree could be loaded, or ``on_error == "raise"`` and one failed.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError("on_error must be 'skip' or 'raise'")
    X = {m: [] for m in modes}
    Y, kept, skipped = [], [], []
    for t in tree_ids:
        try:
            rec = records[t]
            for key in (("Summer", "RGB"), ("Summer", "D"), ("Summer", "GT")):
                if key not in rec.files:
                    raise DatasetError(f"missing file {key}")
            rgb = load_rgb(rec.files[("Summer", "RGB")])
            depth = load_depth(rec.files[("Summer", "D")])
            mask = load_mask(rec.files[("Summer", "GT")])
            prepared = {m: prepare_input(rgb, depth, m, size) for m in modes}
        except (DatasetError, KeyError, OSError) as e:     # OSError covers unreadable images
            if on_error == "raise":
                raise DatasetError(f"{t}: {e}") from e
            skipped.append((t, str(e)))
            continue
        for m in modes:
            X[m].append(prepared[m])
        Y.append(mask)
        kept.append(t)
    if not kept:
        raise DatasetError("no tree could be loaded")
    return {"trees": kept, "X": {m: np.stack(v) for m, v in X.items()},
            "Y": np.stack(Y).astype(np.uint8), "skipped": skipped}


def save_cache(path, trees, X, Y) -> None:
    """Write arrays to a compressed ``.npz`` (keys: trees, Xs, Xc, Y)."""
    arrays = {MODE_KEYS[m]: X[m] for m in X}
    np.savez_compressed(path, trees=np.array(trees), Y=Y, **arrays)


def load_cache(path) -> dict:
    """Read and validate a cache written by :func:`save_cache`.

    Returns
    -------
    dict with ``trees`` (list of str), ``X`` (mode -> array) and ``Y`` (uint8).

    Raises
    ------
    DatasetError
        Missing keys or inconsistent array shapes.
    """
    with np.load(path) as z:
        if "trees" not in z or "Y" not in z:
            raise DatasetError(f"{path}: cache lacks 'trees' or 'Y'")
        trees = [str(t) for t in z["trees"]]
        Y = z["Y"]
        X = {m: z[k] for m, k in MODE_KEYS.items() if k in z}
    if not X:
        raise DatasetError(f"{path}: cache has no input arrays")
    n = len(trees)
    if Y.shape != (n,) + MASK_SHAPE:
        raise DatasetError(f"{path}: Y has shape {Y.shape}, expected {(n,) + MASK_SHAPE}")
    for m, a in X.items():
        if a.ndim != 4 or a.shape[0] != n or a.shape[3] != 4 or a.shape[1] != a.shape[2]:
            raise DatasetError(f"{path}: X[{m}] has unexpected shape {a.shape}")
    return {"trees": trees, "X": X, "Y": Y}
