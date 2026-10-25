"""Helpers shared by the experiment scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv

import numpy as np

from appleskel.data import load_cache


def load(cache):
    c = load_cache(cache)
    return list(c["trees"]), c["X"], c["Y"]


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)


def summarise(res):
    """Mean +/- std over trees for one run_* result dict."""
    return {k: f"{np.mean(res[k]):.3f}+/-{np.std(res[k]):.3f}"
            for k in ("test_dice", "test_iou") } | (
        {"test_boundary_f1": f"{np.mean(res['test_boundary_f1']):.3f}"} if res["test_boundary_f1"] is not None else {})
