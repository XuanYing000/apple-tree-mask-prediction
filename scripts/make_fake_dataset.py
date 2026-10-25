"""Write a small SYNTHETIC dataset (real naming scheme) for trying the pipeline.

    python scripts/make_fake_dataset.py data/fake
"""
import argparse

import _common  # noqa: F401
from appleskel.synthetic import make_fake_dataset

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("out"); ap.add_argument("--rows", type=int, default=10); ap.add_argument("--per-row", type=int, default=8)
a = ap.parse_args()
ids = make_fake_dataset(a.out, rows=range(1, a.rows + 1), per_row=a.per_row)
print(f"wrote {len(ids)} fake trees to {a.out}")
