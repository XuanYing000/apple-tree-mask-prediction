"""Scan the dataset folder, validate it and build the .npz cache (images resized to 128x128).

    python scripts/build_cache.py "/path/to/Dataset (With Summer GT)" data/apple_cache.npz
"""
import argparse

import _common  # noqa: F401
from appleskel.data import build_arrays, save_cache, scan_dataset, summer_complete_ids

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("dataset"); ap.add_argument("cache")
ap.add_argument("--on-error", choices=["skip", "raise"], default="skip")
a = ap.parse_args()
records, issues = scan_dataset(a.dataset)
ids = summer_complete_ids(records)
print(f"{len(records)} trees found, {len(ids)} with summer RGB + depth + ground truth, {len(issues)} naming issues")
for s in issues[:5]:
    print("  issue:", s)
out = build_arrays(records, ids, on_error=a.on_error)
print(f"loaded {len(out['trees'])} trees, skipped {len(out['skipped'])}")
for t, why in out["skipped"]:
    print("  skipped", t, why)
save_cache(a.cache, out["trees"], out["X"], out["Y"])
print("saved", a.cache)
