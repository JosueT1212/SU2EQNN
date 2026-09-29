"""Nested random training subsets for the training-size scaling study (teogpu02 paths).

Reads the sweep's aux file (same jets every variant sees) and writes, per size,
toptagging_<tag>_aux.npz (injection variants) and toptagging_<tag>.npz with
kinematics/labels only (baseline: with aux keys multi_channel adds axis channels). Train is a random subset, nested across sizes
(10k ⊂ 30k ⊂ ... from one permutation), sorted to keep file order. Val and
test stay full, so every point is scored on the same 404k test jets.
Grid 10k -> 1M, 3 seeds per point, 5 validations per run.
"""
import os

import numpy as np

SRC = "/var/tmp/arivero/toptagging_full_aux.npz"
DST = "/var/tmp/arivero/toptagging_{tag}{suffix}.npz"
SIZES = {"10k": 10_000, "30k": 30_000, "100k": 100_000, "300k": 300_000, "1M": 1_000_000}
SEED = 1  # same as the r7 60k subset

src = np.load(SRC)
n_train = len(src["labels_train"])
perm = np.random.default_rng(SEED).permutation(n_train)
held = {k: src[k] for k in src.files if not k.endswith("_train")}
train_keys = [k for k in src.files if k.endswith("_train")]
train = {k: src[k] for k in train_keys}

prev = None
for tag, n in SIZES.items():
    idx = np.sort(perm[:n])
    if prev is not None:
        assert np.isin(prev, idx).all(), tag  # nested
    prev = idx
    out = dict(held, **{k: v[idx] for k, v in train.items()})
    frac = out["labels_train"].mean()
    raw = {k: v for k, v in out.items() if k.startswith(("kinematics_", "labels_"))}
    for suffix, arrays in (("_aux", out), ("", raw)):
        dst = DST.format(tag=tag, suffix=suffix)
        np.savez(dst[:-4] + ".tmp.npz", **arrays)
        os.replace(dst[:-4] + ".tmp.npz", dst)  # atomic: prep trusts any file that exists
    print(tag, n, f"top fraction {frac:.3f}")
