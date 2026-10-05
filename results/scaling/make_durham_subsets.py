"""Little-group (rest-frame Durham) training subsets for the scaling study (teogpu02 paths).

Same nested random permutation as make_subsets.py (seed 1 over the same train jets),
so every little-group point trains on exactly the jets its naive counterpart saw.
Writes toptagging_<tag>_durham.npz: kinematics, labels, subjet_idx. Val and test stay full.
"""
import os

import numpy as np

SRC = "/var/tmp/arivero/toptagging_full_durham.npz"
NAIVE = "/var/tmp/arivero/toptagging_{tag}.npz"   # make_subsets.py output, same jets
DST = "/var/tmp/arivero/toptagging_{tag}_durham.npz"
SIZES = {"30k": 30_000, "100k": 100_000, "300k": 300_000, "1M": 1_000_000}
SEED = 1  # make_subsets.py

src = np.load(SRC)
n_train = len(src["labels_train"])
perm = np.random.default_rng(SEED).permutation(n_train)
held = {k: src[k] for k in src.files if not k.endswith("_train")}
train = {k: src[k] for k in src.files if k.endswith("_train")}

for tag, n in SIZES.items():
    idx = np.sort(perm[:n])
    out = dict(held, **{k: v[idx] for k, v in train.items()})
    naive = np.load(NAIVE.format(tag=tag))
    assert np.array_equal(naive["labels_train"], out["labels_train"]), tag
    assert np.array_equal(naive["kinematics_train"][:1000], out["kinematics_train"][:1000]), tag
    dst = DST.format(tag=tag)
    np.savez(dst[:-4] + ".tmp.npz", **out)
    os.replace(dst[:-4] + ".tmp.npz", dst)  # atomic: prep trusts any file that exists
    print(tag, n, f"top fraction {out['labels_train'].mean():.3f}")
