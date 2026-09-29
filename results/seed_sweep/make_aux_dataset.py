"""Build the aux dataset for the lab injection variants of the seed sweep (teogpu02 paths).

kinematics / labels / axes from toptagging_full_axes.npz, masses and p_lead
from the slim r5d / r8b files, minus the same 4 non-timelike jets dropped for
the rest-frame runs (results/rest_frame_full/dropped_non_timelike.json), so
every version of the sweep sees the same jets.
"""
import json
import numpy as np

AXES = "/home/arivero/SU2EQNN-r8/data/toptagging_full_axes.npz"
MASSES = "/home/arivero/SU2EQNN-r5d/data/toptagging_full_masses_slim.npz"
LEAD = "/home/arivero/SU2EQNN-r8/data/toptagging_full_leading_slim.npz"
DROPPED = "/home/arivero/SU2EQNN-main/results/rest_frame_full/dropped_non_timelike.json"
RAW = "/var/tmp/arivero/toptagging_full.npz"
DST = "/var/tmp/arivero/toptagging_full_aux.npz"

dropped = json.load(open(DROPPED))["dropped"]
a, ms, ld, raw = np.load(AXES), np.load(MASSES), np.load(LEAD), np.load(RAW)
out = {}
for mode in ("train", "val", "test"):
    n = len(a[f"labels_{mode}"])
    keep = np.ones(n, dtype=bool)
    keep[dropped[mode]["indices"]] = False
    for key, src in (("kinematics", a), ("labels", a), ("axes", a), ("masses", ms), ("p_lead", ld)):
        arr = src[f"{key}_{mode}"]
        assert len(arr) == n, (key, mode, len(arr), n)
        out[f"{key}_{mode}"] = arr[keep]
    # same jets, same order as the raw file used by the baseline and rest-frame runs
    assert np.array_equal(out[f"kinematics_{mode}"], raw[f"kinematics_{mode}"]), mode
    assert np.array_equal(out[f"labels_{mode}"], raw[f"labels_{mode}"]), mode
    print(mode, {k: v.shape for k, v in out.items() if k.endswith(mode)})
np.savez(DST, **out)
print("written", DST)
