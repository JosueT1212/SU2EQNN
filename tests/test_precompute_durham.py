"""precompute_features.py --features durham writes per-constituent Durham labels."""

import importlib.util
from pathlib import Path

import numpy as np

from su2eqnn.utils.subjets import durham_exclusive_membership

SCRIPT = Path(__file__).parent.parent / "scripts" / "precompute_features.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("precompute_features", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jets(n, seed):
    rng = np.random.default_rng(seed)
    kin = np.zeros((n, 200, 4), dtype=np.float32)
    for i in range(n):
        k = 6 + i
        p = rng.normal(size=(k, 3)) * 20 + np.array([0, 0, 300.0])
        kin[i, :k, 1:] = p
        kin[i, :k, 0] = np.linalg.norm(p, axis=1)
    return kin


def test_durham_feature_round_trip(tmp_path):
    mod = _load_script()
    src, dst = tmp_path / "src.npz", tmp_path / "dst.npz"
    data = {}
    for j, mode in enumerate(("train", "val", "test")):
        data[f"kinematics_{mode}"] = _jets(3, j)
        data[f"labels_{mode}"] = np.array([True, False, True])
    np.savez(src, **data)
    mod.main(str(src), str(dst), ["durham"])
    out = np.load(dst)
    for mode in ("train", "val", "test"):
        idx = out[f"subjet_idx_{mode}"]
        assert idx.dtype == np.int8 and idx.shape == (3, 200)
        for i in range(3):
            np.testing.assert_array_equal(
                idx[i], durham_exclusive_membership(data[f"kinematics_{mode}"][i]))
        np.testing.assert_array_equal(out[f"kinematics_{mode}"], data[f"kinematics_{mode}"])
