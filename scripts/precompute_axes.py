"""Precompute jet + subjet axes for a top-tagging npz dataset.

Usage:
    python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
"""

import sys

import numpy as np

from su2eqnn.utils.subjets import exclusive_kt_axes, jet_axis

N_SUBJETS = 3
R = 0.8


def compute_axes(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) → (N, 1 + N_SUBJETS, 3) float32."""
    n = kinematics.shape[0]
    out = np.zeros((n, 1 + N_SUBJETS, 3), dtype=np.float32)
    for i in range(n):
        out[i, 0] = jet_axis(kinematics[i])
        out[i, 1:] = exclusive_kt_axes(kinematics[i], n_subjets=N_SUBJETS, R=R)
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def main(src: str, dst: str) -> None:
    raw = dict(np.load(src))
    modes = sorted(k.split("_", 1)[1] for k in raw if k.startswith("kinematics_"))
    for mode in modes:
        print(f"mode={mode}")
        raw[f"axes_{mode}"] = compute_axes(raw[f"kinematics_{mode}"])
    np.savez(dst, **raw)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
