"""Precompute C-A splitting-plane phases for a top-tagging npz dataset.

Usage:
    python scripts/precompute_phases.py data/toptagging_mini_axes.npz data/toptagging_mini_axes_phases.npz

Adds keys phases_{mode} (N, 4) float32 = sin(2*dpsi_k), k=1..4, carrying all
existing keys (kinematics, labels, axes) through unchanged.
"""

import sys

import numpy as np

from su2eqnn.utils.catree import ca_hard_branch_phases


def compute_phases(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i] = ca_hard_branch_phases(kinematics[i])
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def main(src: str, dst: str) -> None:
    raw = dict(np.load(src))
    modes = sorted(k.split("_", 1)[1] for k in raw if k.startswith("kinematics_"))
    for mode in modes:
        print(f"mode={mode}")
        raw[f"phases_{mode}"] = compute_phases(raw[f"kinematics_{mode}"])
    np.savez(dst, **raw)
    print("wrote", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: precompute_phases.py <in.npz> <out.npz>")
    main(sys.argv[1], sys.argv[2])
