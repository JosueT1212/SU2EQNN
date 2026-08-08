"""Precompute leading-constituent momentum for a top-tagging npz dataset.

Usage:
    python scripts/precompute_leading.py data/toptagging_mini_axes.npz data/toptagging_mini_axes_leading.npz

Adds keys p_lead_{mode} (N, 4) float32 = [E, px, py, pz] of the highest-pT
constituent, carrying all existing keys through unchanged.
"""

import sys

import numpy as np

from su2eqnn.utils.subjets import leading_momentum


def compute_leading(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i] = leading_momentum(kinematics[i])
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def main(src: str, dst: str) -> None:
    raw = dict(np.load(src))
    modes = sorted(k.split("_", 1)[1] for k in raw if k.startswith("kinematics_"))
    for mode in modes:
        print(f"mode={mode}")
        raw[f"p_lead_{mode}"] = compute_leading(raw[f"kinematics_{mode}"])
    np.savez(dst, **raw)
    print("wrote", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: precompute_leading.py <in.npz> <out.npz>")
    main(sys.argv[1], sys.argv[2])
