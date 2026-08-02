"""Precompute jet + subjet invariant masses for a top-tagging npz dataset.

Usage:
    python scripts/precompute_masses.py data/toptagging_mini_axes.npz data/toptagging_mini_axes_masses.npz

Adds keys masses_{mode} (N, 4) float32 = [m_jet, m_s1, m_s2, m_s3] in GeV
(unrescaled — the embedding divides by UNITS), carrying all existing keys.
"""

import sys

import numpy as np

from su2eqnn.utils.subjets import exclusive_kt_subjets, jet_mass

N_SUBJETS = 3
R = 0.8


def compute_masses(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i, 0] = jet_mass(kinematics[i])
        out[i, 1:] = exclusive_kt_subjets(kinematics[i], n_subjets=N_SUBJETS, R=R)[1]
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def main(src: str, dst: str) -> None:
    raw = dict(np.load(src))
    modes = sorted(k.split("_", 1)[1] for k in raw if k.startswith("kinematics_"))
    for mode in modes:
        print(f"mode={mode}")
        raw[f"masses_{mode}"] = compute_masses(raw[f"kinematics_{mode}"])
    np.savez(dst, **raw)
    print("wrote", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: precompute_masses.py <in.npz> <out.npz>")
    main(sys.argv[1], sys.argv[2])
