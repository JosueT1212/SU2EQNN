"""Precompute per-jet auxiliary features for a top-tagging npz dataset.

Replaces precompute_axes.py, precompute_masses.py, precompute_phases.py and
precompute_leading.py. Each feature is an independent writer; keys already in
the source npz are carried through unchanged, so re-running with a subset is
safe.

Usage:
    .venv/bin/python scripts/precompute_features.py \
        --features axes,masses data/toptagging_mini.npz data/toptagging_mini_aux.npz
"""

import argparse

import numpy as np

from su2eqnn.utils.catree import ca_hard_branch_phases
from su2eqnn.utils.subjets import (
    exclusive_kt_subjets, jet_axis, jet_mass, leading_momentum,
)

N_SUBJETS = 3
R = 0.8


def _compute_axes(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 1 + N_SUBJETS, 3) float32."""
    n = kinematics.shape[0]
    out = np.zeros((n, 1 + N_SUBJETS, 3), dtype=np.float32)
    for i in range(n):
        out[i, 0] = jet_axis(kinematics[i])
        out[i, 1:] = exclusive_kt_subjets(kinematics[i], n_subjets=N_SUBJETS, R=R)[0]
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def _compute_masses(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32, [m_jet, m_s1, m_s2, m_s3] in GeV."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i, 0] = jet_mass(kinematics[i])
        out[i, 1:] = exclusive_kt_subjets(kinematics[i], n_subjets=N_SUBJETS, R=R)[1]
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def _compute_phases(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32, sin(dpsi) on the C-A hard branch."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i] = ca_hard_branch_phases(kinematics[i])
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def _compute_leading(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32, highest-pT constituent 4-momentum."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i] = leading_momentum(kinematics[i])
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


# feature name -> (compute fn, npz key prefix)
FEATURES = {
    "axes":    (_compute_axes,    "axes"),
    "masses":  (_compute_masses,  "masses"),
    "phases":  (_compute_phases,  "phases"),
    "leading": (_compute_leading, "p_lead"),
}


def compute_feature(name: str, kinematics: np.ndarray) -> np.ndarray:
    """Dispatch to one feature's compute function."""
    if name not in FEATURES:
        raise KeyError(f"unknown feature {name!r}; known: {sorted(FEATURES)}")
    return FEATURES[name][0](kinematics)


def main(src: str, dst: str, features) -> None:
    raw = dict(np.load(src))
    modes = sorted(k.split("_", 1)[1] for k in raw if k.startswith("kinematics_"))
    for mode in modes:
        for name in features:
            print(f"mode={mode} feature={name}")
            key = f"{FEATURES[name][1]}_{mode}"
            raw[key] = compute_feature(name, raw[f"kinematics_{mode}"])
    np.savez(dst, **raw)
    print("wrote", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument(
        "--features", default="axes",
        help="comma-separated subset of " + ",".join(sorted(FEATURES)),
    )
    args = ap.parse_args()
    main(args.src, args.dst, [f.strip() for f in args.features.split(",") if f.strip()])
