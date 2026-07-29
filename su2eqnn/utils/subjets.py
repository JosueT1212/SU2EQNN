"""Exclusive-kT subjet clustering and jet-axis utilities (numpy, no fastjet).

Used offline by scripts/precompute_axes.py — not on the training hot path.
"""

import numpy as np

EPS = 1e-5   # padding threshold, matches experiments/top_tagging/dataset.py


def jet_axis(kinematics: np.ndarray) -> np.ndarray:
    """Unit vector along the summed 3-momentum of all real constituents.

    Parameters
    ----------
    kinematics : (n_pad, 4) float array — E, px, py, pz rows, zero-padded.

    Returns
    -------
    (3,) unit vector. If the jet has no real constituents, returns +z.
    """
    mask = np.abs(kinematics).max(axis=-1) > EPS
    p = kinematics[mask][:, 1:4].sum(axis=0)
    norm = np.linalg.norm(p)
    if norm < EPS:
        return np.array([0.0, 0.0, 1.0])
    return p / norm


def _eta_phi_pt(p3: np.ndarray):
    """p3: (n, 3) momenta → (eta, phi, pt) arrays."""
    pt = np.hypot(p3[:, 0], p3[:, 1])
    pt = np.maximum(pt, 1e-12)
    eta = np.arcsinh(p3[:, 2] / pt)
    phi = np.arctan2(p3[:, 1], p3[:, 0])
    return eta, phi, pt


def exclusive_kt_axes(
    kinematics: np.ndarray, n_subjets: int = 3, R: float = 0.8
) -> np.ndarray:
    """Cluster constituents to exactly `n_subjets` with exclusive kT; return axes.

    kT distance: d_ij = min(pT_i^2, pT_j^2) * dR_ij^2 / R^2, dR^2 = dEta^2 + dPhi^2.
    Pairs with minimal d_ij are merged (4-momentum sum) until n_subjets remain.

    Returns
    -------
    (n_subjets, 3) unit axes, sorted by subjet pT descending. If the jet has
    fewer real constituents than n_subjets, missing rows are filled with the
    jet axis.

    Note: the eta-phi kT metric is beam-frame dependent — resulting axes are
    equivariant only under azimuthal rotations about the beam axis, matching
    the symmetry breaking of the beam spurions.
    """
    nj = jet_axis(kinematics)
    mask = np.abs(kinematics).max(axis=-1) > EPS
    p = kinematics[mask].astype(np.float64).copy()   # (n, 4)

    while len(p) > n_subjets:
        eta, phi, pt = _eta_phi_pt(p[:, 1:4])
        dphi = np.abs(phi[:, None] - phi[None, :])
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        dr2 = (eta[:, None] - eta[None, :]) ** 2 + dphi ** 2
        dij = np.minimum(pt[:, None] ** 2, pt[None, :] ** 2) * dr2 / R ** 2
        np.fill_diagonal(dij, np.inf)
        i, j = np.unravel_index(np.argmin(dij), dij.shape)
        p[i] += p[j]
        p = np.delete(p, j, axis=0)

    # pT-descending order
    _, _, pt = _eta_phi_pt(p[:, 1:4])
    order = np.argsort(-pt)
    p = p[order]

    axes = np.tile(nj, (n_subjets, 1))
    for k in range(min(len(p), n_subjets)):
        norm = np.linalg.norm(p[k, 1:4])
        if norm > EPS:
            axes[k] = p[k, 1:4] / norm
    return axes
