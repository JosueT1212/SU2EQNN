# Authors: J. Tapia-Hernandez and A. Ramirez-Morales
# Exclusive-kT subjet clustering and jet-axis utilities
# Used by scripts/precompute_features.py

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


def leading_momentum(kinematics: np.ndarray) -> np.ndarray:
    """4-momentum row of the highest-pT real (non-zero-padded) constituent.

    Parameters
    ----------
    kinematics : (n_pad, 4) float array — E, px, py, pz rows, zero-padded.

    Returns
    -------
    (4,) row [E, px, py, pz]. All-zero if the jet has no real constituents.
    """
    mask = np.abs(kinematics).max(axis=-1) > EPS
    real = kinematics[mask]
    if len(real) == 0:
        return np.zeros(4)
    pt = np.hypot(real[:, 1], real[:, 2])
    return real[np.argmax(pt)]


def _eta_phi_pt(p3: np.ndarray):
    """p3: (n, 3) momenta → (eta, phi, pt) arrays."""
    pt = np.hypot(p3[:, 0], p3[:, 1])
    pt = np.maximum(pt, 1e-12)
    eta = np.arcsinh(p3[:, 2] / pt)
    phi = np.arctan2(p3[:, 1], p3[:, 0])
    return eta, phi, pt


def _inv_mass(p4: np.ndarray) -> float:
    """Invariant mass of a four-momentum (E, px, py, pz), clamped at 0."""
    return float(np.sqrt(max(p4[0] ** 2 - np.dot(p4[1:], p4[1:]), 0.0)))


def jet_mass(kinematics: np.ndarray) -> float:
    """Invariant mass of the sum of all real constituents (padding ignored)."""
    mask = np.abs(kinematics).max(axis=-1) > EPS
    if not mask.any():
        return 0.0
    return _inv_mass(kinematics[mask].astype(np.float64).sum(axis=0))


def exclusive_kt_subjets(
    kinematics: np.ndarray, n_subjets: int = 3, R: float = 0.8
):
    """Cluster constituents to exactly `n_subjets` with exclusive kT; return axes AND masses.

    kT distance: d_ij = min(pT_i^2, pT_j^2) * dR_ij^2 / R^2, dR^2 = dEta^2 + dPhi^2.
    Pairs with minimal d_ij are merged (4-momentum sum) until n_subjets remain.

    Also returns m_k = sqrt(max(E^2 - |p|^2, 0)) of each subjet's summed
    four-momentum.

    Returns
    -------
    (n_subjets, 3) unit axes, (n_subjets,) masses — both sorted by subjet pT
    descending. If the jet has fewer real constituents than n_subjets,
    missing rows fall back to the jet axis (axes) and 0.0 (masses).

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
    masses = np.zeros(n_subjets)
    for k in range(min(len(p), n_subjets)):
        norm = np.linalg.norm(p[k, 1:4])
        if norm > EPS:
            axes[k] = p[k, 1:4] / norm
        masses[k] = _inv_mass(p[k])
    return axes, masses


def exclusive_kt_axes(
    kinematics: np.ndarray, n_subjets: int = 3, R: float = 0.8
) -> np.ndarray:
    """Backwards-compatible wrapper: axes only. See exclusive_kt_subjets."""
    return exclusive_kt_subjets(kinematics, n_subjets=n_subjets, R=R)[0]


def _boost_to_rest_np(v: np.ndarray, P: np.ndarray) -> np.ndarray:
    """numpy float64 port of experiments/top_tagging/embedding/core.boost_to_rest, one P.

    v : (n, 4) four-vectors, P : (4,) timelike. Returns L(P)^-1 v, (n, 4).
    """
    m = np.sqrt(P[0] ** 2 - P[1:] @ P[1:])
    pdotv = v[:, 1:] @ P[1:]
    out = np.empty_like(v)
    out[:, 0] = (P[0] * v[:, 0] - pdotv) / m
    out[:, 1:] = v[:, 1:] + P[1:][None, :] * (pdotv / (m * (P[0] + m)) - v[:, 0] / m)[:, None]
    return out


def durham_exclusive_membership(kinematics: np.ndarray, n_subjets: int = 3) -> np.ndarray:
    """Durham (e+e- kT) exclusive clustering in the jet rest frame; one label per row.

    y_ij = 2 min(E_i^2, E_j^2) (1 - cos theta_ij) / m_J^2 on rest-frame momenta,
    E-scheme recombination, until `n_subjets` remain. Energies and opening angles
    only, so membership is invariant under any rotation of the rest frame, in
    particular the Wigner rotation of a lab boost. The 1/m_J^2 is a per-jet constant
    and does not change the argmin, so it is skipped.

    Parameters
    ----------
    kinematics : (n_pad, 4) E, px, py, pz rows (lab frame), zero-padded.

    Returns
    -------
    (n_pad,) int8: subjet label 0..n_subjets-1 by descending rest-frame energy
    (a Lorentz invariant, P.p_s / m_J), -1 on padding rows. With fewer real rows
    than n_subjets the missing labels simply never occur.
    """
    mask = np.abs(kinematics).max(axis=-1) > EPS
    labels = np.full(len(kinematics), -1, dtype=np.int8)
    real = np.nonzero(mask)[0]
    if len(real) == 0:
        return labels
    p = kinematics[mask].astype(np.float64)
    P = p.sum(axis=0)
    if P[0] ** 2 - P[1:] @ P[1:] <= 0:
        raise ValueError("non-timelike jet: rest frame undefined")
    c = _boost_to_rest_np(p, P)
    members = [[i] for i in range(len(c))]
    while len(c) > n_subjets:
        u = c[:, 1:] / np.maximum(np.linalg.norm(c[:, 1:], axis=1, keepdims=True), 1e-300)
        e2 = c[:, 0] ** 2
        y = 2 * np.minimum(e2[:, None], e2[None, :]) * (1 - u @ u.T)
        np.fill_diagonal(y, np.inf)
        i, j = sorted(np.unravel_index(np.argmin(y), y.shape))
        c[i] += c[j]
        c = np.delete(c, j, axis=0)
        members[i] += members.pop(j)
    for label, k in enumerate(np.argsort(-c[:, 0], kind="stable")):
        labels[real[members[k]]] = label
    return labels
