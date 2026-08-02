"""Cambridge-Aachen tree utilities: splitting-plane phases along the hard branch.

Physics: PanScales construction (arXiv:2103.16526 Sec 3.2). Each C-A declustering
defines a splitting plane via the cross product of its two prongs; the SIGNED
dihedral angle between successive planes carries the cos(2*dpsi) gluon-helicity
interference modulation. sin(2*dpsi) is parity-odd and rotation-invariant ->
legitimate pseudoscalar (grade-3) input.
"""

import numpy as np

EPS = 1e-5  # padding-mask threshold, same convention as subjets.py
_CROSS_FLOOR = 1e-8  # collinear-prong guard on |p_a x p_b|


def _eta_phi(p3):
    px, py, pz = p3
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / pt) if pt > 0 else 0.0
    phi = np.arctan2(py, px)
    return eta, phi


def _delta_r2(pa3, pb3):
    eta_a, phi_a = _eta_phi(pa3)
    eta_b, phi_b = _eta_phi(pb3)
    dphi = abs(phi_a - phi_b)
    dphi = min(dphi, 2.0 * np.pi - dphi)
    return (eta_a - eta_b) ** 2 + dphi ** 2


def ca_hard_branch_phases(kinematics, n_phases=4, R=0.8):
    """sin(2*dpsi) between successive C-A splitting planes on the hard branch.

    Parameters
    ----------
    kinematics : (n, 4) array, rows (E, px, py, pz); rows with E <= EPS ignored.
    n_phases : number of phases returned (needs n_phases+1 splittings).
    R : jet radius for the C-A metric d_ij = dR^2 / R^2.

    Returns
    -------
    (n_phases,) float32; entries beyond the available tree depth, or involving
    a collinear (|cross| < 1e-8) splitting, are 0.0.
    """
    kinematics = np.asarray(kinematics, dtype=np.float64)
    real = kinematics[kinematics[:, 0] > EPS]
    phases = np.zeros(n_phases, dtype=np.float32)
    if len(real) < 2:
        return phases

    # nodes[i] = (four-momentum, children index pair or None)
    nodes = [(p.copy(), None) for p in real]
    active = list(range(len(nodes)))
    while len(active) > 1:
        best = None
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                d = _delta_r2(nodes[active[ii]][0][1:], nodes[active[jj]][0][1:]) / R**2
                if best is None or d < best[0]:
                    best = (d, ii, jj)
        _, ii, jj = best
        i, j = active[ii], active[jj]
        nodes.append((nodes[i][0] + nodes[j][0], (i, j)))
        active = [a for k, a in enumerate(active) if k not in (ii, jj)]
        active.append(len(nodes) - 1)

    # Walk the hard branch: at each node follow the higher-pT prong.
    normals, hard_dirs = [], []
    node = nodes[active[0]]
    while node[1] is not None and len(normals) < n_phases + 1:
        a, b = nodes[node[1][0]], nodes[node[1][1]]
        if np.hypot(b[0][1], b[0][2]) > np.hypot(a[0][1], a[0][2]):
            a, b = b, a
        cross = np.cross(a[0][1:], b[0][1:])
        norm = np.linalg.norm(cross)
        normals.append(cross / norm if norm > _CROSS_FLOOR else None)
        pa = a[0][1:]
        hard_dirs.append(pa / np.linalg.norm(pa))
        node = a

    for k in range(min(len(normals) - 1, n_phases)):
        n1, n2 = normals[k], normals[k + 1]
        if n1 is None or n2 is None:
            continue
        dpsi = np.arctan2(np.dot(np.cross(n1, n2), hard_dirs[k + 1]), np.dot(n1, n2))
        phases[k] = np.sin(2.0 * dpsi)
    return phases
