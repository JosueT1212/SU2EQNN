import numpy as np
import pytest

from su2eqnn.utils.catree import ca_hard_branch_phases


def _mk(px, py, pz):
    p = np.array([0.0, px, py, pz])
    p[0] = np.linalg.norm(p[1:])
    return p


# Four particles with unambiguous C-A merge order (by eta-phi distance):
# A-B merge first (dR~0.01), then C joins AB (dR~0.125), then D (dR~0.22).
# Tree: root=(ABC, D) -> (AB, C) -> (A, B). Hard branch follows higher-pT prong.
A = _mk(100.0, 0.0, 0.0)
B = _mk(100.0, 1.0, 0.0)
C = _mk(80.0, 0.0, 10.0)
D = _mk(50.0, -10.0, -5.0)


def _plane(pa, pb):
    n = np.cross(pa[1:], pb[1:])
    return n / np.linalg.norm(n)


def _expected_phases():
    p_AB, p_ABC = A + B, A + B + C
    # splitting 1: (ABC | D), harder prong ABC
    n1 = _plane(p_ABC, D)
    # splitting 2: (AB | C), harder prong AB
    n2 = _plane(p_AB, C)
    # splitting 3: (A | B), harder prong B (pT 100.005 > 100)
    n3 = _plane(B, A)
    hd2 = p_AB[1:] / np.linalg.norm(p_AB[1:])
    hd3 = B[1:] / np.linalg.norm(B[1:])
    dpsi1 = np.arctan2(np.dot(np.cross(n1, n2), hd2), np.dot(n1, n2))
    dpsi2 = np.arctan2(np.dot(np.cross(n2, n3), hd3), np.dot(n2, n3))
    return np.array([np.sin(2 * dpsi1), np.sin(2 * dpsi2), 0.0, 0.0], dtype=np.float32)


def test_hand_built_tree_phases():
    kin = np.stack([A, B, C, D])
    out = ca_hard_branch_phases(kin)
    assert out.shape == (4,) and out.dtype == np.float32
    np.testing.assert_allclose(out, _expected_phases(), atol=1e-5)


def test_planar_jet_zero_phases():
    # All momenta in the xz plane -> every normal is +-y_hat -> sin(2*dpsi)=0.
    kin = np.stack([_mk(100, 0, 0), _mk(90, 0, 3), _mk(70, 0, -8), _mk(40, 0, 15)])
    np.testing.assert_allclose(ca_hard_branch_phases(kin), np.zeros(4), atol=1e-6)


def test_rotation_invariance():
    rng = np.random.default_rng(0)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    kin = np.stack([A, B, C, D])
    rot = kin.copy()
    rot[:, 1:] = kin[:, 1:] @ q.T
    np.testing.assert_allclose(
        ca_hard_branch_phases(rot), ca_hard_branch_phases(kin), atol=1e-5
    )


def test_parity_flips_sign():
    kin = np.stack([A, B, C, D])
    mirrored = kin.copy()
    mirrored[:, 1:] *= -1
    np.testing.assert_allclose(
        ca_hard_branch_phases(mirrored), -ca_hard_branch_phases(kin), atol=1e-5
    )


def test_degenerate_jets():
    # 2 particles: one splitting, no successive pair -> all zeros.
    kin = np.stack([A, D])
    np.testing.assert_allclose(ca_hard_branch_phases(kin), np.zeros(4))
    # 1 real particle + padding row -> all zeros, no crash.
    pad = np.zeros(4)
    np.testing.assert_allclose(ca_hard_branch_phases(np.stack([A, pad])), np.zeros(4))
    # collinear prongs -> zero phase, no NaN.
    kin = np.stack([_mk(100, 0, 0), _mk(50, 0, 0), _mk(70, 5, 5)])
    out = ca_hard_branch_phases(kin)
    assert np.all(np.isfinite(out))
