import numpy as np

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
    return np.array([np.sin(dpsi1), np.sin(dpsi2), 0.0, 0.0], dtype=np.float32)


def test_hand_built_tree_phases():
    kin = np.stack([A, B, C, D])
    out = ca_hard_branch_phases(kin)
    assert out.shape == (4,) and out.dtype == np.float32
    np.testing.assert_allclose(out, _expected_phases(), atol=1e-5)


def test_planar_jet_zero_phases():
    # All momenta in the xz plane -> dpsi = 0 or pi -> sin(dpsi)=0.
    kin = np.stack([_mk(100, 0, 0), _mk(90, 0, 3), _mk(70, 0, -8), _mk(40, 0, 15)])
    np.testing.assert_allclose(ca_hard_branch_phases(kin), np.zeros(4), atol=1e-6)


def test_azimuthal_rotation_invariance():
    # Rotate all momenta by a random angle about the z axis (azimuthal rotation).
    # The tree topology and phases should remain unchanged.
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    kin = np.stack([A, B, C, D])
    rot = kin.copy()
    rot[:, 1:] = kin[:, 1:] @ rot_matrix.T
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


def test_dataset_attaches_phases(tmp_path):
    import torch
    from experiments.top_tagging.dataset import TopTaggingDataset

    n_jets, n_part = 3, 5
    kin = np.zeros((n_jets, n_part, 4), dtype=np.float32)
    kin[:, :, 0] = 1.0  # E above EPS
    kin[:, :, 1] = 10.0
    labels = np.zeros(n_jets, dtype=np.int64)
    base = {}
    for mode in ("train", "val", "test"):
        base[f"kinematics_{mode}"] = kin
        base[f"labels_{mode}"] = labels
    for m in ("train", "val", "test"):
        base[f"phases_{m}"] = np.full((n_jets, 4), 0.5, dtype=np.float32)
    np.savez(tmp_path / "toy.npz", **base)

    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(tmp_path / "toy.npz"), "train")
    assert hasattr(ds[0], "phases")
    assert ds[0].phases.shape == (1, 4)
    assert torch.allclose(ds[0].phases, torch.full((1, 4), 0.5))
