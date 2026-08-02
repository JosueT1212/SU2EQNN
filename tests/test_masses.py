import numpy as np

from su2eqnn.utils.subjets import exclusive_kt_axes, exclusive_kt_subjets, jet_mass


def _mk(px, py, pz):
    p = np.array([0.0, px, py, pz])
    p[0] = np.linalg.norm(p[1:])
    return p


# Same 4-particle fixture family as the R5 series tests: A-B are nearly
# collinear (merge first under kT), C and D are far in eta-phi.
A = _mk(100.0, 0.0, 0.0)
B = _mk(100.0, 1.0, 0.0)
C = _mk(80.0, 0.0, 10.0)
D = _mk(50.0, -10.0, -5.0)


def _pair_mass(p1, p2):
    # closed form for massless constituents: m^2 = 2 E1 E2 (1 - cos theta)
    cos = np.dot(p1[1:], p2[1:]) / (np.linalg.norm(p1[1:]) * np.linalg.norm(p2[1:]))
    return np.sqrt(2.0 * p1[0] * p2[0] * (1.0 - cos))


def test_subjet_masses_hand_values():
    kin = np.stack([A, B, C, D])
    axes, masses = exclusive_kt_subjets(kin, n_subjets=3, R=0.8)
    assert masses.shape == (3,)
    # kT merges A+B (tiny dR); subjets pT-descending: [AB, C, D]
    np.testing.assert_allclose(masses[0], _pair_mass(A, B), rtol=1e-6)
    np.testing.assert_allclose(masses[1], 0.0, atol=1e-6)  # single massless
    np.testing.assert_allclose(masses[2], 0.0, atol=1e-6)


def test_jet_mass_hand_value():
    kin = np.stack([A, B, C, D])
    tot = A + B + C + D
    expected = np.sqrt(tot[0] ** 2 - np.dot(tot[1:], tot[1:]))
    np.testing.assert_allclose(jet_mass(kin), expected, rtol=1e-6)
    # padding rows ignored
    kin_pad = np.vstack([kin, np.zeros((2, 4))])
    np.testing.assert_allclose(jet_mass(kin_pad), expected, rtol=1e-6)


def test_mass_clamp_no_nan():
    # single particle with E slightly BELOW |p| (float rounding) -> m=0, not NaN
    p = np.array([[100.0 - 1e-9, 100.0, 0.0, 0.0]])
    _, masses = exclusive_kt_subjets(p, n_subjets=3, R=0.8)
    assert np.all(np.isfinite(masses)) and masses[0] == 0.0


def test_missing_subjets_mass_zero():
    # 2 real particles, n_subjets=3 -> third subjet missing -> mass 0
    kin = np.stack([A, D])
    _, masses = exclusive_kt_subjets(kin, n_subjets=3, R=0.8)
    assert masses.shape == (3,)
    assert masses[2] == 0.0


def test_axes_wrapper_unchanged():
    rng = np.random.default_rng(1)
    for _ in range(5):
        kin = np.zeros((10, 4))
        kin[:, 1:] = rng.normal(scale=50.0, size=(10, 3))
        kin[:, 0] = np.linalg.norm(kin[:, 1:], axis=1)
        axes_wrapper = exclusive_kt_axes(kin, n_subjets=3, R=0.8)
        axes_direct, _ = exclusive_kt_subjets(kin, n_subjets=3, R=0.8)
        np.testing.assert_array_equal(axes_wrapper, axes_direct)
