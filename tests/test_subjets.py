"""Tests for exclusive-kT subjet axis computation."""

import numpy as np

from su2eqnn.utils.subjets import exclusive_kt_axes, jet_axis

RNG = np.random.default_rng(0)


def _make_particle(direction, energy):
    """4-momentum (E, p) of a massless particle along `direction`."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return np.concatenate([[energy], energy * d])


def _make_jet(clusters, n_pad=200):
    """clusters: list of (direction, energy, n_particles). Returns (n_pad, 4)."""
    rows = []
    for direction, energy, n in clusters:
        for _ in range(n):
            smeared = np.asarray(direction) + 0.02 * RNG.normal(size=3)
            rows.append(_make_particle(smeared, energy / n))
    kin = np.zeros((n_pad, 4))
    kin[: len(rows)] = rows
    return kin


def test_jet_axis_unit_and_direction():
    kin = _make_jet([((0.1, 0.0, 1.0), 100.0, 10)])
    ax = jet_axis(kin)
    assert np.isclose(np.linalg.norm(ax), 1.0, atol=1e-6)
    expected = np.array([0.1, 0.0, 1.0]) / np.linalg.norm([0.1, 0.0, 1.0])
    assert ax @ expected > 0.99


def test_three_cluster_recovery():
    """Three well-separated prongs → axes align with prong directions."""
    dirs = [(1.0, 0.0, 3.0), (-1.0, 0.5, 3.0), (0.2, -1.0, 3.0)]
    kin = _make_jet([(d, 100.0, 15) for d in dirs])
    axes = exclusive_kt_axes(kin, n_subjets=3, R=0.8)
    assert axes.shape == (3, 3)
    unit_dirs = [np.asarray(d) / np.linalg.norm(d) for d in dirs]
    for ud in unit_dirs:
        best = max(axes @ ud)
        assert best > 0.98, f"prong {ud} not recovered, best dot {best:.3f}"


def test_pt_ordering():
    """Axes sorted by subjet pT descending."""
    kin = _make_jet([
        ((1.0, 0.0, 2.0), 300.0, 15),   # hard
        ((-1.0, 0.0, 2.0), 100.0, 15),  # medium
        ((0.0, 1.0, 2.0), 30.0, 15),    # soft
    ])
    axes = exclusive_kt_axes(kin)
    hard = np.array([1.0, 0.0, 2.0]); hard /= np.linalg.norm(hard)
    assert axes[0] @ hard > 0.98, "hardest subjet must be first"


def test_single_prong_no_crash():
    kin = _make_jet([((0.0, 0.0, 1.0), 200.0, 40)])
    axes = exclusive_kt_axes(kin)
    assert axes.shape == (3, 3)
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-6)


def test_few_constituents_fallback():
    """2 real particles → third axis falls back to n̂_jet."""
    kin = np.zeros((200, 4))
    kin[0] = _make_particle((1.0, 0.0, 1.0), 50.0)
    kin[1] = _make_particle((-1.0, 0.0, 1.0), 50.0)
    axes = exclusive_kt_axes(kin)
    assert axes.shape == (3, 3)
    nj = jet_axis(kin)
    assert np.allclose(axes[2], nj, atol=1e-6), "missing subjet must fall back to jet axis"


def test_leading_momentum_picks_highest_pt():
    from su2eqnn.utils.subjets import leading_momentum

    kin = np.zeros((200, 4))
    kin[0] = [10.0, 1.0, 0.0, 0.0]    # pt=1
    kin[1] = [50.0, 10.0, 0.0, 3.0]   # pt=10, leading
    kin[2] = [20.0, 2.0, 2.0, 1.0]    # pt=2.83
    lead = leading_momentum(kin)
    assert np.allclose(lead, [50.0, 10.0, 0.0, 3.0])


def test_leading_momentum_ignores_padding():
    from su2eqnn.utils.subjets import leading_momentum

    kin = np.zeros((200, 4))
    kin[0] = [10.0, 1.0, 0.0, 0.0]
    lead = leading_momentum(kin)
    assert np.allclose(lead, [10.0, 1.0, 0.0, 0.0])


def test_leading_momentum_empty_jet():
    from su2eqnn.utils.subjets import leading_momentum

    kin = np.zeros((200, 4))
    lead = leading_momentum(kin)
    assert np.allclose(lead, [0.0, 0.0, 0.0, 0.0])


# --- Durham (e+e- kT) exclusive-3 in the jet rest frame ----------------------

import pytest

from su2eqnn.utils.subjets import _boost_to_rest_np, durham_exclusive_membership


def _rest_rows(kin):
    mask = np.abs(kin).max(axis=-1) > 1e-5
    p = kin[mask].astype(np.float64)
    return _boost_to_rest_np(p, p.sum(axis=0)), mask


def _random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _three_prong():
    return _make_jet([((0.15, 0.0, 1.0), 120.0, 8),
                      ((-0.1, 0.12, 1.0), 90.0, 6),
                      ((-0.05, -0.15, 1.0), 60.0, 5)])


def test_durham_dtype_shape_padding():
    kin = _three_prong()
    lab = durham_exclusive_membership(kin)
    assert lab.dtype == np.int8 and lab.shape == (200,)
    assert (lab[19:] == -1).all() and set(lab[:19].tolist()) == {0, 1, 2}


def test_durham_recovers_prongs():
    lab = durham_exclusive_membership(_three_prong())
    for start, stop in ((0, 8), (8, 14), (14, 19)):
        assert len(set(lab[start:stop].tolist())) == 1


def test_durham_labels_ordered_by_rest_energy():
    kin = _three_prong()
    lab = durham_exclusive_membership(kin)
    rest, mask = _rest_rows(kin)
    e = [rest[lab[mask] == k, 0].sum() for k in range(3)]
    assert e[0] >= e[1] >= e[2]


def test_durham_subjets_sum_to_zero_in_rest_frame():
    kin = _three_prong()
    lab = durham_exclusive_membership(kin)
    rest, mask = _rest_rows(kin)
    s = np.stack([rest[lab[mask] == k].sum(axis=0) for k in range(3)])
    m = rest[:, 0].sum()
    assert np.abs(s[:, 1:].sum(axis=0)).max() <= 1e-9 * m


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_durham_rotation_invariant(seed):
    kin = _three_prong()
    lab = durham_exclusive_membership(kin)
    rest, mask = _rest_rows(kin)
    R = _random_rotation(seed)
    rot = np.zeros_like(kin, dtype=np.float64)
    rot[: mask.sum(), 0] = rest[:, 0]
    rot[: mask.sum(), 1:] = rest[:, 1:] @ R.T
    np.testing.assert_array_equal(durham_exclusive_membership(rot)[: mask.sum()], lab[mask])


def test_durham_hand_toy_merges_soft_collinear_pair():
    # rest-frame massless momenta summing to zero; b is soft and 0.1 rad from a
    a = 10 * np.array([1.0, 0.0, 0.0])
    b = 1.0 * np.array([np.cos(0.1), np.sin(0.1), 0.0])
    c = 8 * np.array([-0.5, np.sqrt(3) / 2, 0.0])
    d = -(a + b + c)
    kin = np.zeros((200, 4))
    for i, p in enumerate((a, b, c, d)):
        kin[i] = [np.linalg.norm(p), *p]
    lab = durham_exclusive_membership(kin)
    assert lab[0] == lab[1] and len({lab[0], lab[2], lab[3]}) == 3


def test_durham_fewer_than_three_constituents():
    kin = np.zeros((200, 4))
    kin[0] = _make_particle((0.1, 0.0, 1.0), 50.0)
    kin[1] = _make_particle((-0.1, 0.05, 1.0), 40.0)
    lab = durham_exclusive_membership(kin)
    assert sorted(lab[:2].tolist()) == [0, 1] and (lab[2:] == -1).all()


def test_durham_empty_jet():
    assert (durham_exclusive_membership(np.zeros((200, 4))) == -1).all()


def test_boost_to_rest_np_matches_torch_reference():
    import torch
    from experiments.top_tagging.embedding.core import boost_to_rest
    rng = np.random.default_rng(7)
    v = rng.normal(size=(30, 4)) * 50
    p = rng.normal(size=(40, 3)) * 30 + np.array([0.0, 0.0, 400.0])
    P = np.concatenate([[np.linalg.norm(p, axis=1).sum() + 50.0], p.sum(axis=0)])
    ref = boost_to_rest(torch.tensor(v), torch.tensor(P)).numpy()
    np.testing.assert_allclose(_boost_to_rest_np(v, P), ref, rtol=1e-10, atol=1e-8)


def test_durham_non_timelike_raises():
    kin = np.zeros((200, 4))
    kin[0] = [1.0, 0.0, 0.0, 2.0]   # spacelike: E < |p|
    with pytest.raises(ValueError, match="non-timelike"):
        durham_exclusive_membership(kin)
