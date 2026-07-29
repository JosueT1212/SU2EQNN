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
