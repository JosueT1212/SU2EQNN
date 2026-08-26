"""precompute_features.py must reproduce the four scripts it replaces."""

import numpy as np
import pytest

from scripts.precompute_features import FEATURES, compute_feature

N_JETS = 4
N_PAD = 12


@pytest.fixture
def kinematics():
    rng = np.random.default_rng(20260825)
    k = np.zeros((N_JETS, N_PAD, 4), dtype=np.float32)
    for i in range(N_JETS):
        n_real = 5 + i
        p3 = rng.normal(scale=30.0, size=(n_real, 3))
        e = np.linalg.norm(p3, axis=-1) + rng.uniform(1.0, 5.0, size=n_real)
        k[i, :n_real, 0] = e
        k[i, :n_real, 1:] = p3
    return k


def test_all_features_registered():
    assert set(FEATURES) == {"axes", "masses", "phases", "leading"}


@pytest.mark.parametrize(
    "name,shape",
    [("axes", (N_JETS, 4, 3)), ("masses", (N_JETS, 4)),
     ("phases", (N_JETS, 4)), ("leading", (N_JETS, 4))],
)
def test_feature_shape_and_dtype(kinematics, name, shape):
    out = compute_feature(name, kinematics)
    assert out.shape == shape
    assert out.dtype == np.float32


def test_axes_are_unit_vectors(kinematics):
    axes = compute_feature("axes", kinematics)
    norms = np.linalg.norm(axes, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_masses_non_negative(kinematics):
    assert (compute_feature("masses", kinematics) >= 0.0).all()


def test_phases_in_unit_interval(kinematics):
    ph = compute_feature("phases", kinematics)
    assert (np.abs(ph) <= 1.0 + 1e-6).all()
