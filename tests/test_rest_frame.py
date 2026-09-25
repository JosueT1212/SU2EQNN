"""Rest-frame (little-group) canonicalization: frame sanity, the canonicalization
theorem f(Λx) = f(x), and the Curie behaviour of boosted beam tokens.

Theory: Draft/little-group-canonicalization.tex, Thm 5.2 and §7 ¶2.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.top_tagging.embedding import build_embedding
from experiments.top_tagging.embedding.core import boost_to_rest, random_z_boost
from experiments.top_tagging.experiment import build_model

MINI = Path(__file__).parent.parent / "data" / "toptagging_mini.npz"


def cfg(frame="rest", beams="none", embedding="multi_channel", inject=()):
    return SimpleNamespace(
        data=SimpleNamespace(
            embedding=embedding, inject=list(inject), two_beams=True,
            rescale_data=True, frame=frame, beams=beams,
        ),
        model=SimpleNamespace(hidden_mv_channels=4, num_blocks=2, num_heads=2),
    )


def toy_jets(n_jets=4, n_min=5, n_max=12, seed=0):
    """Massless float64 constituents, boosted along +z like real jets."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(n_min, n_max + 1, (n_jets,), generator=g)
    n = int(counts.sum())
    p = torch.randn(n, 3, generator=g, dtype=torch.float64) * torch.tensor(
        [30.0, 30.0, 30.0], dtype=torch.float64
    ) + torch.tensor([0.0, 0.0, 200.0], dtype=torch.float64)
    x = torch.cat([p.norm(dim=1, keepdim=True), p], dim=1)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return x, ptr


def mini_jets(n_jets=50):
    raw = np.load(MINI)
    kin = torch.tensor(raw["kinematics_test"][:n_jets], dtype=torch.float64)
    rows, counts = [], []
    for jet in kin:
        real = jet[(jet.abs() > 1e-5).any(dim=-1)]
        rows.append(real)
        counts.append(len(real))
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(counts).cumsum(0)])
    return torch.cat(rows), ptr


def boost_matrix(rapidity, direction):
    n = torch.as_tensor(direction, dtype=torch.float64)
    n = n / n.norm()
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    L = torch.eye(4, dtype=torch.float64)
    L[0, 0] = ch
    L[0, 1:] = sh * n
    L[1:, 0] = sh * n
    L[1:, 1:] += (ch - 1) * torch.outer(n, n)
    return L


def rotation_matrix(axis, angle):
    k = torch.as_tensor(axis, dtype=torch.float64)
    k = k / k.norm()
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
                     dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    L = torch.eye(4, dtype=torch.float64)
    L[1:, 1:] = R
    return L


def random_lorentz(seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(3, generator=g, dtype=torch.float64)
    return (rotation_matrix(torch.randn(3, generator=g), float(r[0]) * 6.28)
            @ boost_matrix(float(r[1]) * 3.0, torch.randn(3, generator=g)))


def model_output(c, x, ptr):
    torch.manual_seed(0)
    model = build_model(c).double().eval()
    with torch.no_grad():
        return model(build_embedding(c, x, ptr, {}))


def jet_momenta(x, ptr):
    batch = torch.arange(len(ptr) - 1).repeat_interleave(ptr[1:] - ptr[:-1])
    return torch.zeros(len(ptr) - 1, 4, dtype=x.dtype).index_add_(0, batch, x), batch


# --- 1-2: frame sanity and known answers -----------------------------------

def test_boost_to_rest_known_answers():
    P = torch.tensor([5.0, 3.0, 0.0, 0.0], dtype=torch.float64)   # m = 4
    v = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0], [5.0, 3.0, 0.0, 0.0]],
                     dtype=torch.float64)
    out = boost_to_rest(v, P)
    expected = torch.tensor([[0.5, 0.5, 0, 0], [2.0, -2.0, 0, 0], [4.0, 0, 0, 0]],
                            dtype=torch.float64)
    assert torch.allclose(out, expected, atol=1e-12)


def test_boost_to_rest_identity_at_rest():
    v = torch.randn(7, 4, dtype=torch.float64)
    P = torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(boost_to_rest(v, P), v, atol=1e-12)


@pytest.mark.parametrize("source", ["toy", "mini"])
def test_rest_frame_sanity(source):
    if source == "mini" and not MINI.exists():
        pytest.skip("mini dataset not present")
    x, ptr = toy_jets() if source == "toy" else mini_jets()
    P, batch = jet_momenta(x, ptr)
    xb = boost_to_rest(x, P[batch])
    Pb, _ = jet_momenta(xb, ptr)
    m = torch.sqrt(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(-1))
    assert torch.allclose(Pb[:, 0], m, rtol=1e-9)
    assert Pb[:, 1:].abs().max() <= 1e-9 * m.max()
    assert (xb[:, 0] > 0).all()


# --- 3: canonicalization theorem --------------------------------------------

@pytest.mark.parametrize("source", ["toy", "mini"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rest_none_is_lorentz_invariant(source, seed):
    if source == "mini" and not MINI.exists():
        pytest.skip("mini dataset not present")
    x, ptr = toy_jets(seed=seed) if source == "toy" else mini_jets()
    L = random_lorentz(seed)
    out = model_output(cfg("rest", "none"), x, ptr)
    out_L = model_output(cfg("rest", "none"), x @ L.T, ptr)
    assert torch.allclose(out, out_L, rtol=1e-8, atol=1e-9)


def test_lab_frame_is_not_lorentz_invariant():
    x, ptr = toy_jets()
    L = random_lorentz(0)
    out = model_output(cfg("lab", "lab"), x, ptr)
    out_L = model_output(cfg("lab", "lab"), x @ L.T, ptr)
    assert (out - out_L).abs().max() > 1e-3


# --- 4: Curie behaviour of boosted beams ------------------------------------

def test_boosted_beams_invariant_under_beam_stabilizer():
    x, ptr = toy_jets()
    L = rotation_matrix([0, 0, 1], 0.7) @ boost_matrix(1.3, [0, 0, 1])
    out = model_output(cfg("rest", "boosted"), x, ptr)
    out_L = model_output(cfg("rest", "boosted"), x @ L.T, ptr)
    assert torch.allclose(out, out_L, rtol=1e-8, atol=1e-9)


def test_boosted_beams_break_transverse_boosts():
    x, ptr = toy_jets()
    L = boost_matrix(0.8, [1, 0, 0])
    out = model_output(cfg("rest", "boosted"), x, ptr)
    out_L = model_output(cfg("rest", "boosted"), x @ L.T, ptr)
    assert (out - out_L).abs().max() > 1e-6


def test_boosted_beam_tokens_are_unit_grade1():
    x, ptr = toy_jets()
    emb = build_embedding(cfg("rest", "boosted"), x, ptr, {})
    beams = emb["mv"][emb["is_beam"], 0]
    assert emb["is_beam"].sum() == 2 * (len(ptr) - 1)
    assert torch.allclose(beams[:, 1:4].norm(dim=-1), torch.ones(len(beams), dtype=x.dtype))
    assert (beams[:, [0, 4, 5, 6, 7]] == 0).all()


def test_rest_none_has_no_beam_tokens():
    x, ptr = toy_jets()
    emb = build_embedding(cfg("rest", "none"), x, ptr, {})
    assert emb["mv"].shape[0] == x.shape[0]
    assert not emb["is_beam"].any()


# --- OOD helper ---------------------------------------------------------------

def test_random_z_boost_preserves_masses_and_rest_output():
    x, ptr = toy_jets()
    g = torch.Generator().manual_seed(0)
    xz = random_z_boost(x, ptr, 2.0, g)
    P, _ = jet_momenta(x, ptr)
    Pz, _ = jet_momenta(xz, ptr)
    m2 = lambda Q: Q[:, 0] ** 2 - (Q[:, 1:] ** 2).sum(-1)
    assert torch.allclose(m2(P), m2(Pz), rtol=1e-9)
    assert not torch.allclose(x, xz)
    for beams in ("none", "boosted"):
        c = cfg("rest", beams)
        assert torch.allclose(model_output(c, x, ptr), model_output(c, xz, ptr),
                              rtol=1e-8, atol=1e-9)


# --- 5: guards ----------------------------------------------------------------

def test_rest_rejects_injectors():
    x, ptr = toy_jets()
    with pytest.raises(ValueError, match="frame: rest"):
        build_embedding(cfg("rest", "none", inject=["subjet_mass"]), x, ptr, {})


def test_rest_rejects_axes_aux():
    x, ptr = toy_jets()
    axes = torch.randn(len(ptr) - 1, 4, 3, dtype=x.dtype)
    with pytest.raises(ValueError, match="frame: rest"):
        build_embedding(cfg("rest", "none"), x, ptr, {"axes": axes})


def test_rest_rejects_lab_beams():
    x, ptr = toy_jets()
    with pytest.raises(ValueError, match="beams"):
        build_embedding(cfg("rest", "lab"), x, ptr, {})


def test_lab_rejects_rest_beams():
    x, ptr = toy_jets()
    with pytest.raises(ValueError, match="beams"):
        build_embedding(cfg("lab", "none"), x, ptr, {})


def test_rest_rejects_non_timelike_jet():
    x = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)   # single massless
    with pytest.raises(ValueError, match="non-timelike"):
        build_embedding(cfg("rest", "none"), x, torch.tensor([0, 1]), {})


# --- OOD evaluation wiring -----------------------------------------------------

def test_evaluate_random_boost_flat_for_rest_and_moves_lab():
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from experiments.top_tagging.experiment import evaluate

    x, ptr = toy_jets(n_jets=16, seed=3)
    data = [Data(x=x[ptr[i]:ptr[i + 1]].float(), label=torch.tensor(i % 2 == 0))
            for i in range(len(ptr) - 1)]
    loader = DataLoader(data, batch_size=8, shuffle=False)

    def scores(frame, beams, Y):
        c = cfg(frame, beams)
        c.evaluation = SimpleNamespace(random_boost_rapidity=Y, boost_seed=0)
        torch.manual_seed(0)
        model = build_model(c).eval()
        return evaluate(model, loader, c, torch.device("cpu"))["auc"]

    assert scores("rest", "none", 2.0) == pytest.approx(scores("rest", "none", 0.0), abs=1e-6)
    assert scores("rest", "boosted", 2.0) == pytest.approx(scores("rest", "boosted", 0.0), abs=1e-6)
    assert scores("lab", "lab", 2.0) != pytest.approx(scores("lab", "lab", 0.0), abs=1e-6)
