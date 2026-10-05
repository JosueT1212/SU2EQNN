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
        model=SimpleNamespace(hidden_mv_channels=4, num_blocks=2, num_heads=2, multi_query=False),
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


# --- 6: rest-frame Durham subjets (little-group injections) -----------------

from experiments.top_tagging.embedding import injectors
from experiments.top_tagging.embedding.core import rest_subjets
from su2eqnn.utils.subjets import durham_exclusive_membership

# name -> (embedding, inject, in_mv_channels, patch_mv_channels)
LITTLE = {
    "a": ("multi_channel", [], 4, None),
    "b": ("patch", [], 1, 3),
    "c": ("multi_channel", ["subjet_mass"], 4, None),
    "d": ("multi_channel", ["local_handedness"], 4, None),
    "e": ("multi_channel", ["plane_normal"], 4, None),
    "f": ("single_channel", [], 1, None),
    "g": ("single_channel", ["lead_pseudoscalar"], 1, None),
}


def cfg_little(name, beams="none", beam_reference=None):
    embedding, inject, in_mv, patch = LITTLE[name]
    c = cfg("rest", beams, embedding=embedding, inject=inject)
    c.data.beam_reference = beam_reference
    c.model.in_mv_channels = in_mv
    c.model.patch_mv_channels = patch
    return c


def durham_idx(x, ptr):
    return torch.cat([
        torch.from_numpy(durham_exclusive_membership(x[a:b].numpy()).astype(np.int64))
        for a, b in zip(ptr[:-1].tolist(), ptr[1:].tolist())
    ])


def model_output_aux(c, x, ptr, aux):
    torch.manual_seed(0)
    model = build_model(c).double().eval()
    with torch.no_grad():
        return model(build_embedding(c, x, ptr, aux))


@pytest.mark.parametrize("name", sorted(LITTLE))
@pytest.mark.parametrize("seed", [0, 1])
def test_little_variants_lorentz_invariant(name, seed):
    x, ptr = toy_jets(seed=seed)
    L = random_lorentz(seed)
    idx = durham_idx(x, ptr)
    assert torch.equal(durham_idx(x @ L.T, ptr), idx)          # membership invariant
    c = cfg_little(name)
    out = model_output_aux(c, x, ptr, {"subjet_idx": idx})
    out_L = model_output_aux(c, x @ L.T, ptr, {"subjet_idx": idx})
    assert torch.allclose(out, out_L, rtol=1e-8, atol=1e-9)


def test_rest_subjets_sum_to_zero_and_lead_is_max_energy():
    x, ptr = toy_jets()
    P, batch = jet_momenta(x, ptr)
    xr = boost_to_rest(x, P[batch])
    out = rest_subjets(xr, ptr, durham_idx(x, ptr))
    assert out["axes"].shape == (len(ptr) - 1, 3, 3) and out["masses"].shape == (len(ptr) - 1, 3)
    assert torch.allclose(out["axes"].norm(dim=-1), torch.ones(len(ptr) - 1, 3, dtype=torch.float64))
    for j, (a, b) in enumerate(zip(ptr[:-1].tolist(), ptr[1:].tolist())):
        assert torch.equal(out["p_lead"][j], xr[a + int(xr[a:b, 0].argmax())])
    # Σ p_s = 0: the three axes weighted by |p_s| cancel
    idx = durham_idx(x, ptr)
    ps = torch.zeros(len(ptr) - 1, 3, 3, dtype=torch.float64).index_put_(
        (batch, idx), xr[:, 1:], accumulate=True)
    m = torch.sqrt(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(-1))
    assert ps.sum(1).abs().max() <= 1e-9 * m.max()
    assert torch.allclose(out["axes"] * ps.norm(dim=-1, keepdim=True), ps, atol=1e-9)


def test_rest_subjets_empty_subjet_gives_zero_axis():
    x = torch.tensor([[50.0, 5.0, 0.0, 49.0], [40.0, -4.0, 3.0, 39.0]], dtype=torch.float64)
    ptr = torch.tensor([0, 2])
    c = cfg_little("c")
    emb = build_embedding(c, x, ptr, {"subjet_idx": durham_idx(x, ptr)})
    assert torch.isfinite(emb["mv"]).all()
    P, batch = jet_momenta(x, ptr)
    out = rest_subjets(boost_to_rest(x, P[batch]), ptr, durham_idx(x, ptr))
    assert (out["axes"][0, 2] == 0).all() and out["masses"][0, 2] == 0


def test_rest_subjets_rejects_padding_label():
    x, ptr = toy_jets()
    idx = durham_idx(x, ptr)
    idx[0] = -1
    with pytest.raises(ValueError, match="subjet_idx"):
        build_embedding(cfg_little("a"), x, ptr, {"subjet_idx": idx})


def test_little_float32_path():
    x, ptr = toy_jets()
    idx = durham_idx(x, ptr)
    emb = build_embedding(cfg_little("a"), x.float(), ptr, {"subjet_idx": idx})
    assert emb["mv"].dtype == torch.float32 and emb["mv"].shape[1] == 4
    assert torch.isfinite(emb["mv"]).all()


def test_build_embedding_does_not_mutate_aux():
    x, ptr = toy_jets()
    aux = {"subjet_idx": durham_idx(x, ptr)}
    build_embedding(cfg_little("c"), x, ptr, aux)
    assert list(aux) == ["subjet_idx"]


def test_rest_injector_without_subjet_idx_raises():
    x, ptr = toy_jets()
    with pytest.raises(ValueError, match="frame: rest"):
        build_embedding(cfg_little("b"), x, ptr, {})


def test_rest_rejects_lab_only_injector():
    x, ptr = toy_jets()
    c = cfg("rest", "none", inject=["triad_handedness"])
    with pytest.raises(ValueError, match="frame: rest"):
        build_embedding(c, x, ptr, {"subjet_idx": durham_idx(x, ptr)})


def test_lab_rejects_subjet_idx():
    x, ptr = toy_jets()
    with pytest.raises(ValueError, match="subjet_idx"):
        build_embedding(cfg("lab", "lab"), x, ptr, {"subjet_idx": durham_idx(x, ptr)})


def test_subjet_mass_width_mismatch_raises():
    mv = torch.zeros(3, 5, 8)
    aux = {"masses": torch.ones(1, 3)}
    with pytest.raises(ValueError, match="subjet_mass"):
        injectors.subjet_mass(mv, None, None, torch.zeros(3, dtype=torch.long), aux, False)


def test_plane_normal_values_and_parity():
    n = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0], [-0.6, -0.8, 0]]], dtype=torch.float64)
    p4 = torch.tensor([[20.0, 3.0, 4.0, 12.0]], dtype=torch.float64)
    mv = torch.zeros(1, 4, 8, dtype=torch.float64)
    injectors.plane_normal(mv, p4, None, torch.zeros(1, dtype=torch.long), {"axes": n}, True)
    p = p4[0, 1:] / 20.0
    for k in range(3):
        expect = p @ torch.linalg.cross(n[0, k], n[0, (k + 1) % 3])
        assert torch.isclose(mv[0, 1 + k, 7], expect)
    # parity: p -> -p, axes -> -axes flips the pseudoscalar
    mv_p = torch.zeros_like(mv)
    p4_p = p4.clone()
    p4_p[:, 1:] *= -1
    injectors.plane_normal(mv_p, p4_p, None, torch.zeros(1, dtype=torch.long), {"axes": -n}, True)
    assert torch.allclose(mv_p[:, 1:4, 7], -mv[:, 1:4, 7])


def test_plane_normal_rejected_in_lab_frame():
    x, ptr = toy_jets()
    axes = torch.randn(len(ptr) - 1, 3, 3, dtype=x.dtype)
    with pytest.raises(ValueError, match="frame: rest"):
        build_embedding(cfg("lab", "lab", inject=["plane_normal"]), x, ptr, {"axes": axes})


# --- 7: boosted xy-plane spurion ---------------------------------------------

from experiments.top_tagging.embedding.core import rest_frame_xyplane


def test_xyplane_token_at_rest_and_under_z_boost():
    for y in (0.0, 1.7, -2.3):
        P = torch.tensor([[2.0 * math.cosh(y), 0.0, 0.0, 2.0 * math.sinh(y)]], dtype=torch.float64)
        tok = rest_frame_xyplane(P, torch.float64)
        expect = torch.tensor([[[0, 0, 0, 0, -1.0, 0, 0, 0]]], dtype=torch.float64)
        assert torch.allclose(tok, expect, atol=1e-12)


def test_xyplane_token_transverse_boost_known_answer():
    y = 0.9
    P = torch.tensor([[3.0 * math.cosh(y), 3.0 * math.sinh(y), 0.0, 0.0]], dtype=torch.float64)
    tok = rest_frame_xyplane(P, torch.float64)
    expect = torch.tensor([[[0, 0, math.sinh(y), 0, -math.cosh(y), 0, 0, 0]]], dtype=torch.float64)
    expect = expect / math.sqrt(math.cosh(2 * y))
    assert torch.allclose(tok, expect, atol=1e-12)


def test_xyplane_token_invariants():
    g = torch.Generator().manual_seed(5)
    p = torch.randn(20, 3, generator=g, dtype=torch.float64) * 200
    P = torch.cat([torch.sqrt((p ** 2).sum(-1, keepdim=True) + 150.0 ** 2), p], dim=1)
    tok = rest_frame_xyplane(P, torch.float64)[:, 0]
    E, B = tok[:, 1:4], tok[:, 4:7]
    assert torch.allclose(tok.norm(dim=-1), torch.ones(20, dtype=torch.float64))
    pT = torch.hypot(P[:, 1], P[:, 2])
    m = torch.sqrt(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(-1))
    assert torch.allclose(E.norm(dim=-1) / B.norm(dim=-1), pT / torch.sqrt(pT ** 2 + m ** 2))
    # simple bivector: E . *B = 0 with *B = (b23, -b13, b12)
    Bdual = torch.stack([B[:, 2], -B[:, 1], B[:, 0]], dim=-1)
    assert (E * Bdual).sum(-1).abs().max() < 1e-9
    assert (tok[:, [0, 7]] == 0).all()


def test_xyplane_one_token_per_jet():
    x, ptr = toy_jets()
    c = cfg("rest", "boosted")
    c.data.beam_reference = "xyplane"
    emb = build_embedding(c, x, ptr, {})
    assert emb["is_beam"].sum() == len(ptr) - 1


@pytest.mark.parametrize("name", ["base", *sorted(LITTLE)])
def test_xyplane_invariant_under_beam_stabilizer(name):
    x, ptr = toy_jets()
    L = rotation_matrix([0, 0, 1], 0.7) @ boost_matrix(1.3, [0, 0, 1])
    if name == "base":
        c, aux = cfg("rest", "boosted"), {}
        c.data.beam_reference = "xyplane"
    else:
        c, aux = cfg_little(name, "boosted", "xyplane"), {"subjet_idx": durham_idx(x, ptr)}
    out = model_output_aux(c, x, ptr, aux)
    out_L = model_output_aux(c, x @ L.T, ptr, aux)
    assert torch.allclose(out, out_L, rtol=1e-8, atol=1e-9)


def test_xyplane_breaks_transverse_boosts():
    x, ptr = toy_jets()
    c = cfg("rest", "boosted")
    c.data.beam_reference = "xyplane"
    L = boost_matrix(0.8, [1, 0, 0])
    assert (model_output_aux(c, x, ptr, {}) - model_output_aux(c, x @ L.T, ptr, {})).abs().max() > 1e-6


def test_xyplane_guards():
    x, ptr = toy_jets()
    c = cfg("rest", "none")
    c.data.beam_reference = "xyplane"
    with pytest.raises(ValueError, match="beams: boosted"):
        build_embedding(c, x, ptr, {})
    c = cfg("lab", "lab")
    c.data.beam_reference = "xyplane"      # lab xyplane is allowed (fixed -e12, test_lab_xyplane)
    c.data.beams = "boosted"
    with pytest.raises(ValueError, match="frame: rest"):
        build_embedding(c, x, ptr, {})
