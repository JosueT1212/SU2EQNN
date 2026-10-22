"""data.patch: true turns any input-injection version into its patch version.

Input keeps only (E, p) in channel 0; everything the version adds is re-injected
every block through patch_mv. With no injector it must reproduce the existing
`patch` layout bit for bit.
"""

from types import SimpleNamespace

import pytest
import torch

from experiments.top_tagging.embedding import build_embedding
from experiments.top_tagging.experiment import build_model
from tests.test_rest_frame import (
    boost_matrix, durham_idx, model_output_aux, rotation_matrix, toy_jets,
)


def lab_cfg(embedding, inject=(), patch=False):
    return SimpleNamespace(data=SimpleNamespace(
        embedding=embedding, inject=list(inject), two_beams=True, rescale_data=True,
        patch=patch))


def lab_aux(ptr, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(ptr) - 1
    axes = torch.randn(n, k, 3, generator=g, dtype=torch.float64)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    return {"axes": axes, "masses": torch.rand(n, 4, generator=g, dtype=torch.float64) * 80,
            "p_lead": torch.randn(n, 4, generator=g, dtype=torch.float64) * 50}


def test_patch_flag_without_injector_equals_patch_layout():
    x, ptr = toy_jets()
    aux = {"axes": lab_aux(ptr)["axes"]}
    ref = build_embedding(lab_cfg("patch"), x, ptr, aux)
    new = build_embedding(lab_cfg("multi_channel", patch=True), x, ptr, aux)
    assert torch.equal(new["mv"], ref["mv"])
    assert torch.equal(new["patch_mv"], ref["patch_mv"])
    assert torch.equal(new["is_beam"], ref["is_beam"])


@pytest.mark.parametrize("inject", [["subjet_mass"], ["local_handedness"], ["triad_handedness"]])
def test_multi_channel_patch_splits_input_version(inject):
    x, ptr = toy_jets()
    aux = lab_aux(ptr)
    full = build_embedding(lab_cfg("multi_channel", inject), x, ptr, aux)
    new = build_embedding(lab_cfg("multi_channel", inject, patch=True), x, ptr, aux)
    assert torch.equal(new["mv"], full["mv"][:, :1])
    assert torch.equal(new["patch_mv"], full["mv"][:, 1:])


@pytest.mark.parametrize("inject", [[], ["lead_pseudoscalar"]])
def test_single_channel_patch_moves_slots_4_to_7(inject):
    x, ptr = toy_jets()
    aux = lab_aux(ptr)
    full = build_embedding(lab_cfg("single_channel", inject), x, ptr, aux)
    new = build_embedding(lab_cfg("single_channel", inject, patch=True), x, ptr, aux)
    assert torch.equal(new["mv"][:, 0, :4], full["mv"][:, 0, :4])
    assert (new["mv"][:, 0, 4:] == 0).all()
    assert new["patch_mv"].shape[1] == 1
    assert torch.equal(new["patch_mv"][:, 0, 4:], full["mv"][:, 0, 4:])
    assert (new["patch_mv"][:, 0, :4] == 0).all()


def test_patch_flag_rejects_patch_layout():
    x, ptr = toy_jets()
    with pytest.raises(ValueError, match="patch"):
        build_embedding(lab_cfg("patch", patch=True), x, ptr, lab_aux(ptr))


# rest-frame patch versions: (embedding, inject, patch_mv_channels)
LITTLE_PATCH = {
    "c": ("multi_channel", ["subjet_mass"], 3),
    "d": ("multi_channel", ["local_handedness"], 3),
    "e": ("multi_channel", ["plane_normal"], 3),
    "f": ("single_channel", [], 1),
    "g": ("single_channel", ["lead_pseudoscalar"], 1),
}


@pytest.mark.parametrize("name", sorted(LITTLE_PATCH))
def test_little_patch_invariant_under_beam_stabilizer(name):
    embedding, inject, k = LITTLE_PATCH[name]
    c = SimpleNamespace(
        data=SimpleNamespace(embedding=embedding, inject=inject, two_beams=True,
                             rescale_data=True, frame="rest", beams="boosted",
                             beam_reference="xyplane", patch=True),
        model=SimpleNamespace(hidden_mv_channels=4, num_blocks=2, num_heads=2,
                              multi_query=False, in_mv_channels=1, patch_mv_channels=k))
    x, ptr = toy_jets()
    aux = {"subjet_idx": durham_idx(x, ptr)}
    L = rotation_matrix([0, 0, 1], 0.7) @ boost_matrix(1.3, [0, 0, 1])
    out = model_output_aux(c, x, ptr, aux)
    out_L = model_output_aux(c, x @ L.T, ptr, aux)
    assert torch.isfinite(out).all()
    # Rest-frame subjet masses carry ~1e-7 float64 cancellation noise after the boost
    # (the input versions too); patch re-injects it every block, so the bound is looser
    # here. A broken symmetry moves the output at O(1e-1).
    assert torch.allclose(out, out_L, rtol=1e-6, atol=1e-7)


def test_little_patch_detects_broken_symmetry():
    embedding, inject, k = LITTLE_PATCH["c"]
    c = SimpleNamespace(
        data=SimpleNamespace(embedding=embedding, inject=inject, two_beams=True,
                             rescale_data=True, frame="rest", beams="boosted",
                             beam_reference="xyplane", patch=True),
        model=SimpleNamespace(hidden_mv_channels=4, num_blocks=2, num_heads=2,
                              multi_query=False, in_mv_channels=1, patch_mv_channels=k))
    x, ptr = toy_jets()
    aux = {"subjet_idx": durham_idx(x, ptr)}
    L = rotation_matrix([1, 0, 0], 0.7)                  # x-rotation moves the beam frame
    out, out_L = model_output_aux(c, x, ptr, aux), model_output_aux(c, x @ L.T, ptr, aux)
    assert not torch.allclose(out, out_L, rtol=1e-6, atol=1e-7)


def test_patch_mv_zero_on_beam_tokens():
    x, ptr = toy_jets()
    emb = build_embedding(lab_cfg("multi_channel", ["subjet_mass"], patch=True), x, ptr, lab_aux(ptr))
    assert emb["is_beam"].any()
    assert (emb["patch_mv"][emb["is_beam"]] == 0).all()


def test_patch_split_float32():
    x, ptr = toy_jets()
    aux = {k: v.float() for k, v in lab_aux(ptr).items()}
    emb = build_embedding(lab_cfg("single_channel", ["lead_pseudoscalar"], patch=True),
                          x.float(), ptr, aux)
    assert emb["mv"].dtype == emb["patch_mv"].dtype == torch.float32
    assert torch.isfinite(emb["mv"]).all() and torch.isfinite(emb["patch_mv"]).all()


def test_patch_flag_without_patch_channels_raises():
    """data.patch with no model.patch_mv_channels would silently train the baseline."""
    c = SimpleNamespace(data=SimpleNamespace(patch=True),
                        model=SimpleNamespace(hidden_mv_channels=4, num_blocks=1, num_heads=2,
                                              in_mv_channels=1))
    with pytest.raises(ValueError, match="patch_mv_channels"):
        build_model(c)


def test_baseline_patch_reinjects_kinematics():
    """Baseline patch: input unchanged, patch = a copy of (E, p) in channel 0."""
    x, ptr = toy_jets()
    base = build_embedding(lab_cfg("multi_channel"), x, ptr, {})
    new = build_embedding(lab_cfg("multi_channel", patch=True), x, ptr, {})
    assert torch.equal(new["mv"], base["mv"])
    assert new["patch_mv"].shape[1] == 1
    particles = ~new["is_beam"]
    assert torch.equal(new["patch_mv"][particles], new["mv"][particles])
    assert (new["patch_mv"][new["is_beam"]] == 0).all()


def test_little_baseline_patch_invariant_under_beam_stabilizer():
    c = SimpleNamespace(
        data=SimpleNamespace(embedding="multi_channel", inject=[], two_beams=True,
                             rescale_data=True, frame="rest", beams="boosted",
                             beam_reference="xyplane", patch=True),
        model=SimpleNamespace(hidden_mv_channels=4, num_blocks=2, num_heads=2,
                              multi_query=False, in_mv_channels=1, patch_mv_channels=1))
    x, ptr = toy_jets()
    L = rotation_matrix([0, 0, 1], 0.7) @ boost_matrix(1.3, [0, 0, 1])
    out, out_L = model_output_aux(c, x, ptr, {}), model_output_aux(c, x @ L.T, ptr, {})
    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_L, rtol=1e-6, atol=1e-7)
    L = rotation_matrix([1, 0, 0], 0.7)                  # x-rotation moves the beam frame
    assert not torch.allclose(out, model_output_aux(c, x @ L.T, ptr, {}), rtol=1e-6, atol=1e-7)
