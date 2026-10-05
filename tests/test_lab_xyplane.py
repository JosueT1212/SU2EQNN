"""frame: lab with beam_reference: xyplane — the naive partner of the little-group runs.

One fixed token -e12 (the unboosted, unit xy-plane bivector) replaces the +-e3 pair,
so naive vs little group differ only in the frame.
"""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from experiments.top_tagging.embedding import build_embedding
from experiments.top_tagging.embedding.core import rest_frame_xyplane
from tests.test_rest_frame import model_output_aux, rotation_matrix, toy_jets

NAIVE = ["r1", "grade2_bivectors", "bivector_patch", "subjet_mass", "pseudoscalar_triad",
         "pseudoscalar_handedness", "single_channel_axis", "single_channel_axis_b"]


def lab_cfg(embedding, inject=()):
    return SimpleNamespace(data=SimpleNamespace(
        embedding=embedding, inject=list(inject), two_beams=True, rescale_data=True))


def xy_cfg(embedding="multi_channel", inject=()):
    c = lab_cfg(embedding, inject)
    c.data.beam_reference = "xyplane"
    return c


def test_lab_xyplane_token_is_minus_e12():
    x, ptr = toy_jets()
    emb = build_embedding(xy_cfg(), x, ptr, {})
    tokens = emb["mv"][emb["is_beam"]]
    assert tokens.shape[0] == len(ptr) - 1          # one token per jet
    expected = torch.zeros(8, dtype=x.dtype)
    expected[4] = -1.0
    assert torch.equal(tokens[:, 0], expected.expand_as(tokens[:, 0]))


def test_lab_xyplane_equals_rest_token_of_jet_at_rest():
    P = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    x, ptr = toy_jets()
    emb = build_embedding(xy_cfg(), x, ptr, {})
    assert torch.equal(emb["mv"][emb["is_beam"]][:1, 0], rest_frame_xyplane(P, x.dtype)[:, 0])


def test_lab_xyplane_model_invariant_under_z_rotation():
    c = xy_cfg()
    c.model = SimpleNamespace(hidden_mv_channels=4, num_blocks=2, num_heads=2,
                              multi_query=False, in_mv_channels=1)
    x, ptr = toy_jets()
    R = rotation_matrix([0, 0, 1], 0.7)
    out, out_R = model_output_aux(c, x, ptr, {}), model_output_aux(c, x @ R.T, ptr, {})
    assert torch.allclose(out, out_R, rtol=1e-8, atol=1e-9)


def test_lab_rejects_unknown_beam_reference():
    c = lab_cfg("multi_channel")
    c.data.beam_reference = "time"
    x, ptr = toy_jets()
    with pytest.raises(ValueError, match="beam_reference"):
        build_embedding(c, x, ptr, {})


@pytest.mark.parametrize("name", NAIVE)
def test_xyplane_config_is_twin_plus_beam_reference(name):
    flat = lambda n: {f"{s}.{k}": v for s, d in OmegaConf.to_container(
        OmegaConf.load(f"config/top_tagging_{n}_full_b12.yaml")).items() for k, v in d.items()}
    twin, new = flat(name), flat(f"{name}_xyplane")
    assert new.pop("data.beam_reference") == "xyplane"
    assert new.pop("training.checkpoint_path") == f"checkpoints/top_tagging_{name}_xyplane_full_b12.pt"
    twin.pop("training.checkpoint_path")
    assert new == twin
