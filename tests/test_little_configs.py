"""Little-group configs: A/B-identical to the reference except the tested factor, and buildable."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from experiments.top_tagging.embedding import build_embedding
from experiments.top_tagging.experiment import build_model

CONFIG = Path(__file__).parent.parent / "config"
REFERENCE = "top_tagging_rest_frame_boosted_beams_full_b12"
LITTLE = {  # config -> (embedding, inject, in_mv, patch, durham data)
    "rest_xyplane": ("multi_channel", [], 1, None, False),
    "rest_durham_axes": ("multi_channel", [], 4, None, True),
    "rest_durham_patch": ("patch", [], 1, 3, True),
    "rest_durham_masses": ("multi_channel", ["subjet_mass"], 4, None, True),
    "rest_durham_handedness": ("multi_channel", ["local_handedness"], 4, None, True),
    "rest_durham_plane_normal": ("multi_channel", ["plane_normal"], 4, None, True),
    "rest_durham_single_channel": ("single_channel", [], 1, None, True),
    "rest_durham_single_channel_b": ("single_channel", ["lead_pseudoscalar"], 1, None, True),
    "rest_durham_masses_patch": ("multi_channel", ["subjet_mass"], 1, 3, True),
    "rest_durham_handedness_patch": ("multi_channel", ["local_handedness"], 1, 3, True),
    "rest_durham_plane_normal_patch": ("multi_channel", ["plane_normal"], 1, 3, True),
    "rest_durham_single_channel_patch": ("single_channel", [], 1, 1, True),
    "rest_durham_single_channel_b_patch": ("single_channel", ["lead_pseudoscalar"], 1, 1, True),
}
ALLOWED = {"data.path", "data.beam_reference", "data.patch", "data.embedding", "data.inject",
           "model.in_mv_channels", "model.patch_mv_channels", "training.checkpoint_path"}


def flat(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def load(stem):
    return OmegaConf.to_container(OmegaConf.load(CONFIG / f"{stem}.yaml"))


@pytest.mark.parametrize("name", sorted(LITTLE))
def test_only_tested_factor_differs(name):
    ref, new = flat(load(REFERENCE)), flat(load(f"top_tagging_{name}_full_b12"))
    diff = {k for k in ref.keys() | new.keys() if ref.get(k) != new.get(k)}
    assert diff <= ALLOWED, diff - ALLOWED
    embedding, inject, in_mv, patch, durham = LITTLE[name]
    assert new["data.beam_reference"] == "xyplane" and new["data.beams"] == "boosted"
    assert new.get("data.embedding", "multi_channel") == embedding
    assert new.get("data.inject", []) == inject
    assert new["model.in_mv_channels"] == in_mv
    assert new.get("model.patch_mv_channels") == patch
    assert new["data.path"].endswith("_durham.npz") == durham
    assert new["training.axis_dropout_p"] == 0.0
    is_flag_patch = name.endswith("_patch") and name != "rest_durham_patch"
    assert new.get("data.patch", False) == is_flag_patch
    assert new["training.checkpoint_path"] == f"checkpoints/top_tagging_{name}_full_b12.pt"


@pytest.mark.parametrize("name", sorted(LITTLE))
def test_config_builds_and_embeds(name):
    c = OmegaConf.load(CONFIG / f"top_tagging_{name}_full_b12.yaml")
    c.model.num_blocks, c.model.hidden_mv_channels, c.model.num_heads = 1, 4, 2
    x = torch.tensor([[50.0, 5.0, 0.0, 49.0], [40.0, -4.0, 3.0, 39.0], [30.0, 1.0, -3.0, 29.0],
                      [20.0, -2.0, -1.0, 19.5]], dtype=torch.float64)
    ptr = torch.tensor([0, 4])
    aux = {"subjet_idx": torch.tensor([0, 1, 2, 0])} if LITTLE[name][4] else {}
    model = build_model(c).double().eval()
    with torch.no_grad():
        out = model(build_embedding(c, x, ptr, aux))
    assert out.shape == (1, 1) and torch.isfinite(out).all()


NAIVE_PATCH = {  # config (lab, xy-plane spurion) -> patch_mv_channels
    "subjet_mass_patch": 4, "pseudoscalar_triad_patch": 4, "pseudoscalar_handedness_patch": 4,
    "single_channel_axis_patch": 1, "single_channel_axis_b_patch": 1,
}


@pytest.mark.parametrize("name", sorted(NAIVE_PATCH))
def test_naive_patch_config_builds_and_embeds(name):
    c = OmegaConf.load(CONFIG / f"top_tagging_{name}_xyplane_full_b12.yaml")
    assert c.data.patch is True and c.model.in_mv_channels == 1
    assert c.model.patch_mv_channels == NAIVE_PATCH[name]
    assert c.data.beam_reference == "xyplane"
    c.model.num_blocks, c.model.hidden_mv_channels, c.model.num_heads = 1, 4, 2
    x = torch.tensor([[50.0, 5.0, 0.0, 49.0], [40.0, -4.0, 3.0, 39.0], [30.0, 1.0, -3.0, 29.0],
                      [20.0, -2.0, -1.0, 19.5]], dtype=torch.float64)
    ptr = torch.tensor([0, 4])
    g = torch.Generator().manual_seed(0)
    axes = torch.randn(1, 4, 3, generator=g, dtype=torch.float64)
    aux = {"axes": axes / axes.norm(dim=-1, keepdim=True),
           "masses": torch.rand(1, 4, generator=g, dtype=torch.float64) * 80,
           "p_lead": x[:1]}
    model = build_model(c).double().eval()
    with torch.no_grad():
        out = model(build_embedding(c, x, ptr, aux))
    assert out.shape == (1, 1) and torch.isfinite(out).all()
