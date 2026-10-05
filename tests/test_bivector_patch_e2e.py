"""End-to-end smoke test for the bivector patch-injection wiring (wrapper + model)."""

from types import SimpleNamespace

import torch

from experiments.top_tagging.embedding import build_embedding
from experiments.top_tagging.wrapper import TopTaggingWrapperBivectorPatch
from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.mlp.config import MLPConfig
from su2eqnn.nets.su2eqnn_net import SU2EQNN


def _cfg():
    return SimpleNamespace(
        data=SimpleNamespace(
            embedding="patch", inject=[], two_beams=True, rescale_data=True,
        )
    )


def _build_bivector_patch_model():
    net = SU2EQNN(
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=8,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=3,
        patch_mv_channels=4,
    )
    return TopTaggingWrapperBivectorPatch(net)


def test_bivector_patch_wrapper_forward_shape():
    """TopTaggingWrapperBivectorPatch produces one logit per jet."""
    model = _build_bivector_patch_model()

    fourmomenta = torch.randn(10, 4)
    ptr = torch.tensor([0, 4, 10])  # 2 jets: 4 and 6 particles
    axes = torch.randn(2, 4, 3)
    axes = axes / axes.norm(dim=-1, keepdim=True)

    embedding = build_embedding(_cfg(), fourmomenta, ptr, {"axes": axes})
    logits = model(embedding)

    assert logits.shape == (2, 1), f"expected (2, 1) logits, got {logits.shape}"
    assert torch.isfinite(logits).all()


def test_bivector_patch_wrapper_backward_pass():
    """Gradients flow through both the main net and the per-block patch projections."""
    model = _build_bivector_patch_model()

    fourmomenta = torch.randn(10, 4)
    ptr = torch.tensor([0, 4, 10])
    axes = torch.randn(2, 4, 3)
    axes = axes / axes.norm(dim=-1, keepdim=True)

    embedding = build_embedding(_cfg(), fourmomenta, ptr, {"axes": axes})
    logits = model(embedding)
    loss = logits.sum()
    loss.backward()

    patch_grad_found = False
    for p in model.net.patch_proj.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            patch_grad_found = True
    assert patch_grad_found, "at least one patch_proj parameter must receive a nonzero gradient"
