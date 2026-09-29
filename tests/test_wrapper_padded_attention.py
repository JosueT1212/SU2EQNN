"""The wrappers batch jets as padded (B, L_max) blocks instead of one dense
(n_total, n_total) masked attention. Logits must match the dense reference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.utils import scatter

from experiments.top_tagging.embedding import build_embedding
from experiments.top_tagging.experiment import build_model
from experiments.top_tagging.wrapper import build_attention_mask


def jets(n_jets=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(3, 40, (n_jets,), generator=g)
    p = torch.randn(int(counts.sum()), 3, generator=g, dtype=torch.float64) * 30
    p[:, 2] += 200
    x = torch.cat([p.norm(dim=1, keepdim=True), p], dim=1)
    return x, torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


def dense_reference(model, emb, **extra):
    """The previous implementation: one (n_total, n_total) block-diagonal mask."""
    mv = emb["mv"].unsqueeze(0)
    mask = build_attention_mask(emb["batch"], mv.dtype)
    out, _ = model.net(mv, scalars=None, attention_mask=mask, **extra)
    n_jets = int(emb["batch"].max()) + 1
    return scatter(out[0, :, :, 0], emb["batch"], dim=0, dim_size=n_jets, reduce="mean")


@pytest.mark.parametrize("multi_query", [True, False])
@pytest.mark.parametrize("frame,beams", [("lab", "lab"), ("rest", "none"), ("rest", "boosted")])
def test_padded_matches_dense_mask(frame, beams, multi_query):
    c = SimpleNamespace(
        data=SimpleNamespace(embedding="multi_channel", inject=[], two_beams=True,
                             rescale_data=True, frame=frame, beams=beams),
        model=SimpleNamespace(hidden_mv_channels=8, num_blocks=3, num_heads=4, head_scale=True,
                              multi_query=multi_query),
    )
    torch.manual_seed(0)
    model = build_model(c).double().eval()
    x, ptr = jets()
    emb = build_embedding(c, x, ptr, {})
    with torch.no_grad():
        assert torch.allclose(model(emb), dense_reference(model, emb), rtol=1e-10, atol=1e-12)


def test_padded_patch_wrapper_matches_dense_mask():
    c = SimpleNamespace(
        data=SimpleNamespace(embedding="patch", inject=[], two_beams=True, rescale_data=True),
        model=SimpleNamespace(hidden_mv_channels=8, num_blocks=3, num_heads=4,
                              patch_mv_channels=4),
    )
    torch.manual_seed(0)
    model = build_model(c).double().eval()
    x, ptr = jets()
    axes = torch.nn.functional.normalize(torch.randn(len(ptr) - 1, 4, 3, dtype=torch.float64), dim=-1)
    emb = build_embedding(c, x, ptr, {"axes": axes})
    with torch.no_grad():
        ref = dense_reference(model, emb, patch_mv=emb["patch_mv"].unsqueeze(0))
        assert torch.allclose(model(emb), ref, rtol=1e-10, atol=1e-12)
