"""Tests for top tagging experiment components."""

import pytest
import torch
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "toptagging_mini.npz"

# ---------------------------------------------------------------------------
# Task 2: Spurions
# ---------------------------------------------------------------------------

from su2eqnn.interface.spurions import embed_spurions, get_num_spurions


def test_get_num_spurions_two_beams():
    assert get_num_spurions(two_beams=True) == 2


def test_get_num_spurions_one_beam():
    assert get_num_spurions(two_beams=False) == 1


def test_embed_spurions_shape_two_beams():
    s = embed_spurions(two_beams=True)
    assert s.shape == (2, 1, 8)


def test_embed_spurions_shape_one_beam():
    s = embed_spurions(two_beams=False)
    assert s.shape == (1, 1, 8)


def test_embed_spurions_pz_values():
    s = embed_spurions(two_beams=True)
    # index 3 = e3 = pz direction (grade-1)
    assert s[0, 0, 3].item() == pytest.approx(1.0)   # +z beam
    assert s[1, 0, 3].item() == pytest.approx(-1.0)  # -z beam


def test_embed_spurions_only_pz_nonzero():
    s = embed_spurions(two_beams=True)
    # all components except index 3 must be zero
    mask = torch.ones(8, dtype=torch.bool)
    mask[3] = False
    assert (s[:, :, mask] == 0).all()


def test_embed_spurions_dtype():
    s = embed_spurions(dtype=torch.float64)
    assert s.dtype == torch.float64


def test_embed_spurions_device():
    s = embed_spurions(device=torch.device("cpu"))
    assert s.device.type == "cpu"


# ---------------------------------------------------------------------------
# Task 3: Dataset
# ---------------------------------------------------------------------------

from experiments.top_tagging.dataset import TopTaggingDataset, EPS


def test_dataset_train_length():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    assert len(ds) == 20000


def test_dataset_test_length():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "test")
    assert len(ds) == 2000


def test_dataset_item_shape():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    item = ds[0]
    assert item.x.ndim == 2
    assert item.x.shape[1] == 4
    assert 1 <= item.x.shape[0] <= 200


def test_dataset_label_dtype():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    assert ds[0].label.dtype == torch.bool


def test_dataset_drops_padding():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "test")
    for item in ds.data_list[:20]:
        # no all-zero rows — any component abs > EPS
        assert (item.x.abs() > EPS).any(dim=-1).all(), \
            "Zero-padded row not filtered"


def test_dataset_fourmomenta_dtype():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    assert ds[0].x.dtype == torch.float32


def test_dataset_loads_axes(tmp_path):
    """Data objects carry per-jet axes (1,4,3) when npz contains axes_{mode}."""
    import numpy as np
    import torch
    from experiments.top_tagging.dataset import TopTaggingDataset

    n_jets = 3
    kin = np.zeros((n_jets, 10, 4), dtype=np.float32)
    kin[:, 0] = [10.0, 1.0, 0.0, 9.0]
    kin[:, 1] = [10.0, -1.0, 0.5, 9.0]
    labels = np.array([True, False, True])
    axes = np.tile(np.eye(3, dtype=np.float32)[None, :3], (n_jets, 1, 1))
    axes = np.concatenate([axes, axes[:, :1]], axis=1)   # (n_jets, 4, 3)

    path = tmp_path / "mini_axes.npz"
    np.savez(path, kinematics_train=kin, labels_train=labels, axes_train=axes)

    ds = TopTaggingDataset(rescale_data=True)
    ds.load_data(str(path), "train")
    d = ds[0]
    assert hasattr(d, "axes")
    assert d.axes.shape == (1, 4, 3)
    assert torch.allclose(d.axes[0], torch.tensor(axes[0]))


def test_dataset_loads_p_lead(tmp_path):
    """Data objects carry per-jet p_lead (1,4) when npz contains p_lead_{mode}."""
    import numpy as np
    import torch
    from experiments.top_tagging.dataset import TopTaggingDataset

    n_jets = 3
    kin = np.zeros((n_jets, 10, 4), dtype=np.float32)
    kin[:, 0] = [10.0, 1.0, 0.0, 9.0]
    kin[:, 1] = [10.0, -1.0, 0.5, 9.0]
    labels = np.array([True, False, True])
    p_lead = np.tile(np.array([50.0, 10.0, 0.0, 3.0], dtype=np.float32), (n_jets, 1))

    path = tmp_path / "mini_leading.npz"
    np.savez(path, kinematics_train=kin, labels_train=labels, p_lead_train=p_lead)

    ds = TopTaggingDataset(rescale_data=True)
    ds.load_data(str(path), "train")
    d = ds[0]
    assert hasattr(d, "p_lead")
    assert d.p_lead.shape == (1, 4)
    assert torch.allclose(d.p_lead[0], torch.tensor(p_lead[0]))


# ---------------------------------------------------------------------------
# Task 4: Embedding
# ---------------------------------------------------------------------------

from experiments.top_tagging.embedding import embed_tagging_data, get_batch_from_ptr, UNITS


def test_get_batch_from_ptr():
    ptr = torch.tensor([0, 3, 5])
    batch = get_batch_from_ptr(ptr)
    expected = torch.tensor([0, 0, 0, 1, 1])
    assert (batch == expected).all()


def test_embed_tagging_data_total_tokens():
    # 2 jets: jet0=3 particles, jet1=2 particles; two_beams=True → 2 beams per jet
    fourmomenta = torch.randn(5, 4)
    ptr = torch.tensor([0, 3, 5])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    # 5 particles + 2*2 beam tokens = 9 total
    assert result["mv"].shape == (9, 1, 8)
    assert result["batch"].shape == (9,)
    assert result["is_beam"].shape == (9,)


def test_embed_tagging_data_beam_count():
    fourmomenta = torch.randn(5, 4)
    ptr = torch.tensor([0, 3, 5])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    # 2 jets * 2 beam tokens = 4 beam tokens
    assert result["is_beam"].sum().item() == 4


def test_embed_tagging_data_beam_values():
    # 1 jet, 3 particles
    fourmomenta = torch.zeros(3, 4)
    ptr = torch.tensor([0, 3])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    beam_mvs = result["mv"][result["is_beam"]]  # (2, 1, 8)
    assert beam_mvs[0, 0, 3].item() == pytest.approx(1.0)   # +z
    assert beam_mvs[1, 0, 3].item() == pytest.approx(-1.0)  # -z


def test_embed_tagging_data_4momentum_placement():
    # E=10, px=1, py=2, pz=3
    fourmomenta = torch.tensor([[10.0, 1.0, 2.0, 3.0]])
    ptr = torch.tensor([0, 1])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=False, rescale=False)
    particle_mv = result["mv"][~result["is_beam"]][0, 0]  # (8,)
    assert particle_mv[0].item() == pytest.approx(10.0)  # E → grade-0
    assert particle_mv[1].item() == pytest.approx(1.0)   # px → e1
    assert particle_mv[2].item() == pytest.approx(2.0)   # py → e2
    assert particle_mv[3].item() == pytest.approx(3.0)   # pz → e3
    assert (particle_mv[4:] == 0).all()


def test_embed_tagging_data_rescale():
    fourmomenta = torch.tensor([[20.0, 20.0, 20.0, 20.0]])
    ptr = torch.tensor([0, 1])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=False, rescale=True)
    particle_mv = result["mv"][~result["is_beam"]][0, 0]  # (8,)
    assert particle_mv[0].item() == pytest.approx(1.0)  # E/20 = 1.0
    assert particle_mv[1].item() == pytest.approx(1.0)  # px/20 = 1.0


def test_embed_tagging_data_beam_prepended():
    # Verify beam tokens come first in each jet's subsequence
    fourmomenta = torch.zeros(3, 4)  # 1 jet, 3 particles
    ptr = torch.tensor([0, 3])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    # Output order: [beam+z, beam-z, p0, p1, p2]
    assert result["is_beam"][0].item() is True  # first token is beam
    assert result["is_beam"][1].item() is True  # second token is beam
    assert not result["is_beam"][2].item()       # third token is particle


def test_embed_tagging_data_batch_indices():
    # 2 jets: 3 particles + 2 beams, 2 particles + 2 beams → jet0 has 5 tokens, jet1 has 4 tokens
    fourmomenta = torch.zeros(5, 4)
    ptr = torch.tensor([0, 3, 5])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    batch = result["batch"]
    assert (batch[:5] == 0).all()  # first 5 tokens belong to jet 0
    assert (batch[5:] == 1).all()  # last 4 tokens belong to jet 1


# ---------------------------------------------------------------------------
# Task 5: Wrapper
# ---------------------------------------------------------------------------

from experiments.top_tagging.wrapper import build_attention_mask, TopTaggingWrapper
from su2eqnn.nets.gatr import GATr
from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.mlp.config import MLPConfig


def _make_tiny_net():
    return GATr(
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=4,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=2),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=2,
    )


def test_build_attention_mask_shape():
    batch = torch.tensor([0, 0, 0, 1, 1])
    mask = build_attention_mask(batch, torch.float32)
    assert mask.shape == (5, 5)


def test_build_attention_mask_within_jet_zero():
    batch = torch.tensor([0, 0, 0, 1, 1])
    mask = build_attention_mask(batch, torch.float32)
    # within jet 0: rows/cols 0-2
    assert (mask[:3, :3] == 0.0).all()
    # within jet 1: rows/cols 3-4
    assert (mask[3:, 3:] == 0.0).all()


def test_build_attention_mask_cross_jet_neginf():
    batch = torch.tensor([0, 0, 0, 1, 1])
    mask = build_attention_mask(batch, torch.float32)
    assert (mask[:3, 3:] == float("-inf")).all()
    assert (mask[3:, :3] == float("-inf")).all()


def test_wrapper_output_shape():
    net = _make_tiny_net()
    wrapper = TopTaggingWrapper(net)
    # 2 jets: jet0 has 3 tokens, jet1 has 2 tokens
    mv = torch.randn(5, 1, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])
    is_beam = torch.zeros(5, dtype=torch.bool)
    logits = wrapper({"mv": mv, "batch": batch, "is_beam": is_beam})
    assert logits.shape == (2, 1)


def test_wrapper_output_dtype():
    net = _make_tiny_net()
    wrapper = TopTaggingWrapper(net)
    mv = torch.randn(5, 1, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])
    is_beam = torch.zeros(5, dtype=torch.bool)
    logits = wrapper({"mv": mv, "batch": batch, "is_beam": is_beam})
    assert logits.dtype == torch.float32


def test_wrapper_gradient_flows():
    net = _make_tiny_net()
    wrapper = TopTaggingWrapper(net)
    mv = torch.randn(5, 1, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])
    is_beam = torch.zeros(5, dtype=torch.bool)
    logits = wrapper({"mv": mv, "batch": batch, "is_beam": is_beam})
    loss = logits.sum()
    loss.backward()
    # at least one parameter should have gradient
    has_grad = any(p.grad is not None for p in net.parameters())
    assert has_grad


# ---------------------------------------------------------------------------
# Task 6: Experiment — smoke test
# ---------------------------------------------------------------------------

from experiments.top_tagging.experiment import build_model
from torch_geometric.loader import DataLoader as GeometricDataLoader
from experiments.top_tagging.embedding import embed_tagging_data


class _FakeCfg:
    """Minimal config object for testing."""
    class model:
        hidden_mv_channels = 4
        num_blocks = 2
        num_heads = 2
    class data:
        rescale_data = True
        two_beams = True
    class training:
        lr = 1e-3


def test_build_model_returns_wrapper():
    model = build_model(_FakeCfg())
    assert isinstance(model, TopTaggingWrapper)


def test_smoke_one_training_step():
    """One forward + backward through the full pipeline on real data."""
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")

    loader = GeometricDataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = build_model(_FakeCfg())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    embedding = embed_tagging_data(
        batch.x, batch.ptr,
        two_beams=True, rescale=True,
    )
    logits = model(embedding)                              # (4, 1)
    targets = batch.label.float().unsqueeze(1)            # (4, 1)
    loss = criterion(logits, targets)

    assert loss.ndim == 0             # scalar
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # parameters changed after optimizer step
    p0 = next(model.parameters())
    assert p0.grad is not None


def test_build_embedding_dispatches_to_r8():
    """cfg.data.embedding == 'single_channel_axis' routes through embed_tagging_data_single_channel_axis."""
    from experiments.top_tagging.experiment import build_embedding
    from torch_geometric.data import Data, Batch

    class _R8Cfg:
        class data:
            two_beams = True
            rescale_data = False
            embedding = "single_channel_axis"

    fourmomenta = torch.randn(4, 4)
    axes = torch.zeros(1, 4, 3)
    axes[0, :, 2] = 1.0
    p_lead = torch.randn(1, 4)
    d = Data(x=fourmomenta, ptr=torch.tensor([0, 4]))
    d.axes = axes
    d.p_lead = p_lead
    batch = Batch.from_data_list([d])

    embedding = build_embedding(batch, _R8Cfg())
    assert embedding["mv"].shape[1] == 1, "single-channel jet-axis embedding must be single-channel"


def test_build_embedding_default_unchanged():
    """Without cfg.data.embedding set, falls back to plain embed_tagging_data."""
    from experiments.top_tagging.experiment import build_embedding

    class _DefaultCfg:
        class data:
            two_beams = True
            rescale_data = False

    fourmomenta = torch.randn(4, 4)
    from torch_geometric.data import Data, Batch
    axes = torch.zeros(1, 4, 3)
    axes[0, :, 2] = 1.0
    d = Data(x=fourmomenta, ptr=torch.tensor([0, 4]))
    d.axes = axes
    batch = Batch.from_data_list([d])

    embedding = build_embedding(batch, _DefaultCfg())
    assert embedding["mv"].shape[1] == 5   # axes present -> multi-axis bivector layout, not single-channel


def test_warm_restart_scheduler_resets_lr():
    """CosineAnnealingWarmRestarts: LR decays within a cycle, resets at T_0."""
    import torch

    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1.0e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=1, eta_min=1.0e-5
    )
    lrs = []
    for _ in range(20):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[8] < 1.0e-4, f"LR should be near eta_min before restart, got {lrs[8]}"
    assert lrs[9] > 9.0e-4, f"LR should reset to ~initial at T_0, got {lrs[9]}"


# ---------------------------------------------------------------------------
# Task 4: single-channel jet-axis embedding
# ---------------------------------------------------------------------------

from experiments.top_tagging.embedding import embed_tagging_data_single_channel_axis


def test_embed_tagging_data_single_channel_axis_shape_and_momentum():
    fourmomenta = torch.tensor([
        [40.0, 3.0, 0.0, 4.0],
        [10.0, 0.0, 1.0, 2.0],
        [20.0, 1.0, 1.0, 5.0],
    ])
    ptr = torch.tensor([0, 2, 3])
    jet_axis = torch.zeros(2, 3)
    jet_axis[:, 2] = 1.0   # +z for both jets

    out = embed_tagging_data_single_channel_axis(fourmomenta, ptr, jet_axis, two_beams=True, rescale=True, p_lead=None)
    mv = out["mv"]
    assert mv.shape[1] == 1, f"expected 1 channel, got {mv.shape[1]}"

    particles = mv[~out["is_beam"]]
    p = fourmomenta / UNITS
    assert torch.allclose(particles[:, 0, 0], p[:, 0])
    assert torch.allclose(particles[:, 0, 1:4], p[:, 1:4])


def test_embed_tagging_data_single_channel_axis_bivector_matches_r4_formula():
    """With n=+z: e12(idx4)=0, e13(idx5)=px, e23(idx6)=py — same as the multi-channel bivector's channel-1 formula."""
    fourmomenta = torch.tensor([
        [40.0, 3.0, 0.0, 4.0],
        [10.0, 0.0, 1.0, 2.0],
    ])
    ptr = torch.tensor([0, 2])
    jet_axis = torch.tensor([[0.0, 0.0, 1.0]])

    out = embed_tagging_data_single_channel_axis(fourmomenta, ptr, jet_axis, two_beams=False, rescale=False, p_lead=None)
    particles = out["mv"][~out["is_beam"]]
    assert particles[:, 0, 4].abs().max() < 1e-6                          # e12 = 0
    assert torch.allclose(particles[:, 0, 5], fourmomenta[:, 1], atol=1e-6)  # e13 = px
    assert torch.allclose(particles[:, 0, 6], fourmomenta[:, 2], atol=1e-6)  # e23 = py


def test_embed_tagging_data_single_channel_axis_variant_a_pseudoscalar_always_zero():
    """Variant (a): p_lead=None -> pseudoscalar slot (index 7) is always 0."""
    fourmomenta = torch.randn(5, 4)
    ptr = torch.tensor([0, 5])
    jet_axis = torch.tensor([[0.3, 0.1, 0.9]])
    jet_axis = jet_axis / jet_axis.norm(dim=-1, keepdim=True)

    out = embed_tagging_data_single_channel_axis(fourmomenta, ptr, jet_axis, two_beams=False, rescale=False, p_lead=None)
    particles = out["mv"][~out["is_beam"]]
    assert particles[:, 0, 7].abs().max() < 1e-6


def test_embed_tagging_data_single_channel_axis_variant_b_leading_particle_pseudoscalar_is_zero():
    """Variant (b): the leading particle's own I is exactly 0 (a . (b x a) = 0 identity)."""
    fourmomenta = torch.tensor([
        [50.0, 10.0, 0.0, 0.0],   # leading (highest pT = 10)
        [20.0, 2.0, 3.0, 1.0],
        [15.0, 1.0, -2.0, 4.0],
    ])
    ptr = torch.tensor([0, 3])
    total_p = fourmomenta[:, 1:4].sum(dim=0, keepdim=True)
    jet_axis = total_p / total_p.norm(dim=-1, keepdim=True)
    p_lead = fourmomenta[0:1]   # (1, 4), the leading row itself

    out = embed_tagging_data_single_channel_axis(
        fourmomenta, ptr, jet_axis, two_beams=False, rescale=False, p_lead=p_lead
    )
    particles = out["mv"][~out["is_beam"]]
    assert particles[0, 0, 7].abs().item() < 1e-5, "leading particle's I must be ~0"
    assert particles[1, 0, 7].abs().item() > 1e-5, "non-leading particle's I should generally be nonzero"


def test_embed_tagging_data_single_channel_axis_variant_b_pseudoscalar_zero_when_axes_parallel():
    """Degenerate 1-prong limit: n_jet parallel to p_lead -> I=0 for every particle."""
    fourmomenta = torch.tensor([
        [50.0, 10.0, 0.0, 0.0],
        [20.0, 2.0, 3.0, 1.0],
    ])
    ptr = torch.tensor([0, 2])
    jet_axis = torch.tensor([[1.0, 0.0, 0.0]])
    p_lead = torch.tensor([[50.0, 5.0, 0.0, 0.0]])   # parallel to jet_axis

    out = embed_tagging_data_single_channel_axis(
        fourmomenta, ptr, jet_axis, two_beams=False, rescale=False, p_lead=p_lead
    )
    particles = out["mv"][~out["is_beam"]]
    assert particles[:, 0, 7].abs().max() < 1e-6


def test_embed_tagging_data_single_channel_axis_variant_b_pseudoscalar_matches_determinant_reference():
    """I must equal det[p_i, n_jet, p_lead_hat] exactly — catches sign flips, swapped operands, missing normalization."""
    torch.manual_seed(0)
    fourmomenta = torch.randn(6, 4)
    ptr = torch.tensor([0, 3, 6])
    jet_axis = torch.randn(2, 3)
    jet_axis = jet_axis / jet_axis.norm(dim=-1, keepdim=True)
    p_lead = torch.randn(2, 4)

    out = embed_tagging_data_single_channel_axis(
        fourmomenta, ptr, jet_axis, two_beams=False, rescale=False, p_lead=p_lead
    )
    particles = out["mv"][~out["is_beam"]]
    computed_I = particles[:, 0, 7]

    particle_batch = get_batch_from_ptr(ptr)
    p = fourmomenta[:, 1:4]
    n = jet_axis[particle_batch]
    pl = p_lead[:, 1:4]
    pl_hat = (pl / pl.norm(dim=-1, keepdim=True))[particle_batch]

    ref_I = torch.stack([
        torch.linalg.det(torch.stack([p[i], n[i], pl_hat[i]]))
        for i in range(p.shape[0])
    ])

    assert torch.allclose(computed_I, ref_I, atol=1e-5), \
        f"pseudoscalar doesn't match determinant reference: max diff {(computed_I - ref_I).abs().max():.2e}"


def test_embed_tagging_data_single_channel_axis_beam_spurions_zero_beyond_vector():
    fourmomenta = torch.zeros(3, 4)
    ptr = torch.tensor([0, 3])
    jet_axis = torch.tensor([[0.0, 0.0, 1.0]])
    out = embed_tagging_data_single_channel_axis(fourmomenta, ptr, jet_axis, two_beams=True, rescale=False, p_lead=None)
    beams = out["mv"][out["is_beam"]]
    assert beams[:, 0, 4:].abs().max() < 1e-6
