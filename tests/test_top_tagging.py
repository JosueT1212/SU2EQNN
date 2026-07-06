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
