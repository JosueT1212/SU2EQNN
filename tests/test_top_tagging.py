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
