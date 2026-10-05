"""Per-constituent subjet_idx: masked like x, collated along dim 0, not incremented."""

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from experiments.top_tagging.dataset import TopTaggingDataset
from experiments.top_tagging.experiment import collect_aux


def test_subjet_idx_masked_and_collated(tmp_path):
    kin = np.zeros((2, 200, 4), dtype=np.float32)
    kin[0, :4] = [[10, 1, 0, 9], [8, 0, 1, 7], [6, 1, 1, 5], [5, -1, 0, 4]]
    kin[1, :3] = [[9, 1, 0, 8], [7, 0, -1, 6], [4, 1, 1, 3]]
    idx = np.full((2, 200), -1, dtype=np.int8)
    idx[0, :4] = [0, 1, 2, 0]
    idx[1, :3] = [2, 0, 1]
    path = tmp_path / "d.npz"
    np.savez(path, kinematics_test=kin, labels_test=np.array([True, False]), subjet_idx_test=idx)

    ds = TopTaggingDataset(rescale_data=True)
    ds.load_data(str(path), "test")
    assert ds[0].subjet_idx.dtype == torch.long
    assert ds[0].subjet_idx.tolist() == [0, 1, 2, 0]

    batch = next(iter(DataLoader(ds.data_list, batch_size=2, shuffle=False)))
    assert batch.subjet_idx.tolist() == [0, 1, 2, 0, 2, 0, 1]   # not incremented
    assert batch.subjet_idx.shape[0] == batch.x.shape[0]
    assert torch.equal(collect_aux(batch)["subjet_idx"], batch.subjet_idx)


def test_no_subjet_idx_key_means_no_attribute(tmp_path):
    kin = np.zeros((1, 200, 4), dtype=np.float32)
    kin[0, :2] = [[10, 1, 0, 9], [8, 0, 1, 7]]
    path = tmp_path / "d.npz"
    np.savez(path, kinematics_test=kin, labels_test=np.array([True]))
    ds = TopTaggingDataset(rescale_data=True)
    ds.load_data(str(path), "test")
    assert getattr(ds[0], "subjet_idx", None) is None
