"""TopTagging dataset for binary jet classification.

Loads .npz files with format:
  kinematics_{mode}: shape (N, 200, 4) — E, px, py, pz per particle
  labels_{mode}:     shape (N,)        — bool, True = top quark jet
"""

import numpy as np
import torch
from torch_geometric.data import Data

EPS = 1e-5


class TopTaggingDataset(torch.utils.data.Dataset):
    """Binary jet classification dataset using torch_geometric Data objects.

    Parameters
    ----------
    rescale_data : bool
        Stored but unused here — rescaling is applied in embed_tagging_data.
    """

    def __init__(self, rescale_data: bool):
        self.rescale_data = rescale_data
        self.data_list = []

    def load_data(self, filename: str, mode: str, dtype=torch.float32):
        """Load jets from .npz file.

        Parameters
        ----------
        filename : str  path to .npz file
        mode     : {"train", "val", "test"}
        """
        raw = np.load(filename)
        kinematics = torch.tensor(raw[f"kinematics_{mode}"], dtype=dtype)
        labels = torch.tensor(raw[f"labels_{mode}"], dtype=torch.bool)
        axes_key = f"axes_{mode}"
        axes = (
            torch.tensor(raw[axes_key], dtype=dtype) if axes_key in raw.files else None
        )
        masses_key = f"masses_{mode}"
        masses = (
            torch.tensor(raw[masses_key], dtype=dtype) if masses_key in raw.files else None
        )

        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded particles: keep rows where any |component| > EPS
            mask = (kinematics[i].abs() > EPS).any(dim=-1)
            fourmomenta = kinematics[i][mask]   # (n_real, 4)
            label = labels[i]                    # scalar bool
            data = Data(x=fourmomenta, label=label)
            if axes is not None:
                data.axes = axes[i].unsqueeze(0)   # (1, 4, 3)
            if masses is not None:
                data.masses = masses[i].unsqueeze(0)   # (1, 4)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
