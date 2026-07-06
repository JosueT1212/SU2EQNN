"""Spin(3)-equivariant layer normalization (nn.Module wrapper)."""

import torch
from torch import nn

from su2eqnn.primitives.normalization import equi_layer_norm



class EquiLayerNorm(nn.Module):
    """Equivariant LayerNorm for Cl(3,0) multivectors.

    Rescales so that mean_channels ||x||^2 = 1 (Cl(3,0) norm).
    No learnable parameters — matches L-GATr's EquiLayerNorm design.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, channel_dim: int = -2) -> torch.Tensor:
        """Normalize multivectors equivariantly.

        Parameters
        ----------
        x           : torch.Tensor with shape (..., channels, 8)
        channel_dim : int

        Returns
        -------
        outputs : torch.Tensor with shape (..., channels, 8)
        """
        return equi_layer_norm(x, channel_dim=channel_dim)
