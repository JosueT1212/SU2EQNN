"""Spin(3)-equivariant layer normalization (nn.Module wrapper)."""

import torch
from torch import nn

from su2eqnn.primitives.normalization import equi_layer_norm


class EquiLayerNorm(nn.Module):
    """Equivariant LayerNorm for Cl(3,0) multivectors.

    Wraps equi_layer_norm with a learnable scalar gain parameter.

    Parameters
    ----------
    gain_init : float
        Initial value for the learnable gain (default 1.0).
    """

    def __init__(self, gain_init: float = 1.0) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain_init))

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
        return equi_layer_norm(x, channel_dim=channel_dim, gain=self.gain.item())
