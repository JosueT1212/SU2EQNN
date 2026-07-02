"""Spin(3)-equivariant Geometric MLP for Cl(3,0)."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from su2eqnn.layers.dropout import GradeDropout
from su2eqnn.layers.linear import EquiLinear
from su2eqnn.layers.mlp.config import MLPConfig
from su2eqnn.layers.mlp.geometric_bilinears import GeometricBilinear
from su2eqnn.layers.mlp.nonlinearities import ScalarGatedNonlinearity


class GeoMLP(nn.Module):
    """Geometric MLP for Cl(3,0) dual-stream (MV + scalar).

    First layer: GeometricBilinear (GP-based equivariant nonlinearity).
    Subsequent layers: ScalarGatedNonlinearity → EquiLinear.
    Optional GradeDropout after first layer.

    Parameters
    ----------
    config : MLPConfig
        mv_channels : [in, h1, ..., out] — at least 2 entries
        s_channels  : [in, h1, ..., out] or None — same length as mv_channels
        activation  : gated activation for nonlinear layers
        dropout_prob : dropout probability after GeometricBilinear, or None
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        assert config.mv_channels is not None and len(config.mv_channels) >= 2, \
            "mv_channels must have at least 2 entries (in, out)"

        s_channels: List[Optional[int]] = (
            [None] * len(config.mv_channels)
            if config.s_channels is None
            else list(config.s_channels)
        )

        layers: List[nn.Module] = []

        # First layer: GeometricBilinear (equivariant nonlinearity via GP)
        layers.append(GeometricBilinear(
            in_mv_channels=config.mv_channels[0],
            out_mv_channels=config.mv_channels[1],
            in_s_channels=s_channels[0],
            out_s_channels=s_channels[1],
        ))

        # Optional dropout after first layer
        if config.dropout_prob is not None:
            layers.append(GradeDropout(config.dropout_prob))

        # Subsequent layers: gated activation → EquiLinear
        for in_mv, out_mv, in_s, out_s in zip(
            config.mv_channels[1:-1],
            config.mv_channels[2:],
            s_channels[1:-1],
            s_channels[2:],
        ):
            layers.append(ScalarGatedNonlinearity(config.activation))
            layers.append(EquiLinear(in_mv, out_mv, in_s_channels=in_s, out_s_channels=out_s))

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., mv_channels[0], 8)
        scalars      : torch.Tensor with shape (..., s_channels[0]) or None

        Returns
        -------
        out_mv : torch.Tensor with shape (..., mv_channels[-1], 8)
        out_s  : torch.Tensor with shape (..., s_channels[-1]) or None
        """
        mv, s = multivectors, scalars
        for layer in self.layers:
            mv, s = layer(mv, s)
        return mv, s
