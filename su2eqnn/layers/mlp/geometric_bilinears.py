"""Spin(3)-equivariant geometric bilinear layer for Cl(3,0)."""

from typing import Optional, Tuple

import torch
from torch import nn

from su2eqnn.layers.layer_norm import EquiLayerNorm
from su2eqnn.layers.linear import EquiLinear
from su2eqnn.primitives.bilinear import geometric_product


class GeometricBilinear(nn.Module):
    """Geometric bilinear layer: projects to hidden, takes geometric product, projects out.

    The geometric product is the only Spin(3)-equivariant nonlinear operation:
      GP(ρ(w)(x), ρ(w)(y)) = ρ(w)(GP(x, y))

    Architecture:
      left  = linear_left(x, s)          # (..., hidden, 8)  standard init
      right = linear_right(x, s)         # (..., hidden, 8)  almost_unit_scalar init
      gp    = geometric_product(left, right)
      out_mv, out_s = linear_out(gp, s)
      out_mv = EquiLayerNorm(out_mv)

    linear_right uses almost_unit_scalar init so the GP starts near the identity
    map, stabilizing early training.

    Parameters
    ----------
    in_mv_channels : int
    out_mv_channels : int
    hidden_mv_channels : int or None
        Defaults to out_mv_channels.
    in_s_channels : int or None
    out_s_channels : int or None
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: Optional[int] = None,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if hidden_mv_channels is None:
            hidden_mv_channels = out_mv_channels

        self.linear_left = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=None,
        )
        # almost_unit_scalar requires in_s_channels; fall back to default when no scalars
        right_init = "almost_unit_scalar" if in_s_channels is not None else "default"
        self.linear_right = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=None,
            initialization=right_init,
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
        )
        self.norm = EquiLayerNorm()

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 8)
        scalars      : torch.Tensor with shape (..., in_s_channels) or None

        Returns
        -------
        out_mv : torch.Tensor with shape (..., out_mv_channels, 8)
        out_s  : torch.Tensor with shape (..., out_s_channels) or None
        """
        left, _  = self.linear_left(multivectors, scalars=scalars)
        right, _ = self.linear_right(multivectors, scalars=scalars)
        gp = geometric_product(left, right)
        out_mv, out_s = self.linear_out(gp, scalars=scalars)
        out_mv = self.norm(out_mv)
        return out_mv, out_s
