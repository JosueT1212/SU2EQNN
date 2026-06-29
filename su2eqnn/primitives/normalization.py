"""Spin(3)-equivariant layer normalization for Cl(3,0) multivectors."""

import torch

from su2eqnn.primitives.invariants import abs_squared_norm


def equi_layer_norm(
    x: torch.Tensor, channel_dim: int = -2, gain: float = 1.0, epsilon: float = 0.01
) -> torch.Tensor:
    """Equivariant LayerNorm for Cl(3,0) multivectors.

    Rescales so that mean_channels ||x||^2 = 1 where the norm is the Cl(3,0)
    positive-definite norm (all inner-product factors = +1).

    Parameters
    ----------
    x           : torch.Tensor with shape (..., channels, 8)
    channel_dim : int — dimension over which to average the norm (default: -2)
    gain        : float — target output scale
    epsilon     : float — prevents division by zero

    Returns
    -------
    outputs : torch.Tensor with shape (..., channels, 8)
    """
    abs_sq = abs_squared_norm(x)                                    # (..., channels, 1)
    abs_sq = torch.mean(abs_sq, dim=channel_dim, keepdim=True)      # (..., 1, 1)
    abs_sq = torch.clamp(abs_sq, epsilon)
    return gain * x / torch.sqrt(abs_sq)
