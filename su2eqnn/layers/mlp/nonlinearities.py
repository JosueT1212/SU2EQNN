"""Scalar-gated nonlinearity for Cl(3,0) dual-stream (MV + scalar)."""

from typing import Tuple

import torch
from torch import nn

from su2eqnn.primitives.nonlinearities import gated_gelu, gated_relu, gated_sigmoid


class ScalarGatedNonlinearity(nn.Module):
    """Gated nonlinearity using grade-0 (scalar) component as gate.

    For MVs: gate = x[..., [0]] (grade-0, Spin(3)-invariant).
    Computes f(gate) * x — invariant scalar times equivariant MV = equivariant.

    For scalars: applies f pointwise (no gating needed).

    Parameters
    ----------
    nonlinearity : {"relu", "sigmoid", "gelu"}
    """

    def __init__(self, nonlinearity: str = "gelu") -> None:
        super().__init__()
        _gated = {"relu": gated_relu, "gelu": gated_gelu, "sigmoid": gated_sigmoid}
        _scalar = {
            "relu": nn.functional.relu,
            "gelu": nn.functional.gelu,
            "sigmoid": nn.functional.sigmoid,
        }
        if nonlinearity not in _gated:
            raise ValueError(f"Unknown nonlinearity '{nonlinearity}', options: {list(_gated)}")
        self.gated_fn = _gated[nonlinearity]
        self.scalar_fn = _scalar[nonlinearity]

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., C, 8)
        scalars      : torch.Tensor with shape (..., S)

        Returns
        -------
        out_mv : torch.Tensor with shape (..., C, 8)
        out_s  : torch.Tensor with shape (..., S)
        """
        gates = multivectors[..., [0]]              # (..., C, 1) — grade-0, invariant
        out_mv = self.gated_fn(multivectors, gates)
        out_s = self.scalar_fn(scalars)
        return out_mv, out_s
