"""Spin(3)-equivariant gated nonlinearities for Cl(3,0) multivectors.

Gate is always a Spin(3)-invariant scalar; value is a multivector.
scalar × multivector = equivariant (Schur / homogeneity argument).

These functions are grade-agnostic: they operate on the last (8-dim) axis
and require only that the gate is invariant under the group action.
"""

import torch


def gated_relu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Equivariant gated ReLU: ReLU(gates) * x.

    Parameters
    ----------
    x     : torch.Tensor with shape (..., 8)
    gates : torch.Tensor with shape (..., 1)  — Spin(3)-invariant

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    return torch.nn.functional.relu(gates) * x


def gated_sigmoid(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Equivariant gated sigmoid: sigmoid(gates) * x.

    Parameters
    ----------
    x     : torch.Tensor with shape (..., 8)
    gates : torch.Tensor with shape (..., 1)  — Spin(3)-invariant

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    return torch.nn.functional.sigmoid(gates) * x


def gated_gelu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Equivariant gated GeLU: GeLU(gates) * x.

    Parameters
    ----------
    x     : torch.Tensor with shape (..., 8)
    gates : torch.Tensor with shape (..., 1)  — Spin(3)-invariant

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    return torch.nn.functional.gelu(gates, approximate="tanh") * x
