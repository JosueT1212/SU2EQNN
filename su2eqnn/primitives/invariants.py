"""Spin(3) ≅ SU(2) invariants for Cl(3,0) multivectors.

Cl(3,0) has a positive-definite inner product: all grade-k norms are positive.
Proof: for every basis element b, <b,b> = <~b * b>_0 = +1 (reversal sign
cancels the sign from e_ij^2 = -1 for grade-2 blades).

Inner product factors: all +1 (unlike Cl(1,3) which has mixed signs from
the Minkowski metric).
"""

from functools import lru_cache

import torch

from su2eqnn.primitives.linear import grade_project
from su2eqnn.utils.einsum import cached_einsum
from su2eqnn.utils.misc import minimum_autocast_precision

# Grade dimensions for Cl(3,0): comb(3, k) for k=0,1,2,3
_GRADE_DIMS = [1, 3, 3, 1]
_N_GRADES = 4
_MV_DIM = 8


@lru_cache()
def _load_inner_product_factors(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Diagonal of the Cl(3,0) inner product metric.

    All +1: the algebra is positive definite (Euclidean signature).

    Returns
    -------
    ip_factors : torch.Tensor with shape (8,)
    """
    factors = torch.ones(_MV_DIM, dtype=torch.float32, device=torch.device("cpu"))
    return factors.to(device=device, dtype=dtype)


@lru_cache()
def _load_metric_grades(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Metric tensor per grade for Cl(3,0).

    Returns
    -------
    m_grades : torch.Tensor of shape (4, 8)
        m_grades[k, i] = inner-product factor of component i if it belongs to grade k, else 0.
    """
    m = _load_inner_product_factors(torch.device("cpu"), torch.float32)
    m_grades = torch.zeros(_N_GRADES, _MV_DIM, dtype=torch.float32, device=torch.device("cpu"))
    offset = 0
    for k, d in enumerate(_GRADE_DIMS):
        m_grades[k, offset:offset + d] = m[offset:offset + d]
        offset += d
    return m_grades.to(device=device, dtype=dtype)


def inner_product(
    x: torch.Tensor, y: torch.Tensor, channel_sum: bool = False
) -> torch.Tensor:
    """Cl(3,0) inner product <x, y> = <~x y>_0 (simplified: all factors = +1).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8) or (..., channels, 8)
    y : torch.Tensor with shape (..., 8) or (..., channels, 8)
    channel_sum : bool
        Whether to also sum over the channel dimension.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
    """
    x = x * _load_inner_product_factors(device=x.device, dtype=x.dtype)

    if channel_sum:
        outputs = cached_einsum("... c i, ... c i -> ...", x, y)
    else:
        outputs = cached_einsum("... i, ... i -> ...", x, y)

    return outputs.unsqueeze(-1)


def squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Squared Cl(3,0) norm ||x||^2 = <x, x>.

    Positive semidefinite (Cl(3,0) is positive definite, but channels can cancel).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
    """
    return inner_product(x, x)


def spin3_invariants(x: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """Computes 4 Spin(3)-invariants: scalar component + grade norms.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)
    epsilon : float
        Regularization for near-zero norms.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 4)
        [scalar_component, ||grade-1||, ||grade-2||, ||grade-3||]
    """
    projections = grade_project(x)                       # (..., 4, 8)
    squared_norms = inner_product(projections, projections)[..., 0]  # (..., 4)
    norms = torch.sqrt(torch.clamp(squared_norms, epsilon))
    return torch.cat((x[..., [0]], norms[..., 1:]), dim=-1)   # (..., 4)


@minimum_autocast_precision(torch.float32)
def abs_squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Positive-semidefinite norm for layer normalization.

    Sums |grade-k norm|^2 over all grades, taking absolute value so that
    the result is always non-negative.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
    """
    m = _load_metric_grades(device=x.device, dtype=x.dtype)
    abs_squared_norms = (
        cached_einsum("... i, ... i, g i -> ... g", x, x, m).abs().sum(-1, keepdim=True)
    )
    return abs_squared_norms
