"""Spin(3)-equivariant geometric attention primitive for Cl(3,0).

Ports L-GATr's sdp_attention to Cl(3,0):
  - MV dim: 8 (not 16)
  - IP factors: all +1 (Euclidean, not Minkowski)
  - Score: dot(q_mv_flat * ones, k_mv_flat) + dot(q_s, k_s) = Spin(3)-invariant
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

from su2eqnn.primitives.invariants import _load_inner_product_factors

_MV_DIM = 8


def sdp_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Optional[Tensor],
    k_s: Optional[Tensor],
    v_s: Optional[Tensor],
    attn_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Equivariant geometric attention for Cl(3,0).

    Score: Cl(3,0) inner product over channels + Euclidean scalar dot product.
    Both terms are Spin(3)-invariant -> attention weights are invariant ->
    weighted sum of equivariant values is equivariant.

    Parameters
    ----------
    q_mv : Tensor with shape (..., items, C, 8)
    k_mv : Tensor with shape (..., items, C, 8)
    v_mv : Tensor with shape (..., items, C, 8)
    q_s  : Tensor with shape (..., items, Ds) or None
    k_s  : Tensor with shape (..., items, Ds) or None
    v_s  : Tensor with shape (..., items, Ds) or None
    attn_mask : optional Tensor with shape (..., items, items)

    Returns
    -------
    out_mv : Tensor with shape (..., items, C, 8)
    out_s  : Tensor with shape (..., items, Ds) or None
    """
    ip_factors = _load_inner_product_factors(device=q_mv.device, dtype=q_mv.dtype)
    num_mv_channels = v_mv.shape[-2]

    # Flatten MV channels: (..., items, C, 8) -> (..., items, C*8)
    q_mv_flat = (q_mv * ip_factors).flatten(start_dim=-2)
    k_mv_flat = k_mv.flatten(start_dim=-2)
    v_mv_flat = v_mv.flatten(start_dim=-2)

    if q_s is not None:
        q = torch.cat([q_mv_flat, q_s], dim=-1)
        k = torch.cat([k_mv_flat, k_s], dim=-1)
        v = torch.cat([v_mv_flat, v_s], dim=-1)
    else:
        q = q_mv_flat
        k = k_mv_flat
        v = v_mv_flat

    out = torch_sdpa(q, k, v, attn_mask=attn_mask)

    out_mv = out[..., : num_mv_channels * _MV_DIM].reshape(
        *out.shape[:-1], num_mv_channels, _MV_DIM
    )
    out_s = out[..., num_mv_channels * _MV_DIM :] if q_s is not None else None

    return out_mv, out_s
