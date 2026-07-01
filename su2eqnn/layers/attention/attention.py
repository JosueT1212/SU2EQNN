"""Geometric attention wrapper for Cl(3,0)."""

from typing import Optional, Tuple

import torch
from torch import nn

from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.primitives.attention import sdp_attention


class GeometricAttention(nn.Module):
    """Spin(3)-equivariant geometric attention.

    Thin stateless wrapper around sdp_attention primitive.

    Parameters
    ----------
    config : SelfAttentionConfig
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

    def forward(
        self,
        q_mv: torch.Tensor,
        k_mv: torch.Tensor,
        v_mv: torch.Tensor,
        q_s: Optional[torch.Tensor],
        k_s: Optional[torch.Tensor],
        v_s: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        q_mv, k_mv, v_mv : Tensor (..., heads, items, C, 8)
        q_s, k_s, v_s    : Tensor (..., heads, items, Ds) or None
        attention_mask    : Tensor (..., items, items) or None

        Returns
        -------
        h_mv : Tensor (..., heads, items, C, 8)
        h_s  : Tensor (..., heads, items, Ds) or None
        """
        return sdp_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attn_mask=attention_mask)
