"""Spin(3)-equivariant self-attention layer."""

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from su2eqnn.layers.attention.attention import GeometricAttention
from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.attention.positional_encoding import ApplyRotaryPositionalEncoding
from su2eqnn.layers.attention.qkv import MultiQueryQKVModule, QKVModule
from su2eqnn.layers.linear import EquiLinear


class SelfAttention(nn.Module):
    """Spin(3)-equivariant self-attention.

    Computes multi-head attention over multivector and scalar streams.
    Score = Cl(3,0) inner product(Q_mv, K_mv) + dot(Q_s, K_s) — Spin(3)-invariant.
    Output weighted sum of equivariant values → equivariant.

    Parameters
    ----------
    config : SelfAttentionConfig
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()
        self.config = config

        self.qkv_module = (
            MultiQueryQKVModule(config) if config.multi_query else QKVModule(config)
        )

        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=(
                None if config.in_s_channels is None
                else config.hidden_s_channels * config.num_heads
            ),
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        if config.pos_encoding:
            self.pos_encoding: nn.Module = ApplyRotaryPositionalEncoding(
                num_channels=config.hidden_s_channels,
                item_dim=-2,
                base=config.pos_encoding_base,
            )
        else:
            self.pos_encoding = nn.Identity()

        self.attention = GeometricAttention(config)

        self.use_head_scale = config.head_scale
        if self.use_head_scale:
            self.head_scale = nn.Parameter(torch.ones(config.num_heads))

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : Tensor (..., items, in_mv_channels, 8)
        scalars      : Tensor (..., items, in_s_channels) or None
        additional_qk_features_mv : Tensor (..., items, add_mv, 8) or None
        additional_qk_features_s  : Tensor (..., items, add_s) or None
        attention_mask : Tensor (..., items, items) or None

        Returns
        -------
        out_mv : Tensor (..., items, out_mv_channels, 8)
        out_s  : Tensor (..., items, out_s_channels) or None
        """
        q_mv, k_mv, v_mv, q_s, k_s, v_s = self.qkv_module(
            multivectors, scalars, additional_qk_features_mv, additional_qk_features_s
        )

        # RoPE on scalar Q/K only (invariant under Spin(3))
        if q_s is not None:
            q_s = self.pos_encoding(q_s)
            k_s = self.pos_encoding(k_s)

        h_mv, h_s = self.attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask)

        if self.use_head_scale:
            # head_scale: (num_heads,) → broadcast over (..., heads, items, channels, 8)
            # h_mv shape: (..., heads, items, hidden, 8) — 5 trailing dims from heads onward
            scale = self.head_scale.view(
                *([1] * (h_mv.ndim - 4)), self.config.num_heads, 1, 1, 1
            )
            h_mv = h_mv * scale
            if h_s is not None:
                scale_s = self.head_scale.view(
                    *([1] * (h_s.ndim - 3)), self.config.num_heads, 1, 1
                )
                h_s = h_s * scale_s

        h_mv = rearrange(h_mv, "... heads items hidden x -> ... items (heads hidden) x")
        if h_s is not None:
            h_s = rearrange(h_s, "... heads items hidden -> ... items (heads hidden)")

        out_mv, out_s = self.out_linear(h_mv, h_s)
        return out_mv, out_s
