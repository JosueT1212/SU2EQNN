"""QKV projection modules for Spin(3)-equivariant multi-head attention."""

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.layer_norm import EquiLayerNorm
from su2eqnn.layers.linear import EquiLinear


class QKVModule(nn.Module):
    """Standard multi-head QKV projection (independent Q, K, V per head).

    Parameters
    ----------
    config : SelfAttentionConfig
    """

    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self.config = config
        total_qkv_mv = 3 * config.hidden_mv_channels * config.num_heads
        total_qkv_s = (
            None if config.in_s_channels is None
            else 3 * config.hidden_s_channels * config.num_heads
        )
        self.in_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=total_qkv_mv,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels
            if config.in_s_channels is not None else None,
            out_s_channels=total_qkv_s,
        )
        self.norm_mv = EquiLayerNorm()
        self.norm_s = nn.LayerNorm(config.hidden_s_channels) if config.in_s_channels is not None else None

    def forward(
        self,
        inputs: torch.Tensor,
        scalars: Optional[torch.Tensor],
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor (..., items, in_mv_channels, 8)
        scalars : Tensor (..., items, in_s_channels) or None
        additional_qk_features_mv : Tensor (..., items, add_mv, 8) or None
        additional_qk_features_s  : Tensor (..., items, add_s) or None

        Returns
        -------
        q_mv, k_mv, v_mv : each (..., num_heads, items, hidden_mv_channels, 8)
        q_s,  k_s,  v_s  : each (..., num_heads, items, hidden_s_channels) or None
        """
        if additional_qk_features_mv is not None:
            inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        if scalars is not None and additional_qk_features_s is not None:
            scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)

        qkv_mv, qkv_s = self.in_linear(inputs, scalars)

        qkv_mv = rearrange(
            qkv_mv,
            "... items (qkv hidden heads) x -> qkv ... heads items hidden x",
            qkv=3,
            hidden=self.config.hidden_mv_channels,
            heads=self.config.num_heads,
        )
        q_mv, k_mv, v_mv = qkv_mv

        if qkv_s is not None:
            qkv_s = rearrange(
                qkv_s,
                "... items (qkv hidden heads) -> qkv ... heads items hidden",
                qkv=3,
                hidden=self.config.hidden_s_channels,
                heads=self.config.num_heads,
            )
            q_s, k_s, v_s = qkv_s
        else:
            q_s = k_s = v_s = None

        q_mv = self.norm_mv(q_mv)
        k_mv = self.norm_mv(k_mv)
        v_mv = self.norm_mv(v_mv)

        if self.norm_s is not None and q_s is not None:
            q_s = self.norm_s(q_s)
            k_s = self.norm_s(k_s)
            v_s = self.norm_s(v_s)

        return q_mv, k_mv, v_mv, q_s, k_s, v_s


class MultiQueryQKVModule(nn.Module):
    """Multi-query QKV: independent Q per head, K and V shared across heads.

    Parameters
    ----------
    config : SelfAttentionConfig
    """

    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self.config = config

        in_mv = config.in_mv_channels + config.additional_qk_mv_channels
        in_s = (
            config.in_s_channels + config.additional_qk_s_channels
            if config.in_s_channels is not None else None
        )

        self.q_linear = EquiLinear(
            in_mv_channels=in_mv,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=in_s,
            out_s_channels=(
                None if in_s is None
                else config.hidden_s_channels * config.num_heads
            ),
        )
        self.k_linear = EquiLinear(
            in_mv_channels=in_mv,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=in_s,
            out_s_channels=None if in_s is None else config.hidden_s_channels,
        )
        self.v_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels,
            out_s_channels=(
                None if config.in_s_channels is None
                else config.hidden_s_channels
            ),
        )
        self.norm_mv = EquiLayerNorm()
        self.norm_s = (
            nn.LayerNorm(config.hidden_s_channels)
            if config.in_s_channels is not None else None
        )

    def forward(
        self,
        inputs: torch.Tensor,
        scalars: Optional[torch.Tensor],
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor (..., items, in_mv_channels, 8)
        scalars : Tensor (..., items, in_s_channels) or None
        additional_qk_features_mv : Tensor (..., items, add_mv, 8) or None
        additional_qk_features_s  : Tensor (..., items, add_s) or None

        Returns
        -------
        q_mv : (..., num_heads, items, hidden_mv_channels, 8)
        k_mv : (..., 1, items, hidden_mv_channels, 8)  — shared across heads
        v_mv : (..., 1, items, hidden_mv_channels, 8)  — shared across heads
        q_s  : (..., num_heads, items, hidden_s_channels) or None
        k_s  : (..., 1, items, hidden_s_channels) or None
        v_s  : (..., 1, items, hidden_s_channels) or None
        """
        if additional_qk_features_mv is not None:
            qk_inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        else:
            qk_inputs = inputs

        if scalars is not None and additional_qk_features_s is not None:
            qk_scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)
        else:
            qk_scalars = scalars

        q_mv, q_s = self.q_linear(qk_inputs, qk_scalars)
        k_mv, k_s = self.k_linear(qk_inputs, qk_scalars)
        v_mv, v_s = self.v_linear(inputs, scalars)

        q_mv = rearrange(
            q_mv,
            "... items (hidden heads) x -> ... heads items hidden x",
            heads=self.config.num_heads,
            hidden=self.config.hidden_mv_channels,
        )
        k_mv = rearrange(k_mv, "... items hidden x -> ... 1 items hidden x")
        v_mv = rearrange(v_mv, "... items hidden x -> ... 1 items hidden x")

        if q_s is not None:
            q_s = rearrange(
                q_s,
                "... items (hidden heads) -> ... heads items hidden",
                heads=self.config.num_heads,
                hidden=self.config.hidden_s_channels,
            )
            k_s = rearrange(k_s, "... items hidden -> ... 1 items hidden")
            v_s = rearrange(v_s, "... items hidden -> ... 1 items hidden")

        q_mv = self.norm_mv(q_mv)
        k_mv = self.norm_mv(k_mv)
        v_mv = self.norm_mv(v_mv)

        if self.norm_s is not None and q_s is not None:
            q_s = self.norm_s(q_s)
            k_s = self.norm_s(k_s)
            v_s = self.norm_s(v_s)

        return q_mv, k_mv, v_mv, q_s, k_s, v_s
