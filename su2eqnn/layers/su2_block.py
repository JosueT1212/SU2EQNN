"""Spin(3)-equivariant transformer block for Cl(3,0)."""

from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn

from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.attention.self_attention import SelfAttention
from su2eqnn.layers.layer_norm import EquiLayerNorm
from su2eqnn.layers.mlp.config import MLPConfig
from su2eqnn.layers.mlp.mlp import GeoMLP


class SU2Block(nn.Module):
    """Spin(3)-equivariant transformer encoder block for Cl(3,0).

    Applies: LayerNorm → SelfAttention → residual → LayerNorm → GeoMLP → residual.

    Parameters
    ----------
    mv_channels : int
        Number of input/output multivector channels.
    s_channels : int or None
        Number of input/output scalar channels. None means no scalar stream.
    attention : SelfAttentionConfig
        Template config; in/out channels, output_init, dropout_prob are overridden here.
    mlp : MLPConfig
        Template config; mv_channels, s_channels, dropout_prob are overridden here.
    dropout_prob : float or None
        Dropout probability forwarded to attention and GeoMLP.
    double_layernorm : bool
        If True, apply LayerNorm after each sub-block in addition to before (pre+post norm).
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: Optional[int],
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.double_layernorm = double_layernorm

        # EquiLayerNorm for the MV stream (stateless; reused for all norm positions).
        self.norm = EquiLayerNorm()
        # Standard LayerNorm for the scalar stream (None when no scalar stream).
        self.norm_s = nn.LayerNorm(s_channels) if s_channels is not None else None

        # Self-attention: fill in channels and output init
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = SelfAttention(attention)

        # GeoMLP: fill in channels; activation and other fields come from template
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(
                (s_channels, 2 * s_channels, s_channels)
                if s_channels is not None else None
            ),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def _norm_dual(
        self,
        mv: torch.Tensor,
        s: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply EquiLayerNorm to MV stream, nn.LayerNorm to scalar stream."""
        out_mv = self.norm(mv)
        out_s = self.norm_s(s) if (self.norm_s is not None and s is not None) else None
        return out_mv, out_s

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
        multivectors : Tensor (..., items, mv_channels, 8)
        scalars : Tensor (..., items, s_channels) or None
        additional_qk_features_mv : Tensor (..., items, add_mv, 8) or None
        additional_qk_features_s : Tensor (..., items, add_s) or None
        attention_mask : Tensor (..., items, items) or None

        Returns
        -------
        out_mv : Tensor (..., items, mv_channels, 8)
        out_s : Tensor (..., items, s_channels) or None
        """
        # --- Attention sub-block (pre-norm) ---
        h_mv, h_s = self._norm_dual(multivectors, scalars)
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
        )
        if self.double_layernorm:
            h_mv, h_s = self._norm_dual(h_mv, h_s)
        out_mv = multivectors + h_mv
        out_s = (scalars + h_s) if scalars is not None else None

        # --- MLP sub-block (pre-norm) ---
        h_mv, h_s = self._norm_dual(out_mv, out_s)
        h_mv, h_s = self.mlp(h_mv, scalars=h_s)
        if self.double_layernorm:
            h_mv, h_s = self._norm_dual(h_mv, h_s)
        out_mv = out_mv + h_mv
        out_s = (out_s + h_s) if out_s is not None else None

        return out_mv, out_s
