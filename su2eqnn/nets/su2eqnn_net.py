"""Spin(3)-equivariant transformer network for Cl(3,0)."""

from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.su2_block import SU2Block
from su2eqnn.layers.linear import EquiLinear
from su2eqnn.layers.mlp.config import MLPConfig


class SU2EQNN(nn.Module):
    """Spin(3)-equivariant transformer for Cl(3,0) multivector data.

    Stacks `num_blocks` SU2Blocks with initial and final EquiLinear projections.

    Input shape:  (..., items, in_mv_channels, 8)
    Output shape: (..., items, out_mv_channels, 8)
    Hidden shape: (..., items, hidden_mv_channels, 8)

    Parameters
    ----------
    in_mv_channels : int
    out_mv_channels : int
    hidden_mv_channels : int
    in_s_channels : int or None
    out_s_channels : int or None
    hidden_s_channels : int or None
    attention : SelfAttentionConfig
        Template; channels and additional_qk_*_channels filled in here.
    mlp : MLPConfig
        Template; channels filled in by each SU2Block.
    num_blocks : int
    reinsert_mv_channels : tuple of int or None
        Indices of input MV channels to re-inject as additional Q/K features in every block.
        Enables stable injection of reference vectors (e.g. beam token) through all blocks.
    reinsert_s_channels : tuple of int or None
        Indices of input scalar channels to re-inject as additional Q/K features in every block.
    patch_mv_channels : int or None
        When set, SU2EQNN owns one EquiLinear(patch_mv_channels -> hidden_mv_channels)
        per block and adds its output to the hidden multivector stream immediately
        before each block runs (Li et al. 2208.07814-style patch structure, adapted
        to be Spin(3)-equivariant). None disables the pathway entirely.
    checkpoint_blocks : bool
        Use gradient checkpointing to save memory at the cost of slower backward.
    dropout_prob : float or None
    double_layernorm : bool
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        reinsert_mv_channels: Optional[Tuple[int, ...]] = None,
        reinsert_s_channels: Optional[Tuple[int, ...]] = None,
        patch_mv_channels: Optional[int] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self._reinsert_mv_channels = reinsert_mv_channels
        self._reinsert_s_channels = reinsert_s_channels
        self._checkpoint_blocks = checkpoint_blocks

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )

        attention = replace(
            SelfAttentionConfig.cast(attention),
            additional_qk_mv_channels=(
                0 if reinsert_mv_channels is None else len(reinsert_mv_channels)
            ),
            additional_qk_s_channels=(
                0 if reinsert_s_channels is None else len(reinsert_s_channels)
            ),
        )
        mlp = MLPConfig.cast(mlp)

        self.blocks = nn.ModuleList([
            SU2Block(
                mv_channels=hidden_mv_channels,
                s_channels=hidden_s_channels,
                attention=attention,
                mlp=mlp,
                dropout_prob=dropout_prob,
                double_layernorm=double_layernorm,
            )
            for _ in range(num_blocks)
        ])

        self.patch_proj = (
            nn.ModuleList([
                EquiLinear(
                    patch_mv_channels,
                    hidden_mv_channels,
                    in_s_channels=None,
                    out_s_channels=None,
                    bias=False,
                )
                for _ in range(num_blocks)
            ])
            if patch_mv_channels is not None
            else None
        )

        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patch_mv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : Tensor (..., items, in_mv_channels, 8)
        scalars : Tensor (..., items, in_s_channels) or None
        attention_mask : Tensor (..., items, items) or None
        patch_mv : Tensor (..., items, patch_mv_channels, 8) or None
            Re-injected at every block when patch_mv_channels was set at construction.

        Returns
        -------
        out_mv : Tensor (..., items, out_mv_channels, 8)
        out_s : Tensor (..., items, out_s_channels) or None
        """
        additional_qk_features_mv, additional_qk_features_s = (
            self._construct_reinserted_channels(multivectors, scalars)
        )

        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)

        for i, block in enumerate(self.blocks):
            if self.patch_proj is not None and patch_mv is not None:
                patch_h_mv, _ = self.patch_proj[i](patch_mv)
                h_mv = h_mv + patch_h_mv
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        out_mv, out_s = self.linear_out(h_mv, scalars=h_s)
        return out_mv, out_s

    def _construct_reinserted_channels(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Slice input channels to re-inject as additional Q/K features in every block."""
        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            if scalars is None:
                raise ValueError("reinsert_s_channels requires scalars to be provided")
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s
