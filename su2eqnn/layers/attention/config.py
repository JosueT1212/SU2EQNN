"""Configuration dataclass for Spin(3)-equivariant self-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class SelfAttentionConfig:
    """Configuration for Spin(3)-equivariant self-attention.

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels.
    out_mv_channels : int
        Output multivector channels.
    in_s_channels : int or None
        Input scalar channels. None → no scalar stream.
    out_s_channels : int or None
        Output scalar channels. None → no scalar stream.
    num_heads : int
        Number of attention heads.
    multi_query : bool
        Use multi-query attention (K/V shared across heads).
    pos_encoding : bool
        Apply RoPE to scalar Q/K.
    pos_encoding_base : int
        Frequency base for RoPE.
    increase_hidden_channels : int
        Hidden channel multiplier before splitting into heads.
    output_init : str
        Initialization scheme for final EquiLinear.
    dropout_prob : float or None
        Dropout probability (not yet implemented; reserved).
    head_scale : bool
        Learnable per-head scalar (NormFormer HeadScaleMHA).
    additional_qk_mv_channels : int
        Extra MV channels concatenated to inputs before QKV projection.
    additional_qk_s_channels : int
        Extra scalar channels concatenated to inputs before QKV projection.
    """

    in_mv_channels: Optional[int] = None
    out_mv_channels: Optional[int] = None
    in_s_channels: Optional[int] = None
    out_s_channels: Optional[int] = None
    num_heads: int = 8
    multi_query: bool = True
    pos_encoding: bool = False
    pos_encoding_base: int = 4096
    increase_hidden_channels: int = 2
    output_init: str = "default"
    dropout_prob: Optional[float] = None
    head_scale: bool = False
    additional_qk_mv_channels: int = 0
    additional_qk_s_channels: int = 0

    def __post_init__(self):
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ("null", "none"):
            self.dropout_prob = None

    @property
    def hidden_mv_channels(self) -> Optional[int]:
        if self.in_mv_channels is None:
            return None
        return max(self.increase_hidden_channels * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> Optional[int]:
        if self.in_s_channels is None:
            return None
        h = max(self.increase_hidden_channels * self.in_s_channels // self.num_heads, 4)
        if self.pos_encoding:
            h = (h + 1) // 2 * 2   # force even
            h = max(h, 8)
        return h

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        if isinstance(config, SelfAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Cannot cast {config} to {cls}")
