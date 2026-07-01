"""Rotary positional encoding (RoPE) for scalar streams.

Applied only to scalar Q/K — not multivectors. Scalars are Spin(3)-invariant,
so RoPE on scalars preserves full Spin(3) equivariance of the attention layer.

Direct port from L-GATr (algebra-agnostic: operates only on scalar tensors).

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
Embedding", arXiv:2104.09864
"""

import torch
from torch import nn

from su2eqnn.utils.einsum import cached_einsum


class ApplyRotaryPositionalEncoding(nn.Module):
    """Applies RoPE to scalar tensors along a token/item dimension.

    Parameters
    ----------
    num_channels : int
        Feature dimension (must be even).
    item_dim : int
        Dimension index along which positions are enumerated (e.g. -2 for
        shape (..., items, channels)).
    base : int
        Frequency base (default 4096, same as L-GATr).
    """

    def __init__(self, num_channels: int, item_dim: int, base: int = 4096):
        super().__init__()

        assert num_channels % 2 == 0, (
            f"num_channels must be even for RoPE, got {num_channels}"
        )

        inv_freq = 1.0 / (base ** (torch.arange(0, num_channels, 2).float() / num_channels))
        self.register_buffer("inv_freq", inv_freq)
        self.item_dim = item_dim
        self.num_channels = num_channels

        self._seq_len_cached = None
        self._device_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to scalars.

        Parameters
        ----------
        scalars : Tensor with shape (..., num_channels)
            Last dimension is feature dim.

        Returns
        -------
        Tensor with same shape as scalars.
        """
        assert scalars.shape[-1] == self.num_channels
        self._update_cache(scalars)
        return scalars * self._cos_cached + self._rotate_half(scalars) * self._sin_cached

    def _update_cache(self, inputs: torch.Tensor) -> None:
        seq_len = inputs.shape[self.item_dim]
        if seq_len == self._seq_len_cached and inputs.device == self._device_cached:
            return
        self._seq_len_cached = seq_len
        self._device_cached = inputs.device
        t = torch.arange(seq_len, device=inputs.device).type_as(self.inv_freq)
        freqs = cached_einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        # Insert dims so broadcast aligns with item_dim
        item_dim = self.item_dim if self.item_dim >= 0 else inputs.ndim + self.item_dim
        for _ in range(item_dim + 1, inputs.ndim - 1):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        self._cos_cached = cos
        self._sin_cached = sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
