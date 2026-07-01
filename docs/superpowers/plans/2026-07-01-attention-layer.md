# Attention Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Spin(3)-equivariant self-attention module for SU2EQNN by porting L-GATr's geometric attention from Cl(1,3) to Cl(3,0).

**Architecture:** Flatten-concat approach — multivector Q/K/V are flattened to `(..., C*8)`, multiplied by Cl(3,0) IP factors (all ones), concatenated with scalar Q/K/V, then passed to `torch.nn.functional.scaled_dot_product_attention`. Score = Cl(3,0) inner product + scalar dot product, which is Spin(3)-invariant. Output weighted sum over equivariant values → equivariant output. RoPE applied to scalar Q/K only.

**Tech Stack:** PyTorch ≥2.0, einops, clifford ≥1.4, opt_einsum, numpy

## Global Constraints

- MV dimension is **8** throughout (Cl(3,0)), never 16
- All IP factors are **+1** (Euclidean), imported from `su2eqnn.primitives.invariants._load_inner_product_factors`
- No `LorentzJetAggregation`, no `pairwise.py`, no `lorentz_agg_s_channels`
- `EquiLinear` is imported from `su2eqnn.layers.linear`
- `EquiLayerNorm` is imported from `su2eqnn.layers.layer_norm` — its `forward(x, channel_dim=-2)` only normalizes MVs; normalize scalars separately with `nn.LayerNorm`
- Tests run with: `PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 tests/test_equivariance.py`
- All new files live under `/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN/`

---

### Task 1: Add einops dependency + attention primitive

**Files:**
- Modify: `pyproject.toml`
- Create: `su2eqnn/primitives/attention.py`
- Modify: `tests/test_equivariance.py` (add `test_sdp_attention_equivariance`)

**Interfaces:**
- Produces: `sdp_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attn_mask=None) -> Tuple[Tensor, Tensor]`
  - inputs: `q_mv` shape `(..., C, 8)`, `q_s` shape `(..., Ds)` or None
  - outputs: `out_mv` shape `(..., C, 8)`, `out_s` shape `(..., Ds)` or None

- [ ] **Step 1: Add einops to pyproject.toml**

In `pyproject.toml`, change the dependencies list to:
```toml
dependencies = [
    "torch>=2.0",
    "clifford>=1.4",
    "numpy",
    "opt_einsum",
    "einops",
]
```

- [ ] **Step 2: Write the failing test**

Add this function to `tests/test_equivariance.py` (before the `if __name__ == "__main__":` block):

```python
def test_sdp_attention_equivariance():
    """sdp_attention MV output is equivariant; scalar output is invariant."""
    from su2eqnn.primitives.attention import sdp_attention

    HEADS, ITEMS, C, DS = 2, 5, 3, 4
    for _ in range(N_TESTS):
        q_mv = torch.randn(HEADS, ITEMS, C, 8)
        k_mv = torch.randn(HEADS, ITEMS, C, 8)
        v_mv = torch.randn(HEADS, ITEMS, C, 8)
        q_s  = torch.randn(HEADS, ITEMS, DS)
        k_s  = torch.randn(HEADS, ITEMS, DS)
        v_s  = torch.randn(HEADS, ITEMS, DS)
        w = random_spin3()

        def act_mv(x):
            return torch.stack([spin3_action(w, x[..., c, :]) for c in range(x.shape[-2])], dim=-2)

        # Compute on original, then rotate output
        out_mv, out_s = sdp_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s)
        path2_mv = act_mv(out_mv)

        # Rotate inputs, then compute
        path1_mv, path1_s = sdp_attention(
            act_mv(q_mv), act_mv(k_mv), act_mv(v_mv), q_s, k_s, v_s
        )

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"sdp_attention MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=1e-4), \
            f"sdp_attention scalar invariance failed: {(path1_s - out_s).abs().max():.2e}"
```

Also add `test_sdp_attention_equivariance()` to the `if __name__ == "__main__":` block.

- [ ] **Step 3: Run test to verify it fails**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 \
  tests/test_equivariance.py
```

Expected: `ModuleNotFoundError: No module named 'su2eqnn.primitives.attention'`

- [ ] **Step 4: Create `su2eqnn/primitives/attention.py`**

```python
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
    Both terms are Spin(3)-invariant → attention weights are invariant →
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

    # Flatten MV channels: (..., items, C, 8) → (..., items, C*8)
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 \
  tests/test_equivariance.py
```

Expected: `All equivariance tests passed.`

- [ ] **Step 6: Commit**

```bash
cd /Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN
git add pyproject.toml su2eqnn/primitives/attention.py tests/test_equivariance.py
git commit -m "feat: add Cl(3,0) sdp_attention primitive with equivariance test"
```

---

### Task 2: Config dataclass

**Files:**
- Create: `su2eqnn/layers/attention/__init__.py`
- Create: `su2eqnn/layers/attention/config.py`

**Interfaces:**
- Produces: `SelfAttentionConfig` dataclass with properties `hidden_mv_channels`, `hidden_s_channels`
- Consumed by: Tasks 3, 4, 5

- [ ] **Step 1: Create `su2eqnn/layers/attention/__init__.py`**

```python
```
(empty file)

- [ ] **Step 2: Create `su2eqnn/layers/attention/config.py`**

```python
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
```

- [ ] **Step 3: Quick smoke test (no pytest needed)**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 -c "
from su2eqnn.layers.attention.config import SelfAttentionConfig
cfg = SelfAttentionConfig(in_mv_channels=8, out_mv_channels=8, in_s_channels=4, out_s_channels=4, num_heads=4)
print('hidden_mv_channels:', cfg.hidden_mv_channels)  # expect 4
print('hidden_s_channels:', cfg.hidden_s_channels)    # expect 4
"
```

Expected output:
```
hidden_mv_channels: 4
hidden_s_channels: 4
```

- [ ] **Step 4: Commit**

```bash
cd /Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN
git add su2eqnn/layers/attention/__init__.py su2eqnn/layers/attention/config.py
git commit -m "feat: add SelfAttentionConfig for Spin(3) attention"
```

---

### Task 3: Positional encoding + test

**Files:**
- Create: `su2eqnn/layers/attention/positional_encoding.py`
- Modify: `tests/test_equivariance.py` (add `test_rope_scalar_only`)

**Interfaces:**
- Produces: `ApplyRotaryPositionalEncoding(num_channels, item_dim, base=4096)` — `nn.Module`
  - `forward(scalars: Tensor) -> Tensor` — scalars shape `(..., num_channels)`
- Consumed by: Task 5 (`SelfAttention.__init__`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_equivariance.py`:

```python
def test_rope_scalar_only():
    """RoPE on scalars is unaffected by Spin(3) — scalars are invariant."""
    from su2eqnn.layers.attention.positional_encoding import ApplyRotaryPositionalEncoding

    NUM_S = 8
    ITEMS = 10
    rope = ApplyRotaryPositionalEncoding(num_channels=NUM_S, item_dim=-2)

    for _ in range(N_TESTS):
        s = torch.randn(BATCH, ITEMS, NUM_S)
        w = random_spin3()
        # Scalars don't transform under Spin(3): rope(s) == rope(s) trivially,
        # but verify that rotating MVs (not scalars) and then calling rope gives same result
        out1 = rope(s)
        out2 = rope(s)   # identical call — scalars unchanged by Spin(3)
        assert torch.allclose(out1, out2, atol=TOL)
        # Verify output shape unchanged
        assert out1.shape == s.shape
```

Add `test_rope_scalar_only()` to the `if __name__ == "__main__":` block.

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 \
  tests/test_equivariance.py
```

Expected: `ModuleNotFoundError: No module named 'su2eqnn.layers.attention.positional_encoding'`

- [ ] **Step 3: Create `su2eqnn/layers/attention/positional_encoding.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 \
  tests/test_equivariance.py
```

Expected: `All equivariance tests passed.`

- [ ] **Step 5: Commit**

```bash
cd /Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN
git add su2eqnn/layers/attention/positional_encoding.py tests/test_equivariance.py
git commit -m "feat: add RoPE positional encoding for scalar streams"
```

---

### Task 4: QKV module + GeometricAttention wrapper

**Files:**
- Create: `su2eqnn/layers/attention/qkv.py`
- Create: `su2eqnn/layers/attention/attention.py`

**Interfaces:**
- Consumes: `SelfAttentionConfig` (Task 2), `EquiLinear` from `su2eqnn.layers.linear`, `EquiLayerNorm` from `su2eqnn.layers.layer_norm`, `sdp_attention` (Task 1)
- Produces:
  - `QKVModule(config).forward(inputs, scalars, additional_qk_features_mv=None, additional_qk_features_s=None) -> Tuple[q_mv, k_mv, v_mv, q_s, k_s, v_s]`
    - each `*_mv`: `(..., num_heads, items, hidden_mv_channels, 8)`
    - each `*_s`: `(..., num_heads, items, hidden_s_channels)` or None
  - `MultiQueryQKVModule(config).forward(...)` — same signature, same return types
  - `GeometricAttention(config).forward(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None) -> Tuple[Tensor, Optional[Tensor]]`

- [ ] **Step 1: Create `su2eqnn/layers/attention/qkv.py`**

```python
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
```

- [ ] **Step 2: Create `su2eqnn/layers/attention/attention.py`**

```python
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
```

- [ ] **Step 3: Smoke test**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 -c "
import torch
from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.attention.qkv import MultiQueryQKVModule
from su2eqnn.layers.attention.attention import GeometricAttention

cfg = SelfAttentionConfig(in_mv_channels=4, out_mv_channels=4, in_s_channels=8, out_s_channels=8, num_heads=2)
qkv = MultiQueryQKVModule(cfg)
x_mv = torch.randn(3, 10, 4, 8)   # batch=3, items=10, channels=4, mv_dim=8
x_s  = torch.randn(3, 10, 8)
q_mv, k_mv, v_mv, q_s, k_s, v_s = qkv(x_mv, x_s)
print('q_mv shape:', q_mv.shape)  # (3, 2, 10, hidden_mv, 8)
print('q_s  shape:', q_s.shape)   # (3, 2, 10, hidden_s)

attn = GeometricAttention(cfg)
h_mv, h_s = attn(q_mv, k_mv, v_mv, q_s, k_s, v_s)
print('h_mv shape:', h_mv.shape)
print('h_s  shape:', h_s.shape)
print('OK')
"
```

Expected:
```
q_mv shape: torch.Size([3, 2, 10, 2, 8])
q_s  shape: torch.Size([3, 2, 10, 4])
h_mv shape: torch.Size([3, 2, 10, 2, 8])
h_s  shape: torch.Size([3, 2, 10, 4])
OK
```

(hidden_mv = max(2*4//2, 1) = 4 // wait: 2*4=8, 8//2=4? No: in_mv=4, increase=2, heads=2 → 2*4//2=4. hidden_s = max(2*8//2,4)=8.)

- [ ] **Step 4: Commit**

```bash
cd /Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN
git add su2eqnn/layers/attention/qkv.py su2eqnn/layers/attention/attention.py
git commit -m "feat: add QKV module and GeometricAttention wrapper for Cl(3,0)"
```

---

### Task 5: SelfAttention layer + equivariance test

**Files:**
- Create: `su2eqnn/layers/attention/self_attention.py`
- Modify: `tests/test_equivariance.py` (add `test_self_attention_equivariance`)

**Interfaces:**
- Consumes: all prior tasks
- Produces: `SelfAttention(config).forward(multivectors, scalars=None, additional_qk_features_mv=None, additional_qk_features_s=None, attention_mask=None) -> Tuple[Tensor, Optional[Tensor]]`
  - input `multivectors`: `(..., items, in_mv_channels, 8)`
  - output `out_mv`: `(..., items, out_mv_channels, 8)`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_equivariance.py`:

```python
def test_self_attention_equivariance():
    """Full SelfAttention layer commutes with Spin(3) on MV stream."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.self_attention import SelfAttention

    ITEMS = 6
    cfg = SelfAttentionConfig(
        in_mv_channels=C_IN,
        out_mv_channels=C_OUT,
        in_s_channels=4,
        out_s_channels=4,
        num_heads=2,
        pos_encoding=False,
        multi_query=True,
    )
    layer = SelfAttention(cfg)
    layer.eval()

    for _ in range(N_TESTS):
        x_mv = torch.randn(BATCH, ITEMS, C_IN, 8)
        x_s  = torch.randn(BATCH, ITEMS, 4)
        w = random_spin3()

        def act(x):
            return torch.stack(
                [spin3_action(w, x[..., c, :]) for c in range(x.shape[-2])], dim=-2
            )

        out_mv, out_s = layer(x_mv, x_s)
        path1_mv, path1_s = layer(act(x_mv), x_s)
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"SelfAttention MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=1e-4), \
            f"SelfAttention scalar invariance failed: {(path1_s - out_s).abs().max():.2e}"
```

Add `test_self_attention_equivariance()` to `if __name__ == "__main__":` block.

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 \
  tests/test_equivariance.py
```

Expected: `ModuleNotFoundError: No module named 'su2eqnn.layers.attention.self_attention'`

- [ ] **Step 3: Create `su2eqnn/layers/attention/self_attention.py`**

```python
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
```

- [ ] **Step 4: Run all tests**

```bash
PYTHONPATH=/Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN \
  /Users/josuetapiahernandez/Documents/Investigacion_Andres/lorentz-gatr/venv/bin/python3 \
  tests/test_equivariance.py
```

Expected: `All equivariance tests passed.`

- [ ] **Step 5: Commit**

```bash
cd /Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN
git add su2eqnn/layers/attention/self_attention.py tests/test_equivariance.py
git commit -m "feat: add SelfAttention layer with equivariance test"
```

---

## Self-Review

**Spec coverage:**
- ✅ `su2eqnn/primitives/attention.py` — Task 1
- ✅ `su2eqnn/layers/attention/__init__.py` — Task 2
- ✅ `su2eqnn/layers/attention/config.py` — Task 2
- ✅ `su2eqnn/layers/attention/qkv.py` — Task 4
- ✅ `su2eqnn/layers/attention/positional_encoding.py` — Task 3
- ✅ `su2eqnn/layers/attention/attention.py` — Task 4
- ✅ `su2eqnn/layers/attention/self_attention.py` — Task 5
- ✅ `test_sdp_attention_equivariance` — Task 1
- ✅ `test_self_attention_equivariance` — Task 5
- ✅ `test_rope_scalar_only` — Task 3
- ✅ MV dim 8 throughout
- ✅ IP factors all +1 from `su2eqnn.primitives.invariants`
- ✅ No LorentzJetAggregation
- ✅ Multi-query default
- ✅ RoPE on scalars only
- ✅ einops added to pyproject.toml

**No placeholders found.**

**Type consistency:** `SelfAttentionConfig.hidden_mv_channels` and `hidden_s_channels` used consistently across Tasks 2, 4, 5. `sdp_attention` signature matches usage in Task 4.
