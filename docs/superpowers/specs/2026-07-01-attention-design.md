# SU2EQNN Attention Layer Design

**Date:** 2026-07-01  
**Status:** Approved

## Goal

Implement a Spin(3)-equivariant self-attention module for SU2EQNN, porting L-GATr's geometric attention from Cl(1,3) to Cl(3,0).

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Attention score | Flatten-concat → torch SDPA | Flash Attention support, minimal delta from L-GATr |
| Positional encoding | RoPE on scalars only | Fully Spin(3)-equivariant; grade-0 invariant |
| QKV style | Multi-query (MQA) | Fewer params, faster inference, L-GATr default |
| Aggregation | None | No physics application yet; add later if needed |
| Scope | `su2eqnn/layers/attention/` + `su2eqnn/primitives/attention.py` | Attention module only; GATr block deferred |

## Files

```
su2eqnn/primitives/attention.py
su2eqnn/layers/attention/__init__.py
su2eqnn/layers/attention/config.py
su2eqnn/layers/attention/qkv.py
su2eqnn/layers/attention/positional_encoding.py
su2eqnn/layers/attention/attention.py
su2eqnn/layers/attention/self_attention.py
```

## Cl(3,0) vs Cl(1,3) Differences

| | L-GATr Cl(1,3) | SU2EQNN Cl(3,0) |
|-|----------------|-----------------|
| MV dim | 16 | 8 |
| IP factors | Minkowski (grade-1: -1, rest: +1) | all +1 (Euclidean) |
| `_load_inner_product_factors` | `gatr.primitives.invariants` | `su2eqnn.primitives.invariants` |
| `LorentzJetAggregation` | optional | removed |
| `pairwise.py` | present | not created |

## Component Specs

### `su2eqnn/primitives/attention.py` — `sdp_attention`

```
inputs:
  q_mv: (..., heads, items, C, 8)
  k_mv: (..., heads, items, C, 8)
  v_mv: (..., heads, items, C, 8)
  q_s:  (..., heads, items, Ds)
  k_s:  (..., heads, items, Ds)
  v_s:  (..., heads, items, Ds)
  attn_mask: optional

score computation:
  q = cat([flatten(q_mv * ip_factors), q_s], dim=-1)  # ip_factors = ones(8)
  k = cat([flatten(k_mv), k_s], dim=-1)
  v = cat([flatten(v_mv), v_s], dim=-1)
  out = torch_sdpa(q, k, v, attn_mask)  # or xformers if available

output split:
  out_mv = out[..., :C*8].reshape(..., C, 8)
  out_s  = out[..., C*8:]
```

The dot product `q·k` = Cl(3,0) inner product (channels+components) + scalar dot product. Spin(3)-invariant since IP factors are all positive and the inner product is invariant under conjugation.

### `su2eqnn/layers/attention/config.py` — `SelfAttentionConfig`

Dataclass with fields:
- `in_mv_channels`, `out_mv_channels`, `in_s_channels`, `out_s_channels`
- `num_heads: int = 8`
- `multi_query: bool = True`
- `pos_encoding: bool = False`
- `pos_encoding_base: int = 4096`
- `increase_hidden_channels: int = 2`
- `output_init: str = "default"`
- `dropout_prob: Optional[float] = None`
- `head_scale: bool = False`
- `additional_qk_mv_channels: int = 0`
- `additional_qk_s_channels: int = 0`

Properties:
- `hidden_mv_channels = max(increase_hidden_channels * in_mv_channels // num_heads, 1)`
- `hidden_s_channels = max(increase_hidden_channels * in_s_channels // num_heads, 4)` (forced even if pos_encoding)

Removed vs L-GATr: `lorentz_agg_s_channels`, `use_pairwise_bias`, `checkpoint`.

### `su2eqnn/layers/attention/qkv.py` — `MultiQueryQKVModule`

Two variants kept for completeness (`QKVModule` standard MHA + `MultiQueryQKVModule` MQA). Default is MQA.

MQA: Q projected to `(num_heads * hidden_mv_channels)`, K and V projected to `(hidden_mv_channels)` shared across heads.

After projection: rearrange + `EquiLayerNorm` on Q, K, V independently.

### `su2eqnn/layers/attention/positional_encoding.py` — `ApplyRotaryPositionalEncoding`

Direct copy from L-GATr. No changes needed — operates on scalars only, algebra-agnostic.

### `su2eqnn/layers/attention/attention.py` — `GeometricAttention`

Thin wrapper: calls `sdp_attention` from `su2eqnn.primitives.attention`. No state.

### `su2eqnn/layers/attention/self_attention.py` — `SelfAttention`

```
__init__:
  qkv_module = MultiQueryQKVModule(config)  [or QKVModule]
  out_linear  = EquiLinear(heads*hidden_mv → out_mv, heads*hidden_s → out_s)
  pos_encoding = ApplyRotaryPositionalEncoding(hidden_s_channels) if config.pos_encoding else Identity
  attention = GeometricAttention(config)
  head_scale = nn.Parameter(ones(num_heads)) if config.head_scale

forward(x_mv, x_s, additional_qk_mv, additional_qk_s, attn_mask):
  q_mv,k_mv,v_mv,q_s,k_s,v_s = qkv_module(x_mv, x_s, ...)
  q_s = pos_encoding(q_s)
  k_s = pos_encoding(k_s)
  h_mv, h_s = attention(q_mv,k_mv,v_mv,q_s,k_s,v_s, attn_mask)
  [optional head_scale]
  h_mv = rearrange heads → (items, heads*hidden, 8)
  h_s  = rearrange heads → (items, heads*hidden_s)
  out_mv, out_s = out_linear(h_mv, h_s)
  return out_mv, out_s
```

## Equivariance Argument

**Score is Spin(3)-invariant:** `q·k = Σ_{c,i} q_mv[c,i] * k_mv[c,i] + Σ_c q_s[c] * k_s[c]`

Under ρ(w): each `q_mv[c]` → `w q_mv[c] w⁻¹`. The Cl(3,0) inner product satisfies `<wx, wy> = <x,y>` for w ∈ Spin(3) (isometry). So the MV score term is invariant. Scalar terms unchanged. Therefore `attn_weights` are invariant.

**Output is equivariant:** `out_mv[i] = Σ_j attn[i,j] * v_mv[j]` — linear combination with invariant coefficients of equivariant values. Linear combinations of equivariant tensors are equivariant.

## Test Plan

Add to `tests/test_equivariance.py`:
- `test_sdp_attention_equivariance` — `sdp_attention(ρ(w)Q, ρ(w)K, ρ(w)V) == ρ(w)(sdp_attention(Q,K,V))` for MV output; scalar output invariant
- `test_self_attention_equivariance` — full `SelfAttention` layer commutes with Spin(3)
- `test_rope_scalar_only` — RoPE output unchanged under Spin(3) (scalars don't transform)
