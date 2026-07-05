# SU2EQNN GATrBlock + GATr Net Design

**Date:** 2026-07-05
**Status:** Approved

## Goal

Port L-GATr's `GATrBlock` and `GATr` net from Cl(1,3) to Cl(3,0). Adds the top-level transformer building block and full stacked network to SU2EQNN.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Port fidelity | Direct port of L-GATr | Architecture identical; only MV dim changes automatically |
| `scalars` type | `Optional[Tensor]` | SU2EQNN convention throughout |
| Residual with optional scalars | Guard `if s is not None` | Avoids crash when scalar stream absent |
| MLP width | 2× expansion: `(C, 2C, C)` | Same as L-GATr |
| reinsert channels | Included (optional, default None) | Needed for beam reference in top tagging |
| Scope | GATrBlock + GATr net only | Top tagging experiment is next subsystem |

## Files

```
su2eqnn/layers/gatr_block.py      # GATrBlock
su2eqnn/nets/__init__.py           # new package
su2eqnn/nets/gatr.py               # GATr stacked net
```

## Component Specs

### `su2eqnn/layers/gatr_block.py` — `GATrBlock`

```python
class GATrBlock(nn.Module):
    def __init__(
        self,
        mv_channels: int,
        s_channels: Optional[int],
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
    ) -> None:
        # norm: single EquiLayerNorm (stateless, reused for both norm positions)
        # attention: SelfAttention with:
        #   in_mv_channels=mv_channels, out_mv_channels=mv_channels
        #   in_s_channels=s_channels, out_s_channels=s_channels
        #   output_init="small", dropout_prob=dropout_prob
        # mlp: GeoMLP with:
        #   mv_channels=(mv_channels, 2*mv_channels, mv_channels)
        #   s_channels=(s_channels, 2*s_channels, s_channels) if s_channels else None
        #   dropout_prob=dropout_prob

    def forward(
        self,
        multivectors: torch.Tensor,              # (..., items, C, 8)
        scalars: Optional[torch.Tensor] = None,  # (..., items, S) or None
        additional_qk_features_mv=None,
        additional_qk_features_s=None,
        attention_mask=None,
        batch_indices=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention block
        h_mv, h_s = norm(mv, scalars=s)
        h_mv, h_s = attention(h_mv, h_s, ...)
        if double_layernorm: h_mv, h_s = norm(h_mv, scalars=h_s)
        mv = mv + h_mv
        s = (s + h_s) if s is not None else None   # guarded residual

        # MLP block
        h_mv, h_s = norm(mv, scalars=s)
        h_mv, h_s = mlp(h_mv, scalars=h_s)
        if double_layernorm: h_mv, h_s = norm(h_mv, scalars=h_s)
        mv = mv + h_mv
        s = (s + h_s) if s is not None else None   # guarded residual

        return mv, s
```

Key: `s_channels=None` → attention and mlp get `in_s_channels=None` → scalar stream absent throughout.

### `su2eqnn/nets/gatr.py` — `GATr`

```python
class GATr(nn.Module):
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
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
    ) -> None:
        # linear_in: EquiLinear(in_mv → hidden_mv, in_s → hidden_s)
        # attention config: augmented with additional_qk_mv/s_channels from reinsert lengths
        # blocks: nn.ModuleList of num_blocks GATrBlocks
        # linear_out: EquiLinear(hidden_mv → out_mv, hidden_s → out_s)

    def forward(
        self,
        multivectors: torch.Tensor,              # (..., items, in_mv, 8)
        scalars: Optional[torch.Tensor] = None,  # (..., items, in_s) or None
        attention_mask=None,
        batch_indices=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Build reinserted channels from input (for Q/K in every block)
        # 2. h_mv, h_s = linear_in(mv, s)
        # 3. For each block: h_mv, h_s = block(h_mv, h_s, reinsert_mv, reinsert_s, mask, batch)
        #    - optionally wrapped in torch.utils.checkpoint.checkpoint
        # 4. out_mv, out_s = linear_out(h_mv, h_s)
        return out_mv, out_s
```

`reinsert_*_channels`: integer indices into input mv/scalar channels. Those channels are sliced from the original input and passed as `additional_qk_features_*` to every block. Enables beam token injection without changing block signature.

## Cl(3,0) vs Cl(1,3) Differences

| | L-GATr Cl(1,3) | SU2EQNN Cl(3,0) |
|-|----------------|-----------------|
| MV dim | 16 | 8 |
| `scalars` type | `torch.Tensor` (required) | `Optional[torch.Tensor]` |
| Residual | `s = s + h_s` | `s = (s + h_s) if s is not None else None` |
| Imports | `gatr.*` | `su2eqnn.*` |

Everything else is identical.

## Test Plan

Add to `tests/test_equivariance.py`:
- `test_gatr_block_equivariance` — GATrBlock(mv_channels=4, s_channels=8) commutes with Spin(3)
- `test_gatr_block_no_scalars_equivariance` — GATrBlock(mv_channels=4, s_channels=None) commutes with Spin(3)
- `test_gatr_equivariance` — GATr(2-block, mv=4→4, s=8→8) commutes with Spin(3)
