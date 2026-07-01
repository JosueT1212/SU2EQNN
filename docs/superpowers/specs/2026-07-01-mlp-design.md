# SU2EQNN GeoMLP Design

**Date:** 2026-07-01  
**Status:** Approved

## Goal

Implement a Spin(3)-equivariant Geometric MLP for SU2EQNN, porting L-GATr's `GeoMLP` from Cl(1,3) to Cl(3,0). Includes grade dropout and gated nonlinearities.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| First layer | GeometricBilinear (GP-based) | Only equivariant nonlinearity; mixes grades |
| Subsequent layers | ScalarGatedNonlinearity + EquiLinear | Standard pattern from L-GATr |
| Dropout | Optional GradeDropout | Port L-GATr's grade_dropout, adjust 5→4 grades |
| Scope | MLP only, no GATrBlock | Deferred |

## Files

```
su2eqnn/primitives/dropout.py
su2eqnn/layers/dropout.py
su2eqnn/layers/mlp/__init__.py
su2eqnn/layers/mlp/config.py
su2eqnn/layers/mlp/nonlinearities.py
su2eqnn/layers/mlp/geometric_bilinears.py
su2eqnn/layers/mlp/mlp.py
```

## Cl(3,0) vs Cl(1,3) Differences

| | L-GATr Cl(1,3) | SU2EQNN Cl(3,0) |
|-|----------------|-----------------|
| MV dim | 16 | 8 |
| Grades | 5 (0–4) | 4 (0–3) |
| `grade_dropout` reshape | `view(-1, 5, 16)` | `view(-1, 4, 8)` |
| Gate index | `x[..., [0]]` | `x[..., [0]]` (same) |
| Imports | `gatr.*` | `su2eqnn.*` |

## Component Specs

### `su2eqnn/primitives/dropout.py` — `grade_dropout`

```python
def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    # x: (..., 8)
    x = grade_project(x)          # (..., 4, 8)
    h = x.view(-1, 4, 8)
    h = F.dropout1d(h, p=p, training=training, inplace=False)
    h = h.view(x.shape)
    return h.sum(dim=-2)           # (..., 8)
```

Drops entire grade blocks independently with probability `p`. Spin(3)-equivariant: zeroing a grade block commutes with the rotor action since each grade is a subrepresentation.

### `su2eqnn/layers/dropout.py` — `GradeDropout`

```
GradeDropout(p: float)
forward(multivectors: (..., C, 8), scalars: (..., S)) -> (out_mv, out_s)
  out_mv = grade_dropout(multivectors, p, training)
  out_s  = F.dropout(scalars, p, training)
```

### `su2eqnn/layers/mlp/config.py` — `MLPConfig`

Dataclass with fields:
- `mv_channels: List[int]` — channel sizes from input to output
- `s_channels: Optional[List[int]]` — scalar channels per layer, same length as mv_channels
- `activation: str = "gelu"` — one of {"relu", "sigmoid", "gelu"}
- `dropout_prob: Optional[float] = None`

### `su2eqnn/layers/mlp/nonlinearities.py` — `ScalarGatedNonlinearity`

```
ScalarGatedNonlinearity(nonlinearity: str = "gelu")
forward(multivectors: (..., C, 8), scalars: (..., S)) -> (out_mv, out_s)
  gate   = multivectors[..., [0]]       # grade-0 component, shape (..., C, 1)
  out_mv = gated_fn(multivectors, gate) # f(gate) * multivectors
  out_s  = scalar_fn(scalars)           # f(scalars) pointwise
```

Gate = grade-0 (scalar part) of each MV channel — Spin(3)-invariant. Multiplying by invariant scalar preserves equivariance.

### `su2eqnn/layers/mlp/geometric_bilinears.py` — `GeometricBilinear`

```
GeometricBilinear(in_mv, out_mv, hidden_mv=None, in_s=None, out_s=None)
forward(multivectors: (..., in_mv, 8), scalars: (..., in_s)) -> (out_mv, out_s)
  left  = linear_left(x, s)             # (..., hidden, 8)  standard init
  right = linear_right(x, s)            # (..., hidden, 8)  almost_unit_scalar init
  gp    = geometric_product(left, right) # (..., hidden, 8)
  out_mv, out_s = linear_out(gp, s)
  out_mv = EquiLayerNorm(out_mv)
```

`linear_right` uses `almost_unit_scalar` init so GP starts near identity, stabilizing early training.

### `su2eqnn/layers/mlp/mlp.py` — `GeoMLP`

```
GeoMLP(config: MLPConfig)
forward(multivectors: (..., mv_channels[0], 8), scalars) -> (out_mv, out_s)

layers built from config.mv_channels = [in, h1, h2, ..., out]:
  [0→1]: GeometricBilinear(mv[0], mv[1], s[0], s[1])
         [optional GradeDropout]
  [1→2]: ScalarGatedNonlinearity + EquiLinear(mv[1], mv[2], s[1], s[2])
  ...
  [n-1→n]: ScalarGatedNonlinearity + EquiLinear(mv[n-1], mv[n], s[n-1], s[n])
```

Minimum 2 channels (in + out). One GeometricBilinear at front, rest are gated linear.

## Equivariance Argument

- **grade_dropout**: zeroes entire grade-k blocks → each grade block is a Spin(3) subrepresentation → zeroing commutes with ρ(w)
- **ScalarGatedNonlinearity**: `f(gate) * x` where gate = grade-0 component (invariant) → scalar × equivariant = equivariant
- **GeometricBilinear**: `GP(ρ(w)(x), ρ(w)(y)) = ρ(w)(GP(x,y))` → equivariant; followed by equivariant EquiLinear and EquiLayerNorm
- **GeoMLP**: composition of equivariant layers = equivariant

## Test Plan

Add to `tests/test_equivariance.py`:
- `test_grade_dropout_equivariance` — grade_dropout commutes with Spin(3) (eval mode, p=0.5)
- `test_scalar_gated_nonlinearity_equivariance` — gated GeLU commutes with Spin(3)
- `test_geometric_bilinear_equivariance` — GeometricBilinear commutes with Spin(3)
- `test_geo_mlp_equivariance` — full GeoMLP (2-layer) commutes with Spin(3)
