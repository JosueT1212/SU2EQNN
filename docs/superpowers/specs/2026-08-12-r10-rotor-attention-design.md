# R10: rotor-attention (RoPE ⊗ CARE ⊗ LieRE, ported to Cl(3,0)/Spin(3))

## Motivation

R1 = R4 exactly at full scale (0.9850 = 0.9850, see `[[r1-baseline-full-result]]`):
additive per-particle bivector injection buys convergence speed, not ceiling
— the exact-equivariant backbone already represents that information given
enough scale. R8's single-channel pseudoscalar variant shows the same
pattern at mini scale. Any new R-series entry that only adds *another*
per-particle additive feature risks tying R4 again for the same structural
reason.

Rotor-attention is a different kind of change: instead of adding a feature
channel, it changes *how information mixes across particles* inside
attention — Q/K get sandwich-rotated by a per-particle rotor before the
dot-product score, making the score depend on relative angular structure
multiplicatively and non-commutatively. `sdp_attention`
(`su2eqnn/primitives/attention.py`) currently has no such mechanism — it's
a plain flattened dot product. This is a genuinely new function class, not
reducible to stacking more `EquiLinear`/`GeoMLP` layers, so it can plausibly
raise ceiling rather than just convergence speed, without giving up exact
equivariance.

Literature basis (full detail in `rope-clifford-care-alibi/report.md`):
RoPE's per-token composition trick (§3.2.1, arXiv:2104.09864) generalizes
from complex-number rotation to Clifford-rotor sandwich (CARE,
arXiv:2511.11665) to non-abelian per-token learned generators (LieRE,
arXiv:2406.10322, §3.1). su(2) has no useful abelian subalgebra beyond a
single fixed axis (STRING's commuting-generator route degenerates to plain
RoPE here), so exploiting the full 3-dimensional bivector space forces the
LieRE non-abelian regime — which in turn forces a choice about how the
per-particle generator is built, since a naively free/frozen generator
basis is not equivariant (see Design below).

## Design

### Core mechanism (shared by both arms)

Per-token rotor, applied like RoPE/LieRE apply per-token rotation — not a
pairwise O(N²) construction:

```
B_i        = generator bivector for particle i, shape (..., 3) → (e12,e13,e23) slots
theta_i    = |B_i|
R_i        = cos(theta_i/2) + sinc_half(theta_i) * B_i     # 8-dim MV, scalar+bivector only
Q'_i       = R_i * Q_i * reverse(R_i)     # sandwich, via geometric_product() x2
K'_i       = R_i * K_i * reverse(R_i)
score_ij   = <Q'_i, K'_j>                 # existing sdp_attention scoring, unchanged
```

`sinc_half(theta) = sin(theta/2)/theta`, computed with a small-angle-safe
series (`0.5 - theta**2/48 + ...`) near `theta≈0` to avoid the 0/0 — same
class of guard as R8's degeneracy constraint. `geometric_product` and
`reverse` are existing primitives (`su2eqnn/primitives/bilinear.py`,
`su2eqnn/primitives/linear.py`) — the sandwich needs zero new
Clifford-algebra machinery, only a new rotor-construction step plus two
`geometric_product` calls.

**Equivariance requirement**: under global rotor `g` applied to the whole
jet, `Q_i → g Q_i g~`. The sandwich score is invariant only if
`R_i(g·x) = g R_i(x) g~` — i.e. the generator `B_i` must itself be built
from quantities that co-rotate with the input. A `B_i` built from a frozen
learned basis contracted with rotation-invariant scalars does **not**
satisfy this (frozen basis doesn't rotate, so `R_i` stays fixed while
`Q_i`/`K_i` rotate under it — score changes, weights stop being invariant,
`softmax(score)·V` stops being equivariant even though `V=X W_V` itself is
fine). Two arms below resolve this differently — that's the experiment.

### R10a — wedge-basis (exact-equivariant, primary)

Generator built entirely from co-rotating vectors already available in the
embedding pipeline (jet axis, subjet axes — same `axes` tensor R4/R8/R9
already precompute via `precompute_axes.py`), combined with **learned
scalar** coefficients only (scalars are invariant, so a scalar-weighted sum
of equivariant bivectors stays equivariant — the same rule
`equi_linear`/`GeoMLP` already rely on):

```
B_i = sum_k  c_k(s_i) * wedge(p_hat_i, axis_k)
```

- `wedge(a, b)` reuses `embed_tagging_data`'s exact convention
  (`experiments/top_tagging/embedding.py` lines ~85-87):
  `e12 = a_x b_y - a_y b_x`, `e13 = a_x b_z - a_z b_x`, `e23 = a_y b_z - a_z b_y`.
- Candidate axes: reuse the existing `(batchsize, K, 3)` axes array
  (jet_axis + up to 3 exclusive-kT subjets, K≤4) — no new precompute.
- `c_k(s_i)`: small learned linear map from the existing per-particle
  scalar stream `s_i` (invariant features already flowing through the
  scalar channel) to `K` coefficients. If `K=1` (jet-axis only, matching
  R8a's proven single-axis signal), `c_1` can be a single learned scalar
  per head (no per-particle dependence needed) — start here, the simplest
  version that is still non-degenerate.

This has zero new equivariance risk: every term is an existing-primitive
combination (`wedge`, scalar-weighted sum), same pattern the codebase
already trusts.

### R10b — frozen-basis (approximate, secondary ablation)

Literal LieRE port: `B_i = sum_k p_i^k * A_k`, `{A_k}` a **free**
`nn.Parameter` (3-vector per head, unconstrained, analogous to LieRE's
`basis_raw_params`), `p_i` a per-particle invariant scalar feature
(e.g. rescaled `|p⃗_i|` or `E_i`, already available pre-embedding).
Only approximately equivariant — no architectural guarantee, optionally
paired with SO(3) rotation-augmentation during training to close the gap
empirically (LieRE's own bet: capacity over exact relative-position
property, see report.md LieRE §3.1 and Spherical RoPE's finding that exact
equivariance isn't necessary for RoPE-style gains in vision).

Purpose: isolate whether *exactness* costs anything, or whether the
rotor-attention *mechanism* is the real lever regardless of arm.

### Baseline / control

R4 config (`config/top_tagging_r4.yaml`) unchanged — rotor-attention is
gated off (`attention.rotor: none`), existing `sdp_attention` path runs
byte-identical to today. This is the third comparison point, not a third
worktree — see Run plan.

## Implementation

**New primitive** `su2eqnn/primitives/rotor.py`:

```python
def build_rotor(bivector_components: Tensor) -> Tensor:
    """bivector_components: (..., 3) -> rotor MV (..., 8), scalar+bivector only."""
    theta = bivector_components.norm(dim=-1)
    half = theta / 2
    # small-angle-safe sin(half)/theta
    coeff = torch.where(theta > 1e-4, torch.sin(half) / theta.clamp_min(1e-12),
                         0.5 - theta**2 / 48)
    rotor = torch.zeros(*bivector_components.shape[:-1], 8, ...)
    rotor[..., 0] = torch.cos(half)
    rotor[..., 4:7] = coeff.unsqueeze(-1) * bivector_components
    return rotor

def apply_rotor(x_mv: Tensor, rotor: Tensor) -> Tensor:
    """Sandwich x_mv by rotor: R x R~. x_mv, rotor: (..., 8) broadcastable."""
    return geometric_product(geometric_product(rotor, x_mv), reverse(rotor))
```

`wedge(a, b)` as a standalone function in the same module, extracted from
`embed_tagging_data`'s inline formula (used by both `embedding.py` and
R10a's generator, single source of truth).

**Wiring** — `su2eqnn/layers/attention/self_attention.py`: rotor
application is the multivector-stream analog of the existing
`pos_encoding` hook (which does scalar-stream RoPE today, line ~85-87).
Add symmetrically:

```python
if config.rotor_encoding:
    q_mv = self.rotor_encoding(q_mv, generator_input)
    k_mv = self.rotor_encoding(k_mv, generator_input)
```

`generator_input` = axes tensor (R10a) or per-particle scalar (R10b),
threaded through the same `additional_qk_features_*` plumbing already used
for axis channels — no new forward-signature surface needed on
`GATrBlock`/`SelfAttention` beyond what R4 already added.

**Config** (`su2eqnn/layers/attention/config.py`): add
`rotor_encoding: Optional[str] = None` (`"wedge"` | `"frozen"` | `None`)
and `rotor_num_axes: int = 1` to `SelfAttentionConfig`, mirroring
`pos_encoding`'s existing bool-gate pattern.

New configs: `config/top_tagging_r10a.yaml`, `config/top_tagging_r10b.yaml`
— copies of `top_tagging_r4.yaml` with `attention.rotor_encoding: wedge`
/ `frozen` added; same `data.path`, same 30-epoch/batch_size 32/lr 1e-3
mini training block as R4/R8/R9 for direct comparability.

## Testing

- Unit: `build_rotor`/`apply_rotor` round-trip — rotor of a zero bivector
  is identity (`apply_rotor(x, build_rotor(zeros)) == x`); rotor norm is 1
  (`R * reverse(R) == scalar 1`) for random bivectors, including near-zero
  (small-angle branch) and large-angle inputs.
- Equivariance test (pattern from `tests/test_equivariance.py`): rotate
  synthetic jet constituents + axes by a random Spin(3) rotor, confirm
  R10a's attention score is invariant to machine precision. **R10b is
  expected to fail this test** — assert the failure explicitly (bounds the
  claim, documents it's the known trade-off, not a bug) rather than
  omitting the check.
- Regression: with `rotor_encoding=None`, `SelfAttention` output must be
  bit-identical to current `main` (confirms the new hook is a true no-op
  when disabled).

## Run plan

Order: R10a mini first (cheap signal, same pattern as R4/R8/R9). Compare
against R1 (0.9850 full / mini baseline), R4 mini (0.9788), R8a mini
(0.9804), L-GATr mini (~0.9813, see `[[r7-patch-injection-results]]`).
Only run R10b if R10a shows separation from R4/R8a at mini scale — R10b
answers "does dropping the equivariance guarantee buy more," which is only
an interesting question if R10a already moved the needle.
5 seeds (0-4), mini dataset (`toptagging_mini_axes.npz`), same eval
protocol as R4/R8/R9.

## Worktrees

Three, in dependency order:

1. **`r10-rotor-primitive`** — `su2eqnn/primitives/rotor.py`,
   `wedge()` extraction, config field additions, self_attention.py wiring
   + no-op regression test. Blocking for 2 and 3.
2. **`r10a-wedge-basis`** — depends on (1). R10a generator
   (`c_k(s_i)` linear map), `config/top_tagging_r10a.yaml`, equivariance
   test, mini run.
3. **`r10b-frozen-basis`** — depends on (1), independent of (2). R10b
   generator (free `nn.Parameter` basis), `config/top_tagging_r10b.yaml`,
   documented-failure equivariance test, held pending R10a's mini signal
   per Run plan.

## Out of scope

- Full-dataset run for either arm — mini only, pending signal.
- Pairwise (non-per-token) rotor construction — rejected earlier in design
  discussion, O(N²) with no efficiency or equivariance advantage over the
  per-token route here.
- Canonicalization-based equivariance fix for R10b — rejected in favor of
  R10a's wedge-basis route, which needs no preprocessing step and has no
  degenerate-axis discontinuity risk. R5's canonicalization work is
  unrelated (Lorentz boost, not this spatial-rotor question) and untouched.
- Rotation-augmentation training for R10b — noted as a possible follow-up
  to close its equivariance gap empirically, not implemented in this pass.
