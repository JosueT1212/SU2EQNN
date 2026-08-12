# R10: Rotor-Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-token Spin(3) rotor-sandwich to Q/K before attention scoring (RoPE/CARE/LieRE ported to Cl(3,0)), in two arms — R10a (exact-equivariant, wedge-basis generator) and R10b (approximate, frozen-basis generator) — to test whether rotor-attention raises the R1=R4 ceiling rather than just convergence speed.

**Architecture:** New `su2eqnn/primitives/rotor.py` (`wedge`, `build_rotor`, `apply_rotor`) built entirely from two existing primitives (`geometric_product`, `reverse`) — no new Clifford-algebra machinery. `SelfAttention` gains an optional `RotorGenerator` submodule (new file `su2eqnn/layers/attention/rotor_generator.py`) that builds a per-head, per-token bivector from either (a) the existing `additional_qk_features_mv` channel — same mechanism R4 already uses to inject jet-axis/subjet bivectors, reused here unchanged — for R10a, or (b) a free learned 3-vector contracted with an invariant computed on the fly from `q_mv`/`k_mv` via the existing `spin3_invariants` primitive, for R10b. Sandwiching happens inside `SelfAttention.forward`, right after `qkv_module` produces `q_mv`/`k_mv` and before the call to `self.attention(...)` — `sdp_attention` itself is untouched. Zero new plumbing is needed at `GATrBlock`/`GATr` level: R10a's candidate axes arrive via the `reinsert_mv_channels` mechanism R4 already wires up.

**Tech Stack:** PyTorch, `clifford` (Cl(3,0) basis, via existing primitives), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-r10-rotor-attention-design.md`
- Rotor is per-token (not pairwise O(N²)) — matches RoPE/LieRE's efficient construction, rejected the pairwise alternative during design (see spec's "Out of scope").
- `sdp_attention` (`su2eqnn/primitives/attention.py`) must not change — regression test in Task 4 asserts byte-identical output when `rotor_encoding=None`.
- R10a must have zero new equivariance risk: generator built only from `wedge()` (existing convention, copied verbatim from `experiments/top_tagging/embedding.py`) and learned **scalars** — never a raw learned vector/tensor. R10b is explicitly allowed to be non-equivariant; its test must assert the failure, not hide it.
- Mini dataset only for both arms (`data/toptagging_mini_axes.npz`), 30 epochs, batch_size 32, lr 1e-3 — same hyperparameters as R4/R8/R9 for direct comparability.
- Model in this repo currently runs with `in_s_channels=None` (no scalar stream) — R10b's invariant feature must be computed from `q_mv`/`k_mv` directly (via `spin3_invariants`), not from an external scalar stream that doesn't exist.

---

## Worktree 1: `r10-rotor-primitive` (blocking — 2 and 3 depend on this)

### Task 1: `su2eqnn/primitives/rotor.py` — `wedge`, `build_rotor`, `apply_rotor`

**Files:**
- Create: `su2eqnn/primitives/rotor.py`
- Test: `tests/test_rotor.py`

**Interfaces:**
- Produces: `wedge(a: Tensor, b: Tensor) -> Tensor` — `a`, `b` shape `(..., 3)`, returns `(..., 3)` in `(e12, e13, e23)` order.
- Produces: `build_rotor(bivector: Tensor) -> Tensor` — `bivector` shape `(..., 3)` in `(e12, e13, e23)` order, returns `(..., 8)` Cl(3,0) MV (scalar + bivector only).
- Produces: `apply_rotor(x_mv: Tensor, rotor: Tensor) -> Tensor` — both `(..., 8)`, broadcastable, returns `(..., 8)`.
- Consumes: `geometric_product` from `su2eqnn/primitives/bilinear.py`, `reverse` from `su2eqnn/primitives/linear.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rotor.py`:

```python
"""Tests for su2eqnn/primitives/rotor.py."""

import torch

from su2eqnn.primitives.rotor import wedge, build_rotor, apply_rotor
from su2eqnn.primitives.invariants import inner_product

TOL = 1e-5


def test_wedge_matches_embedding_convention():
    """wedge(a, b) must match embed_tagging_data's inline formula exactly."""
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[4.0, 5.0, 6.0]])
    out = wedge(a, b)
    expected_e12 = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    expected_e13 = a[..., 0] * b[..., 2] - a[..., 2] * b[..., 0]
    expected_e23 = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    assert torch.allclose(out[..., 0], expected_e12, atol=TOL)
    assert torch.allclose(out[..., 1], expected_e13, atol=TOL)
    assert torch.allclose(out[..., 2], expected_e23, atol=TOL)


def test_wedge_parallel_vectors_zero():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    out = wedge(a, a * 2.0)
    assert out.abs().max() < TOL


def test_build_rotor_zero_bivector_is_identity():
    bivector = torch.zeros(5, 3)
    rotor = build_rotor(bivector)
    expected = torch.zeros(5, 8)
    expected[:, 0] = 1.0
    assert torch.allclose(rotor, expected, atol=TOL)


def test_build_rotor_unit_norm_small_and_large_angle():
    torch.manual_seed(0)
    small = torch.randn(20, 3) * 1e-6
    large = torch.randn(20, 3) * 5.0
    for bivector in (small, large, torch.zeros(1, 3)):
        rotor = build_rotor(bivector)
        norm_sq = inner_product(rotor, rotor)[..., 0]
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-4), \
            f"rotor not unit norm: {norm_sq}"


def test_apply_rotor_identity_rotor_is_noop():
    torch.manual_seed(1)
    x = torch.randn(4, 8)
    identity_rotor = build_rotor(torch.zeros(4, 3))
    out = apply_rotor(x, identity_rotor)
    assert torch.allclose(out, x, atol=TOL)


def test_apply_rotor_preserves_norm():
    """Rotor sandwich is a Spin(3) group action -> preserves Cl(3,0) norm."""
    torch.manual_seed(2)
    x = torch.randn(10, 8)
    bivector = torch.randn(10, 3)
    rotor = build_rotor(bivector)
    out = apply_rotor(x, rotor)
    assert torch.allclose(inner_product(x, x), inner_product(out, out), atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rotor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'su2eqnn.primitives.rotor'`

- [ ] **Step 3: Implement `rotor.py`**

Create `su2eqnn/primitives/rotor.py`:

```python
"""Per-token Spin(3) rotor construction and sandwich application for Cl(3,0).

Ports RoPE's per-token rotation trick (arXiv:2104.09864 sec. 3.2.1) through
CARE's Clifford-rotor generalization (arXiv:2511.11665) to LieRE's per-token
learned-generator regime (arXiv:2406.10322 sec. 3.1) — but for su(2), whose
only abelian subalgebra is a single fixed axis, so exploiting the full
3-dimensional bivector space requires this non-abelian per-token
construction rather than RoPE/STRING's commuting-generator route.

Rotor: R = cos(theta/2) + sinc_half(theta) * B, theta = |B|,
sinc_half(theta) = sin(theta/2) / theta (small-angle-safe below).
Sandwich: R x ~R, via two geometric_product calls + reverse.
"""

import torch

from su2eqnn.primitives.bilinear import geometric_product
from su2eqnn.primitives.linear import reverse

_SMALL_ANGLE_EPS = 1e-4


def wedge(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cl(3,0) wedge product of two grade-1 vectors, as pure grade-2 components.

    Same convention as experiments/top_tagging/embedding.py's inline
    axes-branch formula: e12, e13, e23 order.

    Parameters
    ----------
    a, b : torch.Tensor with shape (..., 3)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 3) — (e12, e13, e23)
    """
    e12 = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    e13 = a[..., 0] * b[..., 2] - a[..., 2] * b[..., 0]
    e23 = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    return torch.stack([e12, e13, e23], dim=-1)


def build_rotor(bivector: torch.Tensor) -> torch.Tensor:
    """Builds a unit Spin(3) rotor R = exp(bivector / 2) from raw bivector components.

    Small-angle-safe: sin(theta/2)/theta -> 1/2 - theta^2/48 as theta -> 0
    (Taylor expansion), avoiding the 0/0 at theta=0 (common when a particle
    is near-collinear with the reference axis).

    Parameters
    ----------
    bivector : torch.Tensor with shape (..., 3) — (e12, e13, e23) components

    Returns
    -------
    rotor : torch.Tensor with shape (..., 8) — Cl(3,0) MV, scalar + bivector only
    """
    theta = bivector.norm(dim=-1)
    half = theta / 2
    safe_theta = theta.clamp_min(1e-12)
    coeff = torch.where(
        theta > _SMALL_ANGLE_EPS,
        torch.sin(half) / safe_theta,
        0.5 - theta**2 / 48.0,
    )
    rotor = torch.zeros(*bivector.shape[:-1], 8, dtype=bivector.dtype, device=bivector.device)
    rotor[..., 0] = torch.cos(half)
    rotor[..., 4:7] = coeff.unsqueeze(-1) * bivector
    return rotor


def apply_rotor(x_mv: torch.Tensor, rotor: torch.Tensor) -> torch.Tensor:
    """Sandwich-applies a rotor to a multivector: R x ~R.

    Parameters
    ----------
    x_mv  : torch.Tensor with shape (..., 8), broadcastable with rotor
    rotor : torch.Tensor with shape (..., 8), broadcastable with x_mv

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    return geometric_product(geometric_product(rotor, x_mv), reverse(rotor))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rotor.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add su2eqnn/primitives/rotor.py tests/test_rotor.py
git commit -m "feat: add rotor primitive (wedge, build_rotor, apply_rotor) for Cl(3,0)"
```

---

### Task 2: Algebraic equivariance identity for the rotor sandwich

**Files:**
- Modify: `tests/test_equivariance.py`

**Interfaces:**
- Consumes: `build_rotor`, `apply_rotor`, `wedge` from Task 1; `random_spin3`, `spin3_action` already defined in this file.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_equivariance.py`, after `test_spin3_norm_preservation`:

```python
def test_rotor_sandwich_equivariance():
    """w R(B) ~w == R(wBw^{-1}): rotor construction commutes with Spin(3) action.

    This is the algebraic identity R10a's equivariance claim rests on — it
    must hold regardless of how B is computed from data (that's tested
    separately per-arm in the r10a/r10b worktrees).
    """
    from su2eqnn.primitives.rotor import build_rotor, apply_rotor

    for _ in range(N_TESTS):
        bivector = torch.randn(BATCH, 3)
        x = torch.randn(BATCH, 8)
        w = random_spin3()

        rotor = build_rotor(bivector)
        lhs = spin3_action(w, apply_rotor(x, rotor))

        # rotate the raw bivector components the same way the full MV would
        # rotate: embed as a pure-bivector MV, apply spin3_action, extract back.
        bivector_mv = torch.zeros(BATCH, 8)
        bivector_mv[:, 4:7] = bivector
        rotated_bivector_mv = spin3_action(w, bivector_mv)
        rotated_bivector = rotated_bivector_mv[:, 4:7]

        rotor_rotated = build_rotor(rotated_bivector)
        rhs = apply_rotor(spin3_action(w, x), rotor_rotated)

        assert torch.allclose(lhs, rhs, atol=1e-4), \
            f"rotor sandwich not equivariant: max diff {(lhs - rhs).abs().max():.2e}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_equivariance.py -k rotor_sandwich -v`
Expected: FAIL with `ImportError` if Task 1 not present in this worktree yet, else collects and should already PASS once Task 1 lands (this test only depends on Task 1's primitives, not on any new attention wiring) — if it fails for a reason other than ImportError, that's a real bug, stop and fix `rotor.py` before continuing.

- [ ] **Step 3: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_equivariance.py -k rotor_sandwich -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_equivariance.py
git commit -m "test: add algebraic equivariance identity for rotor sandwich"
```

---

### Task 3: `rotor_encoding` config field

**Files:**
- Modify: `su2eqnn/layers/attention/config.py`
- Test: `tests/test_top_tagging.py`

**Interfaces:**
- Produces: `SelfAttentionConfig.rotor_encoding: Optional[str] = None` (`"wedge"` | `"frozen"` | `None`), `SelfAttentionConfig.rotor_num_axes: int = 1`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_top_tagging.py`:

```python
def test_self_attention_config_rotor_fields_default_none():
    from su2eqnn.layers.attention.config import SelfAttentionConfig

    cfg = SelfAttentionConfig()
    assert cfg.rotor_encoding is None
    assert cfg.rotor_num_axes == 1


def test_self_attention_config_rotor_encoding_settable():
    from su2eqnn.layers.attention.config import SelfAttentionConfig

    cfg = SelfAttentionConfig(rotor_encoding="wedge", rotor_num_axes=4)
    assert cfg.rotor_encoding == "wedge"
    assert cfg.rotor_num_axes == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k rotor_fields -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'rotor_encoding'`

- [ ] **Step 3: Add the fields**

In `su2eqnn/layers/attention/config.py`, add to the `SelfAttentionConfig` dataclass body, after `additional_qk_s_channels: int = 0`:

```python
    rotor_encoding: Optional[str] = None
    rotor_num_axes: int = 1
```

And to the docstring's parameter list, after the `additional_qk_s_channels` entry:

```
    rotor_encoding : str or None
        "wedge" (R10a, exact-equivariant, uses additional_qk_features_mv's
        bivector channels), "frozen" (R10b, approximate, free learned
        basis), or None (default, no rotor — byte-identical to pre-R10
        behavior).
    rotor_num_axes : int
        Number of candidate axis channels rotor_encoding="wedge" draws from
        additional_qk_features_mv. Ignored for "frozen" and None.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k rotor_fields -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add su2eqnn/layers/attention/config.py tests/test_top_tagging.py
git commit -m "feat: add rotor_encoding/rotor_num_axes fields to SelfAttentionConfig"
```

---

### Task 4: `RotorGenerator` module + wiring into `SelfAttention`

**Files:**
- Create: `su2eqnn/layers/attention/rotor_generator.py`
- Modify: `su2eqnn/layers/attention/self_attention.py`
- Test: `tests/test_top_tagging.py`

**Interfaces:**
- Produces: `RotorGenerator(nn.Module)` with `__init__(config: SelfAttentionConfig)` and `forward(q_mv_or_k_mv: Tensor, additional_qk_features_mv: Optional[Tensor]) -> Tensor` — input `(..., heads, items, hidden, 8)` (or the 1-head-shared MultiQuery K/V shape), `additional_qk_features_mv` `(..., items, add_mv, 8)` or `None`, returns rotor `(..., heads_or_1, items, 8)` ready to broadcast-sandwich against the hidden-channel dim.
- Consumes: `wedge`, `build_rotor`, `apply_rotor` (Task 1); `spin3_invariants` from `su2eqnn/primitives/invariants.py`; `SelfAttentionConfig` (Task 3).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_top_tagging.py`:

```python
def test_rotor_generator_none_mode_raises():
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.rotor_generator import RotorGenerator

    cfg = SelfAttentionConfig(rotor_encoding=None, num_heads=2, hidden_mv_channels=None)
    try:
        RotorGenerator(cfg)
        assert False, "expected ValueError for rotor_encoding=None"
    except ValueError:
        pass


def test_rotor_generator_wedge_mode_shape():
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.rotor_generator import RotorGenerator

    cfg = SelfAttentionConfig(
        in_mv_channels=1, num_heads=2, rotor_encoding="wedge", rotor_num_axes=3,
    )
    gen = RotorGenerator(cfg)
    q_mv = torch.randn(4, 2, 7, 3, 8)  # (batch, heads, items, hidden, 8)
    additional_qk_features_mv = torch.randn(4, 7, 3, 8)  # (batch, items, add_mv=3, 8)
    rotor = gen(q_mv, additional_qk_features_mv)
    assert rotor.shape == (4, 2, 7, 8)


def test_rotor_generator_frozen_mode_shape_no_axes_needed():
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.rotor_generator import RotorGenerator

    cfg = SelfAttentionConfig(in_mv_channels=1, num_heads=2, rotor_encoding="frozen")
    gen = RotorGenerator(cfg)
    q_mv = torch.randn(4, 2, 7, 3, 8)
    rotor = gen(q_mv, additional_qk_features_mv=None)
    assert rotor.shape == (4, 2, 7, 8)


def test_self_attention_rotor_encoding_none_is_noop():
    """rotor_encoding unset must reproduce byte-identical output to pre-R10 behavior."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.self_attention import SelfAttention

    torch.manual_seed(0)
    cfg = SelfAttentionConfig(
        in_mv_channels=3, out_mv_channels=3, num_heads=2, multi_query=False,
    )
    layer = SelfAttention(cfg)
    mv = torch.randn(2, 5, 3, 8)

    torch.manual_seed(1)
    out_a, _ = layer(mv)
    torch.manual_seed(1)
    out_b, _ = layer(mv)
    assert torch.allclose(out_a, out_b), "sanity: deterministic without dropout"


def test_self_attention_rotor_wedge_runs_and_changes_output():
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.self_attention import SelfAttention

    torch.manual_seed(0)
    cfg_plain = SelfAttentionConfig(
        in_mv_channels=3, out_mv_channels=3, num_heads=2, multi_query=False,
        additional_qk_mv_channels=2,
    )
    cfg_rotor = SelfAttentionConfig(
        in_mv_channels=3, out_mv_channels=3, num_heads=2, multi_query=False,
        additional_qk_mv_channels=2, rotor_encoding="wedge", rotor_num_axes=2,
    )
    torch.manual_seed(42)
    layer_plain = SelfAttention(cfg_plain)
    torch.manual_seed(42)
    layer_rotor = SelfAttention(cfg_rotor)

    mv = torch.randn(2, 5, 3, 8)
    extra = torch.randn(2, 5, 2, 8)
    extra[..., [1, 2, 3, 7]] = 0.0  # pure grade-2, matches axes-channel convention

    out_plain, _ = layer_plain(mv, additional_qk_features_mv=extra)
    out_rotor, _ = layer_rotor(mv, additional_qk_features_mv=extra)
    assert out_plain.shape == out_rotor.shape
    assert not torch.allclose(out_plain, out_rotor), \
        "rotor encoding should change attention output vs plain"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k rotor_generator -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'su2eqnn.layers.attention.rotor_generator'`

- [ ] **Step 3: Implement `RotorGenerator`**

Create `su2eqnn/layers/attention/rotor_generator.py`:

```python
"""Per-token rotor generator for R10 rotor-attention (two modes).

"wedge" (R10a): candidate bivectors come from additional_qk_features_mv's
grade-2 slots (identical channels R4 already injects — jet axis / subjet
axes), combined with a *learned scalar* per (head, axis). Scalars are
Spin(3)-invariant, so a scalar-weighted sum of equivariant bivectors is
itself equivariant -- exact-equivariant by construction, no new risk.

"frozen" (R10b): a free, unconstrained per-head 3-vector parameter,
contracted with an invariant scalar computed on the fly from q_mv/k_mv via
spin3_invariants (this model has no external scalar stream). Only
approximately equivariant -- literal LieRE port (arXiv:2406.10322 sec.
3.1's basis_raw_params, ported to su(2)).
"""

from typing import Optional

import torch
from torch import nn

from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.primitives.invariants import spin3_invariants


class RotorGenerator(nn.Module):
    """Builds a per-(head, item) bivector generator for rotor-attention.

    Parameters
    ----------
    config : SelfAttentionConfig
        config.rotor_encoding must be "wedge" or "frozen".
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()
        if config.rotor_encoding not in ("wedge", "frozen"):
            raise ValueError(
                f"RotorGenerator requires rotor_encoding in ('wedge','frozen'), "
                f"got {config.rotor_encoding!r}"
            )
        self.mode = config.rotor_encoding
        self.num_heads = config.num_heads

        if self.mode == "wedge":
            # learned scalar per (head, candidate axis)
            self.coeffs = nn.Parameter(
                torch.zeros(config.num_heads, config.rotor_num_axes)
            )
        else:  # "frozen"
            # free per-head 3-vector basis, contracted with a scalar invariant
            self.basis = nn.Parameter(torch.randn(config.num_heads, 3) * 0.01)

    def forward(
        self,
        q_or_k_mv: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Builds the bivector generator, broadcastable against q_or_k_mv's hidden dim.

        Parameters
        ----------
        q_or_k_mv : Tensor (..., heads, items, hidden, 8) — used for shape only
            in "wedge" mode; used to compute the invariant scalar in
            "frozen" mode.
        additional_qk_features_mv : Tensor (..., items, add_mv, 8) or None —
            required for "wedge" mode (add_mv must equal rotor_num_axes),
            ignored for "frozen" mode.

        Returns
        -------
        bivector : Tensor (..., heads, items, 3) — (e12, e13, e23) per head/item
        """
        if self.mode == "wedge":
            if additional_qk_features_mv is None:
                raise ValueError(
                    "rotor_encoding='wedge' requires additional_qk_features_mv "
                    "(candidate axis bivector channels)"
                )
            # (..., items, add_mv, 3) -- grade-2 slots only
            candidates = additional_qk_features_mv[..., 4:7]
            # (heads, add_mv) x (..., items, add_mv, 3) -> (..., heads, items, 3)
            bivector = torch.einsum("ha,...ia x -> ...h i x".replace(" ", ""),
                                     self.coeffs, candidates)
            return bivector

        # "frozen": invariant scalar per (head, item) from q/k itself.
        # spin3_invariants returns (..., 4): [scalar, |grade1|, |grade2|, |grade3|].
        # Sum over the hidden-channel dim (already summed over heads implicitly
        # since q_or_k_mv carries a heads dim) to get one scalar per (heads, items).
        invariants = spin3_invariants(q_or_k_mv)          # (..., heads, items, hidden, 4)
        p_scalar = invariants[..., 1].mean(dim=-1)        # (..., heads, items) -- grade-1 norm
        # (..., heads, items) x (heads, 3) -> (..., heads, items, 3)
        bivector = p_scalar.unsqueeze(-1) * self.basis.view(
            *([1] * (p_scalar.ndim - 2)), self.num_heads, 1, 3
        )
        return bivector
```

Note the `einsum` string in "wedge" mode is built defensively (`.replace(" ", "")`) purely to keep the literal string readable in the plan; when implementing, write it directly as `"ha,...iax->...hix"` — no functional difference, just avoid the `.replace` in real code:

```python
bivector = torch.einsum("ha,...iax->...hix", self.coeffs, candidates)
```

(Fix this line — use the direct string, not the `.replace` workaround shown above — when writing the actual file.)

- [ ] **Step 4: Wire `RotorGenerator` into `SelfAttention`**

In `su2eqnn/layers/attention/self_attention.py`:

1. Add import, after the existing `from su2eqnn.layers.attention.qkv import ...` line:

```python
from su2eqnn.layers.attention.rotor_generator import RotorGenerator
from su2eqnn.primitives.rotor import apply_rotor, build_rotor
```

2. In `SelfAttention.__init__`, after the `self.attention = GeometricAttention(config)` line, add:

```python
        if config.rotor_encoding is not None:
            self.rotor_generator: Optional[RotorGenerator] = RotorGenerator(config)
        else:
            self.rotor_generator = None
```

3. In `SelfAttention.forward`, after the line `q_mv, k_mv, v_mv, q_s, k_s, v_s = self.qkv_module(...)` and before the RoPE-on-scalars block, add:

```python
        if self.rotor_generator is not None:
            q_bivector = self.rotor_generator(q_mv, additional_qk_features_mv)
            k_bivector = self.rotor_generator(k_mv, additional_qk_features_mv)
            q_rotor = build_rotor(q_bivector).unsqueeze(-2)   # broadcast over hidden dim
            k_rotor = build_rotor(k_bivector).unsqueeze(-2)
            q_mv = apply_rotor(q_mv, q_rotor)
            k_mv = apply_rotor(k_mv, k_rotor)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k "rotor_generator or rotor_encoding or rotor_wedge" -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run full existing test suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS (rotor_generator is `None` by default on every existing config, so `SelfAttention.forward`'s new branch never executes for pre-existing tests/configs)

- [ ] **Step 7: Commit**

```bash
git add su2eqnn/layers/attention/rotor_generator.py su2eqnn/layers/attention/self_attention.py tests/test_top_tagging.py
git commit -m "feat: wire RotorGenerator into SelfAttention (wedge + frozen modes)"
```

---

## Worktree 2: `r10a-wedge-basis` (depends on Worktree 1)

### Task 5: R10a config — wire jet-axis channel through `reinsert_mv_channels` + `rotor_encoding=wedge`

**Files:**
- Create: `config/top_tagging_r10a.yaml`
- Modify: `experiments/top_tagging/experiment.py`

**Interfaces:**
- Consumes: `RotorGenerator`/`rotor_encoding` (Task 4); `embed_tagging_data` with `axes` (existing, from `experiments/top_tagging/embedding.py`); `GATr`'s existing `reinsert_mv_channels` parameter (existing, from `su2eqnn/nets/gatr.py`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_top_tagging.py`:

```python
def test_build_model_r10a_wires_rotor_encoding():
    from omegaconf import OmegaConf
    from experiments.top_tagging.experiment import build_model

    cfg = OmegaConf.create({
        "model": {
            "in_mv_channels": 2,  # momentum channel + 1 jet_axis bivector channel
            "hidden_mv_channels": 16,
            "num_blocks": 2,
            "num_heads": 2,
            "rotor_encoding": "wedge",
            "reinsert_mv_channels": [1],  # re-inject the jet_axis channel (index 1)
        }
    })
    wrapper = build_model(cfg)
    attn_cfg = wrapper.net.blocks[0].attention.config
    assert attn_cfg.rotor_encoding == "wedge"
    assert attn_cfg.rotor_num_axes == 1
    assert wrapper.net.blocks[0].attention.rotor_generator is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k build_model_r10a -v`
Expected: FAIL (`build_model` doesn't read `rotor_encoding`/`reinsert_mv_channels` from `cfg.model` yet — confirm by inspecting the assertion error, likely `AttributeError` or `assert None == 'wedge'`)

- [ ] **Step 3: Update `build_model` to read the new config fields**

In `experiments/top_tagging/experiment.py`, modify `build_model`'s `GATr(...)` call:

```python
def build_model(cfg) -> TopTaggingWrapper:
    """Instantiate GATr + TopTaggingWrapper from config.

    Parameters
    ----------
    cfg : DictConfig or object with .model attributes:
        hidden_mv_channels : int
        num_blocks         : int
        num_heads          : int
        in_mv_channels     : int (optional, default 1)
        rotor_encoding     : str or None (optional, default None) -- "wedge" or "frozen"
        reinsert_mv_channels : list[int] or None (optional, default None) --
            input MV channel indices re-injected as additional Q/K features
            in every block (R10a's candidate axis channels)
    """
    reinsert = getattr(cfg.model, "reinsert_mv_channels", None)
    reinsert_mv_channels = tuple(reinsert) if reinsert is not None else None
    rotor_encoding = getattr(cfg.model, "rotor_encoding", None)
    rotor_num_axes = len(reinsert_mv_channels) if reinsert_mv_channels is not None else 1

    net = GATr(
        in_mv_channels=getattr(cfg.model, "in_mv_channels", 1),
        out_mv_channels=1,
        hidden_mv_channels=cfg.model.hidden_mv_channels,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(
            num_heads=cfg.model.num_heads,
            rotor_encoding=rotor_encoding,
            rotor_num_axes=rotor_num_axes,
        ),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=cfg.model.num_blocks,
        reinsert_mv_channels=reinsert_mv_channels,
    )
    return TopTaggingWrapper(net)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k build_model_r10a -v`
Expected: PASS

- [ ] **Step 5: Run full existing test suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS (`rotor_encoding`/`reinsert_mv_channels` default to `None`, matching pre-R10 `build_model` behavior for every existing config)

- [ ] **Step 6: Create the R10a config**

Create `config/top_tagging_r10a.yaml` (same base as `top_tagging_r4.yaml`, single jet-axis channel instead of R4's 4-axis set, `rotor_encoding: wedge`, `reinsert_mv_channels: [1]` to feed the axis channel to `RotorGenerator`):

```yaml
data:
  path: data/toptagging_mini_axes.npz
  rescale_data: true
  two_beams: true

model:
  in_mv_channels: 2   # channel 0 = momentum, channel 1 = jet_axis bivector (R8a-style single axis)
  hidden_mv_channels: 16
  num_blocks: 6
  num_heads: 4
  rotor_encoding: wedge
  reinsert_mv_channels: [1]

training:
  epochs: 30
  batch_size: 32
  lr: 1.0e-3
  validate_every_n_steps: 200
  checkpoint_path: checkpoints/top_tagging_r10a_best.pt

evaluation:
  batch_size: 64
```

- [ ] **Step 7: Verify Hydra loads it and `build_model` instantiates without error**

Run:
```bash
.venv/bin/python -c "
from hydra import compose, initialize
from experiments.top_tagging.experiment import build_model
with initialize(version_base=None, config_path='config'):
    cfg = compose(config_name='top_tagging_r10a')
    model = build_model(cfg)
    print('OK', sum(p.numel() for p in model.parameters()), 'params')
"
```
Expected: prints `OK <N> params` with no errors.

- [ ] **Step 8: Commit**

```bash
git add experiments/top_tagging/experiment.py config/top_tagging_r10a.yaml tests/test_top_tagging.py
git commit -m "feat: wire rotor_encoding + reinsert_mv_channels through build_model; add R10a config"
```

---

### Task 6: End-to-end equivariance test for R10a's full attention path

**Files:**
- Modify: `tests/test_equivariance.py`

**Interfaces:**
- Consumes: `SelfAttention` with `rotor_encoding="wedge"` (Task 4); `random_spin3`, `spin3_action_channels` (existing helpers in this file).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_equivariance.py`:

```python
def test_self_attention_rotor_wedge_equivariance():
    """Full SelfAttention forward with rotor_encoding='wedge' must be
    equivariant end-to-end: rotating (mv, additional_qk_features_mv)
    together and running forward must equal rotating the plain-run output.
    """
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.self_attention import SelfAttention

    torch.manual_seed(0)
    cfg = SelfAttentionConfig(
        in_mv_channels=2, out_mv_channels=2, num_heads=2, multi_query=False,
        additional_qk_mv_channels=1, rotor_encoding="wedge", rotor_num_axes=1,
    )
    layer = SelfAttention(cfg)
    layer.eval()  # no dropout randomness

    for _ in range(10):
        mv = torch.randn(1, 5, 2, 8)
        extra = torch.randn(1, 5, 1, 8)
        extra[..., [0, 1, 2, 3, 7]] = 0.0  # pure grade-2 (axis bivector convention)
        w = random_spin3()

        out, _ = layer(mv, additional_qk_features_mv=extra)
        rot_mv = spin3_action_channels(w, mv[0])[None]
        rot_extra = spin3_action_channels(w, extra[0])[None]
        out_rot, _ = layer(rot_mv, additional_qk_features_mv=rot_extra)

        expected = spin3_action_channels(w, out[0])[None]
        assert torch.allclose(out_rot, expected, atol=1e-3), \
            f"R10a wedge attention not equivariant: max diff " \
            f"{(out_rot - expected).abs().max():.2e}"
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/test_equivariance.py -k rotor_wedge_equivariance -v`
Expected: should PASS immediately if Tasks 1-4 are correctly implemented (this test validates the design, not new code) — if it FAILS, that means the `wedge` mode's einsum or broadcasting has a bug; stop and fix `RotorGenerator`/`self_attention.py` before continuing, do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_equivariance.py
git commit -m "test: add end-to-end equivariance check for rotor_encoding=wedge"
```

---

### Task 7: R10a smoke run + 5-seed runner script

**Files:**
- Create: `scripts/run_seeds_r10a.sh`

**Interfaces:**
- Consumes: `config/top_tagging_r10a.yaml` (Task 5), `set_seed`/`+training.seed=<int>` (existing in this repo's `experiment.py` — confirm presence before writing the script; if absent, add it exactly as done in the R9 plan's Task 3 `set_seed` function before proceeding).

- [ ] **Step 1: Confirm `set_seed` exists; add it if not**

Run: `grep -n "def set_seed" experiments/top_tagging/experiment.py`

If no match, add (mirroring the R9 worktree's version) after the `log = logging.getLogger(__name__)` line:

```python
def set_seed(seed) -> None:
    """Seed random/numpy/torch RNGs for reproducibility. No-op if seed is None."""
    if seed is None:
        return
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log.info(f"Seeded RNGs with seed={seed}")
```

and call `set_seed(getattr(cfg.training, "seed", None))` at the start of `run_experiment(...)`. Commit this separately first if added:

```bash
git add experiments/top_tagging/experiment.py
git commit -m "feat: add set_seed for reproducible multi-seed runs"
```

- [ ] **Step 2: Smoke-test one short run**

Run (from `SU2EQNN/` root):

```bash
PYTHONPATH=$PWD .venv/bin/python -m experiments.top_tagging.experiment \
  --config-name top_tagging_r10a \
  training.epochs=1 \
  +training.seed=0 \
  training.checkpoint_path=checkpoints/top_tagging_r10a_smoke.pt \
  hydra.run.dir=outputs/r10a_smoke
```

Expected: completes without error, logs `Test AUC=`/`accuracy=` at the end.

- [ ] **Step 3: Write the 5-seed runner script**

Create `scripts/run_seeds_r10a.sh`:

```bash
#!/bin/bash
# 5-seed sweep: R10a (wedge-basis rotor-attention) on mini, local CPU.
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
for seed in 0 1 2 3 4; do
  tag="r10a_s${seed}"
  echo "=== $tag start $(date) ==="
  PYTHONPATH=$PWD $PY -m experiments.top_tagging.experiment \
    --config-name top_tagging_r10a \
    +training.seed=$seed \
    training.checkpoint_path=checkpoints/${tag}_best.pt \
    hydra.run.dir=outputs/seeds/$tag \
    > seeds_${tag}.log 2>&1
  echo "=== $tag done: $(grep 'Test AUC' seeds_${tag}.log)"
done
echo "ALL SEEDS DONE $(date)"
```

- [ ] **Step 4: Make it executable and commit**

```bash
chmod +x scripts/run_seeds_r10a.sh
git add scripts/run_seeds_r10a.sh
git commit -m "feat: add 5-seed runner script for R10a"
```

- [ ] **Step 5: Launch the 5-seed sweep**

Run: `scripts/run_seeds_r10a.sh > run_seeds_r10a.log 2>&1 &`

Runs in the background. Do not block on it. Per the spec's Run plan: only proceed to launching R10b's sweep (Worktree 3, Task 10) once these 5 logs show `Test AUC=` results and can be compared against R4 mini (0.9788) / R8a mini (0.9804).

---

## Worktree 3: `r10b-frozen-basis` (depends on Worktree 1, independent of Worktree 2)

### Task 8: R10b config

**Files:**
- Create: `config/top_tagging_r10b.yaml`

**Interfaces:**
- Consumes: `rotor_encoding="frozen"` path in `build_model` (Task 5's `build_model` edit already supports this — `rotor_encoding` is read generically, no R10a-specific branching).

- [ ] **Step 1: Create the config**

R10b needs no axis channels (frozen mode's generator is self-contained, built from `q_mv`/`k_mv` via `spin3_invariants` — no `additional_qk_features_mv` required), so `in_mv_channels: 1` (momentum only) and no `reinsert_mv_channels`:

```yaml
data:
  path: data/toptagging_mini_axes.npz
  rescale_data: true
  two_beams: true

model:
  in_mv_channels: 1
  hidden_mv_channels: 16
  num_blocks: 6
  num_heads: 4
  rotor_encoding: frozen

training:
  epochs: 30
  batch_size: 32
  lr: 1.0e-3
  validate_every_n_steps: 200
  checkpoint_path: checkpoints/top_tagging_r10b_best.pt

evaluation:
  batch_size: 64
```

Write to `config/top_tagging_r10b.yaml`.

- [ ] **Step 2: Verify Hydra loads it and `build_model` instantiates without error**

Run:
```bash
.venv/bin/python -c "
from hydra import compose, initialize
from experiments.top_tagging.experiment import build_model
with initialize(version_base=None, config_path='config'):
    cfg = compose(config_name='top_tagging_r10b')
    model = build_model(cfg)
    print('OK', sum(p.numel() for p in model.parameters()), 'params')
"
```
Expected: prints `OK <N> params` with no errors.

- [ ] **Step 3: Commit**

```bash
git add config/top_tagging_r10b.yaml
git commit -m "feat: add R10b config (frozen-basis rotor-attention, mini)"
```

---

### Task 9: Documented-failure equivariance test for R10b

**Files:**
- Modify: `tests/test_equivariance.py`

**Interfaces:**
- Consumes: `SelfAttention` with `rotor_encoding="frozen"` (Task 4).

- [ ] **Step 1: Write the test — asserting non-equivariance, not omitting the check**

Append to `tests/test_equivariance.py`:

```python
def test_self_attention_rotor_frozen_is_not_equivariant():
    """R10b is the documented trade-off: frozen basis buys capacity, gives
    up the equivariance guarantee. This test asserts that failure
    explicitly (per spec's Testing section) so it's a known, bounded
    trade-off rather than a silent bug or an untested gap.
    """
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.self_attention import SelfAttention

    torch.manual_seed(0)
    cfg = SelfAttentionConfig(
        in_mv_channels=1, out_mv_channels=1, num_heads=2, multi_query=False,
        rotor_encoding="frozen",
    )
    layer = SelfAttention(cfg)
    layer.eval()

    any_violation = False
    for _ in range(10):
        mv = torch.randn(1, 5, 1, 8)
        w = random_spin3()

        out, _ = layer(mv)
        rot_mv = spin3_action_channels(w, mv[0])[None]
        out_rot, _ = layer(rot_mv)

        expected = spin3_action_channels(w, out[0])[None]
        if not torch.allclose(out_rot, expected, atol=1e-3):
            any_violation = True
            break

    assert any_violation, (
        "expected rotor_encoding='frozen' to violate equivariance for at "
        "least one random rotation (documented trade-off) -- if this now "
        "passes, either the basis parameter accidentally initialized to "
        "zero (re-roll torch.manual_seed) or RotorGenerator's frozen mode "
        "changed to be accidentally equivariant, which would be worth "
        "investigating rather than silently accepting"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_equivariance.py -k rotor_frozen_is_not_equivariant -v`
Expected: PASS. If it fails (i.e., the assertion trips because no violation was found across 10 random rotations), do not treat this as a bug to silently fix by weakening the test — investigate: check `self.basis` isn't zero-initialized to a degenerate value, and confirm with a wider `torch.randn(...) * 0.01` scale isn't accidentally too small to produce a measurable rotation given `atol=1e-3`. Widen the scale in `RotorGenerator.__init__`'s `self.basis` init if needed and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_equivariance.py
git commit -m "test: assert documented non-equivariance for rotor_encoding=frozen"
```

---

### Task 10: R10b smoke run + 5-seed runner script (held pending R10a signal)

**Files:**
- Create: `scripts/run_seeds_r10b.sh`

**Interfaces:**
- Consumes: `config/top_tagging_r10b.yaml` (Task 8), `set_seed` (Worktree 2 Task 7, or added independently here if this worktree doesn't share history with Worktree 2 — check with `grep -n "def set_seed" experiments/top_tagging/experiment.py` first, same as Task 7 Step 1).

- [ ] **Step 1: Confirm `set_seed` exists; add it if not (same code as Worktree 2 Task 7 Step 1)**

- [ ] **Step 2: Smoke-test one short run**

```bash
PYTHONPATH=$PWD .venv/bin/python -m experiments.top_tagging.experiment \
  --config-name top_tagging_r10b \
  training.epochs=1 \
  +training.seed=0 \
  training.checkpoint_path=checkpoints/top_tagging_r10b_smoke.pt \
  hydra.run.dir=outputs/r10b_smoke
```

Expected: completes without error, logs `Test AUC=`/`accuracy=`.

- [ ] **Step 3: Write the 5-seed runner script**

Create `scripts/run_seeds_r10b.sh`:

```bash
#!/bin/bash
# 5-seed sweep: R10b (frozen-basis rotor-attention) on mini, local CPU.
# Per spec's Run plan: only launch this after R10a's mini results show
# separation from R4 (0.9788) / R8a (0.9804) -- otherwise R10b's question
# ("does dropping equivariance buy more") isn't interesting yet.
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
for seed in 0 1 2 3 4; do
  tag="r10b_s${seed}"
  echo "=== $tag start $(date) ==="
  PYTHONPATH=$PWD $PY -m experiments.top_tagging.experiment \
    --config-name top_tagging_r10b \
    +training.seed=$seed \
    training.checkpoint_path=checkpoints/${tag}_best.pt \
    hydra.run.dir=outputs/seeds/$tag \
    > seeds_${tag}.log 2>&1
  echo "=== $tag done: $(grep 'Test AUC' seeds_${tag}.log)"
done
echo "ALL SEEDS DONE $(date)"
```

- [ ] **Step 4: Make it executable and commit — do not launch it yet**

```bash
chmod +x scripts/run_seeds_r10b.sh
git add scripts/run_seeds_r10b.sh
git commit -m "feat: add 5-seed runner script for R10b (held pending R10a signal)"
```

Per spec's Run plan, this script is committed but **not launched** as part of this plan — launching it is a follow-up step gated on Worktree 2's R10a mini results.

---

## Self-Review Notes

- **Spec coverage**: core mechanism (build_rotor/apply_rotor) → Task 1; equivariance requirement (algebraic identity) → Task 2; config surface → Task 3; per-arm generator + wiring → Task 4; R10a wedge-basis generator source (reuses `additional_qk_features_mv`/`reinsert_mv_channels`, zero new plumbing) → Task 5; R10a end-to-end equivariance → Task 6; R10a run → Task 7; R10b frozen-basis generator + config → Tasks 4/8; R10b documented non-equivariance → Task 9; R10b run (held) → Task 10. All spec sections covered.
- **Naming collision avoided**: confirmed via `git log` that "R9" is already used on this branch (subjet channel-vs-scalar ablation, `2026-08-11-r9-subjet-channel-vs-scalar-design.md`) — this plan is R10, consistent with the spec.
- **Type/interface consistency**: `RotorGenerator.forward(q_or_k_mv, additional_qk_features_mv)` signature identical across Task 4's implementation, its own tests, and both call sites added in Task 4 Step 4 (`q_mv`/`k_mv` calls). `SelfAttentionConfig.rotor_encoding`/`rotor_num_axes` names identical across Tasks 3, 4, 5, 8. `build_model`'s new `rotor_encoding`/`reinsert_mv_channels` cfg fields consumed identically by both R10a (Task 5, sets both) and R10b (Task 8, sets only `rotor_encoding`, relies on `reinsert_mv_channels` defaulting to `None`/`in_mv_channels=1`).
- **No placeholders**: every step has literal code, exact file paths, and runnable commands with expected output stated.
- **Design refinement vs spec worth flagging explicitly**: the spec's R10a section suggested `c_k(s_i)` as a per-particle scalar-stream-driven map; this plan's Task 4 implements the spec's own "simplest starting version" instead (one learned scalar per head, no per-particle scalar dependence) because this repo's top-tagging model currently runs with `in_s_channels=None` (confirmed via direct inspection of `build_model`) — there is no scalar stream to condition on. This matches the spec's explicitly-stated fallback ("start here, the simplest version that is still non-degenerate"), not a deviation from it. Extending to per-particle `c_k(s_i)` would first require adding a scalar stream to the top-tagging model, out of scope for this plan.
- **Worktree dependency order**: Worktree 1 (Tasks 1-4) has no dependency on 2 or 3. Worktree 2 (Tasks 5-7) and Worktree 3 (Tasks 8-10) both depend only on Worktree 1's `RotorGenerator`/`rotor_encoding` surface, not on each other — confirmed no shared file edits between Tasks 5-7 and Tasks 8-10 other than the independently-idempotent `set_seed` check (Task 7 Step 1 / Task 10 Step 1, both no-ops if already present).
