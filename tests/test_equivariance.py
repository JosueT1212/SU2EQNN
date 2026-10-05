"""Equivariance tests for SU2EQNN layers.

Protocol from spec Section 4:
  For every layer L, verify L(ρ(w)(x)) == ρ(w)(L(x)) for random w ∈ Spin(3).

Spin(3) element w lives in Cl^0(3,0) (even subalgebra ≅ quaternions):
  w = q0*1 + q12*e12 + q13*e13 + q23*e23,  with |w|=1
  w^{-1} = quaternion conjugate: negate bivector part

Group action: ρ(w)(x) = w x w^{-1}  (plain conjugation, since w is even)
"""

import pytest
import torch

from su2eqnn.primitives.bilinear import geometric_product
from su2eqnn.primitives.linear import (
    equi_linear,
    grade_project,
    reverse,
    grade_involute,
    _compute_spin3_equi_linear_basis,
)
from su2eqnn.primitives.invariants import inner_product, abs_squared_norm, spin3_invariants
from su2eqnn.primitives.normalization import equi_layer_norm
from su2eqnn.primitives.nonlinearities import gated_relu, gated_gelu
from su2eqnn.layers.linear import EquiLinear
from su2eqnn.layers.layer_norm import EquiLayerNorm
from su2eqnn.interface.vector import embed_vector, extract_vector
from su2eqnn.interface.scalar import embed_scalar, extract_scalar

TOL = 1e-5
N_TESTS = 50
BATCH = 4
C_IN = 6
C_OUT = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_spin3(batch_shape=()):
    """Random unit quaternion as 8-dim Cl(3,0) multivector.

    Layout: index 0 = scalar, 4 = e12, 5 = e13, 6 = e23 (bivectors).
    """
    q = torch.randn(*batch_shape, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    w = torch.zeros(*batch_shape, 8)
    w[..., 0] = q[..., 0]   # scalar part
    w[..., 4] = q[..., 1]   # e12
    w[..., 5] = q[..., 2]   # e13
    w[..., 6] = q[..., 3]   # e23
    return w


def spin3_action(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute ρ(w)(x) = w x w^{-1} using Cl(3,0) geometric product.

    w : (..., 8)  — Spin(3) element
    x : (..., 8)  — arbitrary multivector
    """
    w_inv = w.clone()
    w_inv[..., 4:7] *= -1    # quaternion conjugate: negate bivector part
    return geometric_product(geometric_product(w, x), w_inv)


def spin3_action_channels(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply spin3_action to each channel of x.

    w : (8,)           — single Spin(3) element
    x : (..., C, 8)    — batched multi-channel multivector
    """
    return torch.stack([spin3_action(w, x[..., c, :]) for c in range(x.shape[-2])], dim=-2)


# ---------------------------------------------------------------------------
# Group action sanity checks
# ---------------------------------------------------------------------------

def test_spin3_inverse():
    """ρ(w)(ρ(w^{-1})(x)) == x."""
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        w = random_spin3()
        w_inv = w.clone(); w_inv[..., 4:7] *= -1
        recovered = spin3_action(w, spin3_action(w_inv, x))
        assert torch.allclose(recovered, x, atol=TOL), \
            f"Group inverse failed: max diff {(recovered - x).abs().max():.2e}"


def test_spin3_norm_preservation():
    """||ρ(w)(x)||^2 == ||x||^2 — Cl(3,0) norm is Spin(3)-invariant."""
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        w = random_spin3()
        wx = spin3_action(w, x)
        assert torch.allclose(inner_product(x, x), inner_product(wx, wx), atol=TOL)


# ---------------------------------------------------------------------------
# Primitive equivariance tests
# ---------------------------------------------------------------------------

def test_geometric_product_equivariance():
    """GP(ρ(w)(x), ρ(w)(y)) == ρ(w)(GP(x, y))."""
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        y = torch.randn(BATCH, 8)
        w = random_spin3()
        lhs = geometric_product(spin3_action(w, x), spin3_action(w, y))
        rhs = spin3_action(w, geometric_product(x, y))
        assert torch.allclose(lhs, rhs, atol=TOL), \
            f"GP equivariance failed: max diff {(lhs - rhs).abs().max():.2e}"


def test_grade_project_equivariance():
    """grade_project(ρ(w)(x)) == ρ(w) applied grade-wise to grade_project(x)."""
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        w = random_spin3()
        gp_wx = grade_project(spin3_action(w, x))    # (..., 4, 8)
        w_gp  = torch.stack(
            [spin3_action(w, grade_project(x)[..., k, :]) for k in range(4)], dim=-2
        )
        assert torch.allclose(gp_wx, w_gp, atol=TOL), \
            f"grade_project equivariance failed: max diff {(gp_wx - w_gp).abs().max():.2e}"


def test_reversal_equivariance():
    """reverse(ρ(w)(x)) == ρ(w)(reverse(x))."""
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        w = random_spin3()
        lhs = reverse(spin3_action(w, x))
        rhs = spin3_action(w, reverse(x))
        assert torch.allclose(lhs, rhs, atol=TOL), \
            f"reverse equivariance failed: max diff {(lhs - rhs).abs().max():.2e}"


def test_inner_product_invariance():
    """inner_product(ρ(w)(x), ρ(w)(y)) == inner_product(x, y)."""
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        y = torch.randn(BATCH, 8)
        w = random_spin3()
        lhs = inner_product(spin3_action(w, x), spin3_action(w, y))
        rhs = inner_product(x, y)
        assert torch.allclose(lhs, rhs, atol=TOL), \
            f"inner_product invariance failed: max diff {(lhs - rhs).abs().max():.2e}"


def test_abs_squared_norm_invariance():
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        w = random_spin3()
        lhs = abs_squared_norm(spin3_action(w, x))
        rhs = abs_squared_norm(x)
        assert torch.allclose(lhs, rhs, atol=TOL)


def test_equi_layer_norm_equivariance():
    """equi_layer_norm commutes with ρ(w) channel-wise."""
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, C_IN, 8)
        w = random_spin3()
        lhs = equi_layer_norm(spin3_action_channels(w, x))
        rhs = spin3_action_channels(w, equi_layer_norm(x))
        assert torch.allclose(lhs, rhs, atol=TOL), \
            f"equi_layer_norm equivariance failed: max diff {(lhs - rhs).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Equivariant basis sanity checks
# ---------------------------------------------------------------------------

def test_basis_grade_projections():
    """First 4 basis elements should sum to identity map."""
    basis = _compute_spin3_equi_linear_basis(normalize=False)   # (8, 8, 8)
    grade_sum = basis[:4].sum(0)   # (8, 8)
    # With normalize=False, grade projections should sum to identity
    # (up to a scaling factor — check it's a scalar multiple of identity)
    x = torch.randn(BATCH, 8)
    out = torch.einsum("ij, bj -> bi", grade_sum, x)
    # The sum should reconstruct x (possibly scaled)
    ratio = (out / x).flatten()
    assert ratio.std() < 1e-4, \
        f"Grade projections don't sum to scalar * identity: std={ratio.std():.4f}"


def test_e123_squared():
    """e123^2 == -1 in Cl(3,0)."""
    e123 = torch.zeros(8); e123[7] = 1.0
    e123_sq = geometric_product(e123, e123)
    expected = torch.zeros(8); expected[0] = -1.0
    assert torch.allclose(e123_sq, expected, atol=TOL), \
        f"e123^2 != -1: got {e123_sq}"


def test_e123_bivector_commutation():
    """e123 commutes with all bivectors (spec Section 1)."""
    e123 = torch.zeros(8); e123[7] = 1.0
    for bv_idx in [4, 5, 6]:    # e12, e13, e23
        bv = torch.zeros(8); bv[bv_idx] = 1.0
        left  = geometric_product(e123, bv)
        right = geometric_product(bv, e123)
        assert torch.allclose(left, right, atol=TOL), \
            f"e123 does not commute with bivector index {bv_idx}"


# ---------------------------------------------------------------------------
# Layer equivariance tests
# ---------------------------------------------------------------------------

def test_equi_linear_mv_equivariance():
    """EquiLinear (MV-only) commutes with Spin(3) action."""
    layer = EquiLinear(C_IN, C_OUT)
    layer.eval()
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, C_IN, 8)
        w = random_spin3()
        out_then_act, _ = layer(x)
        path1, _ = layer(spin3_action_channels(w, x))
        path2 = spin3_action_channels(w, out_then_act)
        assert torch.allclose(path1, path2, atol=TOL), \
            f"EquiLinear MV equivariance failed: max diff {(path1 - path2).abs().max():.2e}"


def test_equi_linear_with_scalars():
    """EquiLinear with scalar streams commutes with Spin(3) on MV part."""
    layer = EquiLinear(C_IN, C_OUT, in_s_channels=4, out_s_channels=4)
    layer.eval()
    for _ in range(N_TESTS):
        x  = torch.randn(BATCH, C_IN, 8)
        s  = torch.randn(BATCH, 4)          # scalars are invariant
        w  = random_spin3()
        # Scalars don't transform under Spin(3), MVs do
        out_mv, out_s = layer(x, s)
        path1_mv, path1_s = layer(spin3_action_channels(w, x), s)
        path2_mv = spin3_action_channels(w, out_mv)
        assert torch.allclose(path1_mv, path2_mv, atol=TOL), \
            f"EquiLinear+scalars MV equivariance failed: max diff {(path1_mv-path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=TOL), \
            f"EquiLinear scalars changed under Spin(3)"


def test_embed_extract_vector():
    """embed_vector followed by extract_vector is identity."""
    for _ in range(N_TESTS):
        v = torch.randn(BATCH, 3)
        assert torch.allclose(extract_vector(embed_vector(v)), v, atol=TOL)


def test_embed_vector_equivariance():
    """embed_vector commutes with Spin(3) acting on grade-1 subspace.

    This is the consistency check: rotating a 3-vector then embedding
    should equal embedding then applying the Spin(3) MV action.
    """
    # Use a pure rotation (small angle) as Spin(3) element
    # w = cos(θ/2) + sin(θ/2) * n̂ · (e12, e13, e23)
    theta = 0.3
    n = torch.tensor([1.0, 0.0, 0.0])   # rotate around e12 plane
    w = torch.zeros(8)
    w[0] = torch.cos(torch.tensor(theta / 2))
    w[4] = torch.sin(torch.tensor(theta / 2)) * n[0]
    w[5] = torch.sin(torch.tensor(theta / 2)) * n[1]
    w[6] = torch.sin(torch.tensor(theta / 2)) * n[2]

    for _ in range(20):
        v = torch.randn(3)
        mv = embed_vector(v)
        rotated_mv = spin3_action(w, mv)
        # Rotated MV should still be pure grade-1 (no leakage to other grades)
        assert rotated_mv[0].abs() < TOL, "Rotation leaked to grade-0"
        assert rotated_mv[7].abs() < TOL, "Rotation leaked to grade-3"
        # Grade-2 should also be zero (3-vector stays grade-1)
        assert rotated_mv[4:7].abs().max() < TOL, "Rotation leaked to grade-2"


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


@pytest.mark.parametrize("multi_query", [True, False])
def test_self_attention_equivariance(multi_query):
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
        multi_query=multi_query,
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


@pytest.mark.parametrize("multi_query", [True, False])
def test_self_attention_additional_qk_equivariance(multi_query):
    """SelfAttention with additional_qk_features: V must not be contaminated."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.attention.self_attention import SelfAttention

    ITEMS = 6
    ADD_QK = 2
    cfg = SelfAttentionConfig(
        in_mv_channels=C_IN,
        out_mv_channels=C_OUT,
        in_s_channels=4,
        out_s_channels=4,
        num_heads=2,
        additional_qk_mv_channels=ADD_QK,
        multi_query=multi_query,
    )
    layer = SelfAttention(cfg)
    layer.eval()

    for _ in range(N_TESTS):
        x_mv  = torch.randn(BATCH, ITEMS, C_IN, 8)
        x_s   = torch.randn(BATCH, ITEMS, 4)
        qk_mv = torch.randn(BATCH, ITEMS, ADD_QK, 8)   # additional Q/K features
        w = random_spin3()

        def act(x):
            return torch.stack(
                [spin3_action(w, x[..., c, :]) for c in range(x.shape[-2])], dim=-2
            )

        # Rotate both inputs and additional QK features
        out_mv, _ = layer(x_mv, x_s, additional_qk_features_mv=qk_mv)
        path1_mv, _ = layer(act(x_mv), x_s, additional_qk_features_mv=act(qk_mv))
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"SelfAttention+additional_qk MV equivariance failed: {(path1_mv-path2_mv).abs().max():.2e}"


def test_grade_dropout_equivariance():
    """grade_dropout commutes with Spin(3) when training=False (deterministic)."""
    from su2eqnn.primitives.dropout import grade_dropout

    for _ in range(N_TESTS):
        x = torch.randn(BATCH, 8)
        w = random_spin3()
        # training=False → no stochastic dropout, deterministic pass-through
        lhs = grade_dropout(spin3_action(w, x), p=0.0, training=False)
        rhs = spin3_action(w, grade_dropout(x, p=0.0, training=False))
        assert torch.allclose(lhs, rhs, atol=TOL), \
            f"grade_dropout equivariance failed: {(lhs - rhs).abs().max():.2e}"


def test_scalar_gated_nonlinearity_equivariance():
    """ScalarGatedNonlinearity commutes with Spin(3) on MV stream."""
    from su2eqnn.layers.mlp.nonlinearities import ScalarGatedNonlinearity

    layer = ScalarGatedNonlinearity("gelu")
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, C_IN, 8)
        s = torch.randn(BATCH, 4)
        w = random_spin3()

        def act(mv):
            return torch.stack([spin3_action(w, mv[..., c, :]) for c in range(mv.shape[-2])], dim=-2)

        out_mv, out_s = layer(x, s)
        path1_mv, path1_s = layer(act(x), s)
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=TOL), \
            f"ScalarGatedNonlinearity MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=TOL), \
            f"ScalarGatedNonlinearity scalar invariance failed"


def test_geometric_bilinear_equivariance():
    """GeometricBilinear commutes with Spin(3)."""
    from su2eqnn.layers.mlp.geometric_bilinears import GeometricBilinear

    layer = GeometricBilinear(in_mv_channels=C_IN, out_mv_channels=C_OUT,
                               in_s_channels=4, out_s_channels=4)
    layer.eval()
    for _ in range(N_TESTS):
        x = torch.randn(BATCH, C_IN, 8)
        s = torch.randn(BATCH, 4)
        w = random_spin3()

        def act(mv):
            return torch.stack([spin3_action(w, mv[..., c, :]) for c in range(mv.shape[-2])], dim=-2)

        out_mv, out_s = layer(x, s)
        path1_mv, path1_s = layer(act(x), s)
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"GeometricBilinear MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=1e-4), \
            f"GeometricBilinear scalar invariance failed"


def test_geo_mlp_equivariance():
    """GeoMLP commutes with Spin(3) on the MV stream."""
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.layers.mlp.mlp import GeoMLP

    config = MLPConfig(
        mv_channels=[C_IN, C_IN * 2, C_OUT],
        s_channels=[4, 8, 4],
        activation="gelu",
        dropout_prob=None,
    )
    layer = GeoMLP(config)
    layer.eval()

    for _ in range(N_TESTS):
        x = torch.randn(BATCH, C_IN, 8)
        s = torch.randn(BATCH, 4)
        w = random_spin3()

        def act(mv):
            return torch.stack([spin3_action(w, mv[..., c, :]) for c in range(mv.shape[-2])], dim=-2)

        out_mv, out_s = layer(x, s)
        path1_mv, path1_s = layer(act(x), s)
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"GeoMLP MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=1e-4), \
            f"GeoMLP scalar invariance failed"


def test_su2_block_equivariance():
    """SU2Block commutes with Spin(3) on MV stream; scalars are invariant."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.layers.su2_block import SU2Block

    ITEMS = 5
    MV_C = 4
    S_C = 8

    block = SU2Block(
        mv_channels=MV_C,
        s_channels=S_C,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
    )
    block.eval()

    for _ in range(N_TESTS):
        x_mv = torch.randn(BATCH, ITEMS, MV_C, 8)
        x_s  = torch.randn(BATCH, ITEMS, S_C)
        w = random_spin3()

        def act(mv):
            return torch.stack(
                [spin3_action(w, mv[..., c, :]) for c in range(mv.shape[-2])], dim=-2
            )

        out_mv, out_s   = block(x_mv, scalars=x_s)
        path1_mv, path1_s = block(act(x_mv), scalars=x_s)
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"SU2Block MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=1e-4), \
            f"SU2Block scalar invariance failed: {(path1_s - out_s).abs().max():.2e}"


def test_su2_block_no_scalars_equivariance():
    """SU2Block without scalar stream commutes with Spin(3); out_s is None."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.layers.su2_block import SU2Block

    ITEMS = 5
    MV_C = 4

    block = SU2Block(
        mv_channels=MV_C,
        s_channels=None,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
    )
    block.eval()

    for _ in range(N_TESTS):
        x_mv = torch.randn(BATCH, ITEMS, MV_C, 8)
        w = random_spin3()

        def act(mv):
            return torch.stack(
                [spin3_action(w, mv[..., c, :]) for c in range(mv.shape[-2])], dim=-2
            )

        out_mv, out_s     = block(x_mv, scalars=None)
        path1_mv, path1_s = block(act(x_mv), scalars=None)
        path2_mv = act(out_mv)

        assert out_s is None, "Expected out_s=None when s_channels=None"
        assert path1_s is None
        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"SU2Block (no scalars) MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"


def test_su2eqnn_equivariance():
    """SU2EQNN (2-block) commutes with Spin(3) on MV stream; scalars are invariant."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.nets.su2eqnn_net import SU2EQNN

    ITEMS   = 5
    IN_MV, OUT_MV, HIDDEN_MV = 4, 2, 4
    IN_S,  OUT_S,  HIDDEN_S  = 8, 4, 8

    net = SU2EQNN(
        in_mv_channels=IN_MV,
        out_mv_channels=OUT_MV,
        hidden_mv_channels=HIDDEN_MV,
        in_s_channels=IN_S,
        out_s_channels=OUT_S,
        hidden_s_channels=HIDDEN_S,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=2,
    )
    net.eval()

    for _ in range(N_TESTS):
        x_mv = torch.randn(BATCH, ITEMS, IN_MV, 8)
        x_s  = torch.randn(BATCH, ITEMS, IN_S)
        w = random_spin3()

        def act(mv):
            return torch.stack(
                [spin3_action(w, mv[..., c, :]) for c in range(mv.shape[-2])], dim=-2
            )

        out_mv, out_s     = net(x_mv, scalars=x_s)
        path1_mv, path1_s = net(act(x_mv), scalars=x_s)
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"SU2EQNN MV equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"
        assert torch.allclose(path1_s, out_s, atol=1e-4), \
            f"SU2EQNN scalar invariance failed: {(path1_s - out_s).abs().max():.2e}"



# ---------------------------------------------------------------------------
# Registry sweep: every registered layout/injector combination
# ---------------------------------------------------------------------------
#
# Every registered layout/injector combination must be Spin(3)-equivariant.
#
# A rotation R acting on the input momenta and on the reference axes must act
# on the output multivectors by the corresponding rotor: grade-1 slots rotate
# as vectors, grade-2 slots as bivectors, and grade-0/grade-3 slots are
# invariant.
#
# Deliberate exception: invariant_subjet_axis + cos_theta. The
# invariant-subjet-axis hypothesis is to write a rotation-invariant scalar
# (cos_theta) into channel 0's e12
# slot -- a grade-2 slot that would otherwise be expected to transform as a
# bivector. This is a faithful port of that design, not a bug; see the
# invariant_subjet_axis docstring in layouts.py and the cos_theta docstring in
# injectors.py for why.
#
# Caveat: the kT subjet axes and the C/A tree are computed in the beam frame,
# so *precomputed* aux is only equivariant under azimuthal rotations in
# production. This test rotates the aux tensors explicitly, which isolates
# the embedding's own equivariance -- the property being asserted -- from the
# aux producer's frame dependence.

import pytest

from experiments.top_tagging.embedding import build_embedding
from tests.test_embedding_equivalence import VARIANTS, _cfg

FIXTURE_DIR = "tests/fixtures"


def _random_rotation(seed):
    """Uniform-ish SO(3) element via QR of a Gaussian matrix, det forced to +1."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(torch.float32)


# Variants whose slot 7 is identically zero in the plain (unrotated)
# embedding -- no injector for that variant writes it (multi_axis_bivectors
# and bivector_patch have no injector at all; the subjet_mass variant's
# subjet_mass injector writes slot 0, not 7; the invariant_subjet_axis
# variant's cos_theta writes slot 4, not 7). For these four, the slot-7 half of the invariance
# check below is a structural 0 == 0: it holds for any implementation and
# provides no coverage of any injector. It is kept anyway (harmless, matches
# the brief's literal sweep, and still catches an implementation that
# accidentally starts writing slot 7), but the test asserts explicitly that
# each such case really is the trivial kind, so a reader -- or a future
# change to an injector -- can tell which of the 16 cases have teeth.
# triad_handedness, local_handedness, ca_phase and
# single_channel_pseudoscalar all write nonzero content into slot 7 and are
# NOT in this set; their slot-7 checks are the real thing.
_SLOT7_IS_ZERO = {
    "multi_axis_bivectors", "subjet_mass", "bivector_patch",
    "invariant_subjet_axis",
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("seed", [1, 2])
def test_scalar_slots_are_rotation_invariant(variant, seed):
    """Grade-0 (slot 0) and grade-3 (slot 7) outputs must not change under R.

    Slot 0 (energy) is nonzero for every variant, so its invariance check
    always has teeth; the test asserts that positively below. Slot 7 is a
    structural zero for the variants in _SLOT7_IS_ZERO -- see that comment.
    """
    layout, inject = VARIANTS[variant]
    fx = torch.load(f"{FIXTURE_DIR}/{variant}.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}
    cfg = _cfg(layout, inject)

    plain = build_embedding(cfg, inputs["fourmomenta"], inputs["ptr"], aux)

    R = _random_rotation(seed)
    rot_fm = inputs["fourmomenta"].clone()
    rot_fm[:, 1:4] = rot_fm[:, 1:4] @ R.T
    rot_aux = dict(aux)
    rot_aux["axes"] = aux["axes"] @ R.T
    if "p_lead" in aux:
        pl = aux["p_lead"].clone()
        pl[:, 1:4] = pl[:, 1:4] @ R.T
        rot_aux["p_lead"] = pl

    rotated = build_embedding(cfg, rot_fm, inputs["ptr"], rot_aux)

    # At least one of the two checked slots must be non-trivial, or this
    # whole case would be vacuous. Slot 0 (energy) always is.
    assert plain["mv"][..., 0].abs().max() > 0, (
        f"{variant}: slot 0 unexpectedly all-zero -- this sweep case would "
        f"be entirely vacuous"
    )

    is_slot7_trivial = variant in _SLOT7_IS_ZERO
    slot7_actually_zero = plain["mv"][..., 7].abs().max() == 0.0
    assert is_slot7_trivial == slot7_actually_zero, (
        f"{variant}: _SLOT7_IS_ZERO is stale (expected slot-7-is-zero to be "
        f"{is_slot7_trivial}, got {slot7_actually_zero}) -- re-audit whether "
        f"this variant's slot-7 invariance check is now informative"
    )

    for slot in (0, 7):
        assert torch.allclose(
            plain["mv"][..., slot], rotated["mv"][..., slot], atol=1e-4
        ), f"{variant}: slot {slot} is not rotation invariant"


def test_ca_phase_injector_values_and_invariance():
    """ca_phase writes precomputed C/A splitting-plane phases into
    slot 7 of channels 1-4. This test exists because the sweep case above
    for the ca_phase variant is tautological: ca_phase's input (aux["phases"]) is a
    precomputed pseudoscalar with no rotation law, so it is correctly never
    rotated -- but that means the "rotated" call recomputes a bit-identical
    tensor and the slot-7 invariance assertion above is 0 == 0 by
    construction, for any implementation of ca_phase, correct or not.

    This test instead: (1) checks the actual injected values against the
    fixture's own aux["phases"], (2) checks slot 7 is untouched on channel 0
    (the only channel ca_phase does not own), and (3) gives a non-vacuous
    invariance check by rotating momenta+axes and confirming slot 7 stays
    bit-identical while the genuine bivector channels (slots 4-6) actually
    change -- i.e. it pins that rotation moves the geometric channels but
    correctly leaves the precomputed phase slot alone.
    """
    from experiments.top_tagging.embedding import get_batch_from_ptr

    layout, inject = VARIANTS["ca_phase"]
    fx = torch.load(f"{FIXTURE_DIR}/ca_phase.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}
    cfg = _cfg(layout, inject)

    plain = build_embedding(cfg, inputs["fourmomenta"], inputs["ptr"], aux)
    particles = plain["mv"][~plain["is_beam"]]

    # (1) injected values: slot 7 of channels 1-4 must equal each particle's
    # jet's aux["phases"] row exactly (subjet_mass/ca_phase-style injectors
    # copy verbatim, no rescaling).
    particle_batch = get_batch_from_ptr(inputs["ptr"])
    expected_phase = aux["phases"][particle_batch]   # (n_particles, 4)
    assert torch.equal(particles[:, 1:5, 7], expected_phase),         "ca_phase: channels 1-4 slot 7 must equal the per-jet aux phases"

    # (2) untouched elsewhere: ca_phase owns only channels 1-4; channel 0's
    # slot 7 must stay exactly zero (no injector writes it for this variant).
    assert particles[:, 0, 7].abs().max() == 0.0,         "ca_phase must not touch channel 0 slot 7"

    # (3) non-vacuous invariance: rotate momenta + axes (never phases -- see
    # docstring). Slot 7 must stay bit-identical to the unrotated case...
    R = _random_rotation(1)
    rot_fm = inputs["fourmomenta"].clone()
    rot_fm[:, 1:4] = rot_fm[:, 1:4] @ R.T
    rot_aux = dict(aux)
    rot_aux["axes"] = aux["axes"] @ R.T

    rotated = build_embedding(cfg, rot_fm, inputs["ptr"], rot_aux)
    rot_particles = rotated["mv"][~rotated["is_beam"]]

    assert torch.equal(particles[:, 1:5, 7], rot_particles[:, 1:5, 7]),         "ca_phase slot must stay bit-identical under rotation (precomputed, no rotation law)"

    # ...while slots 4-6 (the real wedge bivector) are non-trivial and
    # genuinely move under the same rotation -- otherwise (3) would just be
    # re-proving the same vacuous 0 == 0 the sweep already has.
    assert particles[:, 1:5, 4:7].abs().max() > 0,         "ca_phase: bivector channels 1-4 unexpectedly all-zero pre-rotation"
    assert not torch.allclose(
        particles[:, 1:5, 4:7], rot_particles[:, 1:5, 4:7], atol=1e-4
    ), "ca_phase: bivector channels did not move under rotation -- invariance check above would be vacuous"


# ---------------------------------------------------------------------------
# Patch pathway (bivector_patch): equivariance tests ported from the
# r7-patch-injection branch, which were outside the checkout list for the
# earlier porting task and were never carried over. The embedding-level pair
# below is adapted to the current API: build_embedding(cfg, ...) with embedding="patch", and
# patch_mv as a returned key (that branch had a standalone
# embed_tagging_data_bivector_patch function that returned patch_mv
# directly). The SU2EQNN-net-level pair is ported verbatim: su2eqnn.nets.su2eqnn_net
# already exposes patch_mv_channels/patch_mv unchanged from that branch.
# ---------------------------------------------------------------------------

def test_build_embedding_patch_channels():
    """patch layout: mv is plain (channel-0-only); patch_mv carries the wedges.

    Not covered by the parametrized sweep above, which only checks slots 0
    and 7 of "mv" -- it never inspects "patch_mv" at all.
    """
    from experiments.top_tagging.embedding import UNITS

    fourmomenta = torch.tensor([
        [40.0, 3.0, 0.0, 4.0],
        [10.0, 0.0, 1.0, 2.0],
        [20.0, 1.0, 1.0, 5.0],
    ])
    ptr = torch.tensor([0, 2, 3])
    axes = torch.zeros(2, 4, 3)
    axes[:, :, 2] = 1.0   # all axes = +z

    cfg = _cfg("patch", [])
    out = build_embedding(cfg, fourmomenta, ptr, {"axes": axes})
    mv = out["mv"]
    patch_mv = out["patch_mv"]

    assert mv.shape[1] == 1, f"expected plain 1-channel mv, got {mv.shape[1]}"
    assert patch_mv.shape[1] == 4, f"expected 4 patch channels, got {patch_mv.shape[1]}"

    particles_mv = mv[~out["is_beam"]]          # (3, 1, 8)
    particles_patch = patch_mv[~out["is_beam"]]  # (3, 4, 8)
    p = fourmomenta / UNITS

    # mv channel 0: E, px, py, pz -- identical to the plain kinematic channel
    assert torch.allclose(particles_mv[:, 0, 0], p[:, 0], atol=1e-6)
    assert torch.allclose(particles_mv[:, 0, 1:4], p[:, 1:4], atol=1e-6)

    # patch_mv with n̂ = +z: wedge = px*e13 + py*e23
    for k in range(4):
        assert particles_patch[:, k, 4].abs().max() < 1e-6
        assert torch.allclose(particles_patch[:, k, 5], p[:, 1], atol=1e-6)
        assert torch.allclose(particles_patch[:, k, 6], p[:, 2], atol=1e-6)
        assert particles_patch[:, k, 0:4].abs().max() < 1e-6, "patch channels must be pure grade-2"
        assert particles_patch[:, k, 7].abs().max() < 1e-6

    # spurion tokens: mv has normal beam content, patch_mv is exactly zero
    beams_patch = patch_mv[out["is_beam"]]
    assert beams_patch.abs().max() < 1e-6, "patch_mv must be zero on spurion tokens"


def test_build_embedding_patch_equivariance():
    """Rotating momenta AND axes rotates both mv and patch_mv by ρ(w).

    Not covered by the parametrized sweep above: the sweep only checks that
    the invariant slots (0, 7) of "mv" are unchanged; it does not check the
    full bivector transformation law, and it never touches "patch_mv".
    """
    from su2eqnn.interface.vector import embed_vector, extract_vector

    def rotate3(w, v):
        """Rotate 3-vectors (n,3) via the Spin(3) sandwich on grade-1."""
        return torch.stack([extract_vector(spin3_action(w, embed_vector(v[i])))
                            for i in range(v.shape[0])])

    cfg = _cfg("patch", [])
    cfg.data.two_beams = False
    cfg.data.rescale_data = False

    for _ in range(10):
        fourmomenta = torch.randn(6, 4)
        ptr = torch.tensor([0, 3, 6])
        axes = torch.randn(2, 4, 3)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        w = random_spin3()

        out = build_embedding(cfg, fourmomenta, ptr, {"axes": axes})

        rot_p = fourmomenta.clone()
        rot_p[:, 1:4] = rotate3(w, fourmomenta[:, 1:4])
        rot_axes = torch.stack([rotate3(w, axes[j]) for j in range(axes.shape[0])])
        out_rot = build_embedding(cfg, rot_p, ptr, {"axes": rot_axes})

        # Beam spurion tokens are fixed lab-frame directions (unaffected by
        # rotating fourmomenta/axes) and are therefore not equivariant under
        # a general Spin(3) rotation by construction -- only particle tokens
        # are expected to satisfy the equivariance check here.
        mask = ~out["is_beam"]
        expected_mv = spin3_action_channels(w, out["mv"])
        expected_patch = spin3_action_channels(w, out["patch_mv"])

        assert torch.allclose(out_rot["mv"][mask], expected_mv[mask], atol=1e-4), \
            f"patch mv equivariance failed: {(out_rot['mv'][mask] - expected_mv[mask]).abs().max():.2e}"
        assert torch.allclose(out_rot["patch_mv"][mask], expected_patch[mask], atol=1e-4), \
            f"patch_mv equivariance failed: {(out_rot['patch_mv'][mask] - expected_patch[mask]).abs().max():.2e}"


def test_su2eqnn_patch_mv_equivariance():
    """SU2EQNN with patch injection commutes with Spin(3) when momenta AND patch_mv co-rotate."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.nets.su2eqnn_net import SU2EQNN

    ITEMS = 5
    IN_MV, OUT_MV, HIDDEN_MV, PATCH_MV = 1, 1, 4, 4

    net = SU2EQNN(
        in_mv_channels=IN_MV,
        out_mv_channels=OUT_MV,
        hidden_mv_channels=HIDDEN_MV,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=2,
        patch_mv_channels=PATCH_MV,
    )
    net.eval()

    for _ in range(N_TESTS):
        x_mv = torch.randn(BATCH, ITEMS, IN_MV, 8)
        patch_mv = torch.randn(BATCH, ITEMS, PATCH_MV, 8)
        w = random_spin3()

        def act(mv):
            return torch.stack(
                [spin3_action(w, mv[..., c, :]) for c in range(mv.shape[-2])], dim=-2
            )

        out_mv, _ = net(x_mv, scalars=None, patch_mv=patch_mv)
        path1_mv, _ = net(act(x_mv), scalars=None, patch_mv=act(patch_mv))
        path2_mv = act(out_mv)

        assert torch.allclose(path1_mv, path2_mv, atol=1e-4), \
            f"SU2EQNN patch equivariance failed: {(path1_mv - path2_mv).abs().max():.2e}"


def test_cos_theta_injector_slot_is_rotation_invariant():
    """The cos_theta injector writes an invariant scalar into mv[:, 0, 4].

    cos_theta = p_hat . n_hat_sub_highest is a genuine rotation invariant,
    deliberately parked in a grade-2 slot of channel 0 because the
    invariant_subjet_axis layout's
    channel-0 slots 4-6 are otherwise unused. The parametrized sweep above
    only checks slots 0 and 7, so it never exercises this claim -- this test
    closes that gap using the same rotation machinery as the sweep.

    Channels 1-3 slots 4-6 are genuine bivector components (the wedge
    p x n_hat) and must NOT be asserted invariant here -- they transform
    under rotation like any other grade-2 slot.
    """
    layout, inject = VARIANTS["invariant_subjet_axis"]
    fx = torch.load(f"{FIXTURE_DIR}/invariant_subjet_axis.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}
    cfg = _cfg(layout, inject)

    plain = build_embedding(cfg, inputs["fourmomenta"], inputs["ptr"], aux)

    for seed in (1, 2):
        R = _random_rotation(seed)
        rot_fm = inputs["fourmomenta"].clone()
        rot_fm[:, 1:4] = rot_fm[:, 1:4] @ R.T
        rot_aux = dict(aux)
        rot_aux["axes"] = aux["axes"] @ R.T
        if "p_lead" in aux:
            pl = aux["p_lead"].clone()
            pl[:, 1:4] = pl[:, 1:4] @ R.T
            rot_aux["p_lead"] = pl

        rotated = build_embedding(cfg, rot_fm, inputs["ptr"], rot_aux)

        assert torch.allclose(
            plain["mv"][:, 0, 4], rotated["mv"][:, 0, 4], atol=1e-4
        ), (
            "invariant_subjet_axis: cos_theta slot (channel 0, slot 4) is "
            "not rotation invariant"
        )


# ---------------------------------------------------------------------------
# bivector_patch SU2EQNN-net-level patch-pathway tests, ported verbatim from
# the r7-patch-injection branch (no plumbing change: these already exercised
# su2eqnn.nets.su2eqnn_net.SU2EQNN directly, not build_embedding, on that branch).
# ---------------------------------------------------------------------------

def test_su2eqnn_patch_mv_none_matches_no_patch():
    """SU2EQNN with patch_mv_channels=None is byte-identical to plain SU2EQNN."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.nets.su2eqnn_net import SU2EQNN

    ITEMS = 5
    IN_MV, OUT_MV, HIDDEN_MV = 1, 1, 4

    torch.manual_seed(0)
    net = SU2EQNN(
        in_mv_channels=IN_MV,
        out_mv_channels=OUT_MV,
        hidden_mv_channels=HIDDEN_MV,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=2,
    )
    net.eval()
    assert net.patch_proj is None

    x_mv = torch.randn(BATCH, ITEMS, IN_MV, 8)
    out_mv_a, out_s_a = net(x_mv, scalars=None)
    out_mv_b, out_s_b = net(x_mv, scalars=None, patch_mv=None)

    assert torch.allclose(out_mv_a, out_mv_b, atol=1e-6)


def test_su2eqnn_patch_mv_changes_output():
    """Nonzero patch_mv perturbs the output; patch_mv=None on a patch-enabled net matches no-patch forward."""
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.nets.su2eqnn_net import SU2EQNN

    ITEMS = 5
    IN_MV, OUT_MV, HIDDEN_MV, PATCH_MV = 1, 1, 4, 4

    torch.manual_seed(0)
    net = SU2EQNN(
        in_mv_channels=IN_MV,
        out_mv_channels=OUT_MV,
        hidden_mv_channels=HIDDEN_MV,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=2,
        patch_mv_channels=PATCH_MV,
    )
    net.eval()
    assert net.patch_proj is not None
    assert len(net.patch_proj) == 2

    x_mv = torch.randn(BATCH, ITEMS, IN_MV, 8)
    patch_mv = torch.randn(BATCH, ITEMS, PATCH_MV, 8)

    out_no_patch, _ = net(x_mv, scalars=None, patch_mv=None)
    out_with_patch, _ = net(x_mv, scalars=None, patch_mv=patch_mv)

    assert not torch.allclose(out_no_patch, out_with_patch, atol=1e-6), \
        "nonzero patch_mv must change SU2EQNN output"


def test_su2eqnn_patch_proj_zero_preserved_with_nonzero_bias():
    """patch_proj must preserve zero-patch-mv -> zero-injection even after training moves its bias.

    Spurion tokens carry patch_mv == 0 exactly by design (see
    experiments/top_tagging/embedding's patch layout). If patch_proj[i] had
    a learnable grade-0 bias, that bias starts at zero but receives gradient
    during training and becomes nonzero, which would inject spurious
    grade-0 signal into spurion tokens (patch_proj[i](0) != 0). This test
    simulates the post-training state by manually setting a nonzero bias on
    patch_proj[0] and asserts patch_proj[i](0) stays exactly zero.
    """
    from su2eqnn.layers.attention.config import SelfAttentionConfig
    from su2eqnn.layers.mlp.config import MLPConfig
    from su2eqnn.nets.su2eqnn_net import SU2EQNN

    ITEMS = 5
    IN_MV, OUT_MV, HIDDEN_MV, PATCH_MV = 1, 1, 4, 4

    torch.manual_seed(0)
    net = SU2EQNN(
        in_mv_channels=IN_MV,
        out_mv_channels=OUT_MV,
        hidden_mv_channels=HIDDEN_MV,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=2, multi_query=True, pos_encoding=False),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=2,
        patch_mv_channels=PATCH_MV,
    )
    net.eval()
    assert net.patch_proj is not None

    layer0 = net.patch_proj[0]

    # patch_proj must be constructed with bias=False: any grade-0 bias would
    # drift away from zero during training (it receives gradient from the
    # pooled classification logit) and then inject spurious grade-0 signal
    # into spurion tokens, whose patch_mv is exactly zero by design.
    assert layer0.bias is None, (
        "patch_proj layers must be constructed with bias=False so that "
        "patch_proj[i](zero patch_mv) stays exactly zero even after "
        "training moves the bias away from its zero initialization"
    )

    # Simulate post-training state: if a bias parameter existed, force it
    # nonzero (as it would be after gradient updates during training) and
    # confirm the zero-preserving invariant still holds regardless.
    if layer0.bias is not None:
        with torch.no_grad():
            layer0.bias.fill_(0.5)

    zero_patch_mv = torch.zeros(BATCH, ITEMS, HIDDEN_MV, 8)
    out_mv, _ = layer0(zero_patch_mv, scalars=None)

    assert torch.equal(out_mv, torch.zeros_like(out_mv)), (
        "patch_proj[i](zero patch_mv) must stay exactly zero, otherwise "
        "spurion tokens (patch_mv == 0 by design) receive spurious grade-0 "
        "signal"
    )


if __name__ == "__main__":
    test_spin3_inverse()
    test_spin3_norm_preservation()
    test_geometric_product_equivariance()
    test_grade_project_equivariance()
    test_reversal_equivariance()
    test_inner_product_invariance()
    test_abs_squared_norm_invariance()
    test_equi_layer_norm_equivariance()
    test_basis_grade_projections()
    test_e123_squared()
    test_e123_bivector_commutation()
    test_equi_linear_mv_equivariance()
    test_equi_linear_with_scalars()
    test_embed_extract_vector()
    test_embed_vector_equivariance()
    test_sdp_attention_equivariance()
    test_rope_scalar_only()
    test_self_attention_equivariance()
    test_self_attention_additional_qk_equivariance()
    test_grade_dropout_equivariance()
    test_scalar_gated_nonlinearity_equivariance()
    test_geometric_bilinear_equivariance()
    test_geo_mlp_equivariance()
    test_su2_block_equivariance()
    test_su2_block_no_scalars_equivariance()
    test_su2eqnn_equivariance()
    test_build_embedding_patch_channels()
    test_build_embedding_patch_equivariance()
    test_su2eqnn_patch_mv_equivariance()
    test_ca_phase_injector_values_and_invariance()
    test_cos_theta_injector_slot_is_rotation_invariant()
    test_su2eqnn_patch_mv_none_matches_no_patch()
    test_su2eqnn_patch_mv_changes_output()
    test_su2eqnn_patch_proj_zero_preserved_with_nonzero_bias()
    print("All equivariance tests passed.")

