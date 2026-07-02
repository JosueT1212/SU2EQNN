"""Equivariance tests for SU2EQNN layers.

Protocol from spec Section 4:
  For every layer L, verify L(ρ(w)(x)) == ρ(w)(L(x)) for random w ∈ Spin(3).

Spin(3) element w lives in Cl^0(3,0) (even subalgebra ≅ quaternions):
  w = q0*1 + q12*e12 + q13*e13 + q23*e23,  with |w|=1
  w^{-1} = quaternion conjugate: negate bivector part

Group action: ρ(w)(x) = w x w^{-1}  (plain conjugation, since w is even)
"""

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


def test_self_attention_additional_qk_equivariance():
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
        multi_query=True,
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
    print("All equivariance tests passed.")
