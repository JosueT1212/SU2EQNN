"""Spin(3) ≅ SU(2) equivariant linear maps on Cl(3,0) multivectors.

Basis construction follows the same logic as L-GATr (Pin(1,3) case) but for Cl(3,0):
  - 4 grades (0..3) instead of 5
  - 2 multipliers: identity and pseudoscalar e123 (Hodge dual)
  - Total: 4 × 2 = 8 equivariant basis elements

Cross-grade (e123) terms are allowed under Spin(3) but NOT under O(3)/Pin(3):
e123 flips sign under reflections, so those maps break equivariance for Pin(3).

Multivector layout for Cl(3,0):
  index 0     → 1       (grade 0, scalar)
  indices 1-3 → e1,e2,e3   (grade 1, vectors)
  indices 4-6 → e12,e13,e23 (grade 2, bivectors)
  index 7     → e123    (grade 3, pseudoscalar)
"""

from functools import lru_cache

import torch
import clifford
import numpy as np

from su2eqnn.utils.einsum import cached_einsum, custom_einsum

MV_DIM = 8
NUM_SPIN3_LINEAR_BASIS_ELEMENTS = 8   # 4 grades × 2 (id + e123 Hodge)


@lru_cache()
def _compute_spin3_equi_linear_basis(
    device=torch.device("cpu"), dtype=torch.float32, normalize=True
) -> torch.Tensor:
    """Constructs basis elements for Spin(3)-equivariant linear maps on Cl(3,0).

    Returns
    -------
    basis : torch.Tensor with shape (8, 8, 8)
        basis[a, i, j]: the a-th basis map, sending component j to component i.
    """
    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        basis = _compute_spin3_equi_linear_basis(normalize=normalize)
    else:
        layout, blades = clifford.Cl(3, 0)
        linear_basis = []
        # identity first, then e123 Hodge multiplication (mirrors L-GATr pseudoScalar trick)
        mults = [1, layout.pseudoScalar]
        for mult in mults:
            for grade in range(4):          # grades 0, 1, 2, 3
                w = np.stack(
                    [(x(grade) * mult).value for x in blades.values()], axis=1
                )
                w = w.astype(np.float32)
                if normalize:
                    sv_max = np.linalg.svd(w)[1].max()
                    if sv_max > 0:
                        w /= sv_max
                linear_basis.append(w)
        linear_basis = np.stack(linear_basis)
        basis = torch.tensor(linear_basis, dtype=torch.float32).to_dense()
    return basis.to(device=device, dtype=dtype)


@lru_cache()
def _compute_reversal(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Diagonal of the reversal map for Cl(3,0).

    Reversal sign by grade: grade-k gets (-1)^(k(k-1)/2).
      grade 0 → +1, grade 1 → +1, grade 2 → -1, grade 3 → -1

    Returns
    -------
    reversal_diag : torch.Tensor with shape (8,)
    """
    rev = torch.ones(8, device=device, dtype=dtype)
    rev[4:8] = -1   # grades 2 (indices 4-6) and 3 (index 7)
    return rev


@lru_cache()
def _compute_grade_involution(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Diagonal of the grade involution map for Cl(3,0).

    Involution sign by grade: grade-k gets (-1)^k.
      grade 0 → +1, grade 1 → -1, grade 2 → +1, grade 3 → -1

    Returns
    -------
    involution_diag : torch.Tensor with shape (8,)
    """
    inv = torch.ones(8, device=device, dtype=dtype)
    inv[1:4] = -1   # grade 1 (indices 1-3)
    inv[7] = -1     # grade 3 (index 7)
    return inv


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Spin(3)-equivariant linear map on Cl(3,0) multivectors.

    Computes sum_{a,j} coeffs_a W^a_ij x_j over 8 pre-defined basis elements W^a.

    Parameters
    ----------
    x      : torch.Tensor with shape (..., in_channels, 8)
    coeffs : torch.Tensor with shape (out_channels, in_channels, 8)

    Returns
    -------
    outputs : torch.Tensor with shape (..., out_channels, 8)
    """
    basis = _compute_spin3_equi_linear_basis(device=x.device, dtype=x.dtype)
    return custom_einsum(
        "y x a, a i j, ... x j -> ... y i", coeffs, basis, x, path=[0, 1, 0, 1]
    )


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects x to each of the 4 grades of Cl(3,0).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)

    Returns
    -------
    projections : torch.Tensor with shape (..., 4, 8)
        projections[..., k, :] contains only grade-k components.
    """
    basis = _compute_spin3_equi_linear_basis(
        device=x.device, dtype=x.dtype, normalize=False
    )
    basis = basis[:4]  # first 4 are grade projections (identity multiplier)
    return cached_einsum("g i j, ... j -> ... g i", basis, x)


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Computes the reversal of a Cl(3,0) multivector.

    Grade 0,1 unchanged; grades 2,3 flip sign.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Computes the grade involution of a Cl(3,0) multivector.

    Grades 0,2 unchanged; grades 1,3 flip sign.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
