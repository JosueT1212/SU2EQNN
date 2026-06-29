"""Embedding of 3-vectors into Cl(3,0) multivectors (grade-1 subspace).

Cl(3,0) layout:
  index 0     → scalar  (grade 0)
  indices 1-3 → e1,e2,e3 (grade 1) ← 3-vectors live here
  indices 4-6 → e12,e13,e23 (grade 2)
  index 7     → e123   (grade 3)
"""

import torch


def embed_vector(vector: torch.Tensor) -> torch.Tensor:
    """Embeds a 3-vector into an 8-dim Cl(3,0) multivector.

    Parameters
    ----------
    vector : torch.Tensor with shape (..., 3)

    Returns
    -------
    multivector : torch.Tensor with shape (..., 8)
        Grade-1 components (indices 1-3) set to vector; all others zero.
    """
    batch_shape = vector.shape[:-1]
    multivector = torch.zeros(*batch_shape, 8, dtype=vector.dtype, device=vector.device)
    multivector[..., 1:4] = vector
    return multivector


def extract_vector(multivector: torch.Tensor) -> torch.Tensor:
    """Extracts the grade-1 (3-vector) part from a Cl(3,0) multivector.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 8)

    Returns
    -------
    vector : torch.Tensor with shape (..., 3)
    """
    return multivector[..., 1:4]
