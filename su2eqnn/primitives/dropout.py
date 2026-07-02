"""Spin(3)-equivariant grade dropout for Cl(3,0) multivectors.

Drops entire grade-k blocks independently. Each grade block is a Spin(3)
subrepresentation, so zeroing a block commutes with the rotor action ρ(w).
"""

import torch
import torch.nn.functional as F

from su2eqnn.primitives.linear import grade_project

_N_GRADES = 4
_MV_DIM = 8


def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Drop entire grade blocks of a Cl(3,0) multivector with probability p.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)
    p : float
        Probability of dropping each grade block.
    training : bool
        If False, no dropout is applied (inference mode).

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    x_grades = grade_project(x)                          # (..., 4, 8)
    batch_shape = x_grades.shape[:-2]
    h = x_grades.view(-1, _N_GRADES, _MV_DIM)           # (N, 4, 8)
    h = F.dropout1d(h, p=p, training=training, inplace=False)
    h = h.view(*batch_shape, _N_GRADES, _MV_DIM)        # (..., 4, 8)
    return h.sum(dim=-2)                                  # (..., 8)
