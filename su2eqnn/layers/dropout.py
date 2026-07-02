"""GradeDropout nn.Module for Cl(3,0) dual-stream (MV + scalar)."""

from typing import Tuple

import torch
from torch import nn

from su2eqnn.primitives.dropout import grade_dropout


class GradeDropout(nn.Module):
    """Equivariant dropout: drops grade blocks in MV stream, standard dropout in scalar stream.

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self._p = p

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., C, 8)
        scalars      : torch.Tensor with shape (..., S)

        Returns
        -------
        out_mv : torch.Tensor with shape (..., C, 8)
        out_s  : torch.Tensor with shape (..., S)
        """
        out_mv = grade_dropout(multivectors, p=self._p, training=self.training)
        out_s = torch.nn.functional.dropout(scalars, p=self._p, training=self.training)
        return out_mv, out_s
