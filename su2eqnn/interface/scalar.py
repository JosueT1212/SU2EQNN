"""Embedding of scalars into Cl(3,0) multivectors (grade-0 component)."""

import torch


def embed_scalar(scalars: torch.Tensor) -> torch.Tensor:
    """Embeds a scalar tensor into an 8-dim Cl(3,0) multivector.

    Parameters
    ----------
    scalars : torch.Tensor with shape (..., 1)

    Returns
    -------
    multivectors : torch.Tensor with shape (..., 8)
        multivectors[..., 0] == scalars; all other components are zero.
    """
    non_scalar_shape = list(scalars.shape[:-1]) + [7]   # 8 - 1 = 7
    padding = torch.zeros(non_scalar_shape, device=scalars.device, dtype=scalars.dtype)
    return torch.cat((scalars, padding), dim=-1)


def extract_scalar(multivectors: torch.Tensor) -> torch.Tensor:
    """Extracts the grade-0 (scalar) component from a Cl(3,0) multivector.

    Parameters
    ----------
    multivectors : torch.Tensor with shape (..., 8)

    Returns
    -------
    scalars : torch.Tensor with shape (..., 1)
    """
    return multivectors[..., [0]]
