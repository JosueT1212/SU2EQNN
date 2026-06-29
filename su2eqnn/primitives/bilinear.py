"""Geometric product for Cl(3,0) — equivariant under Spin(3) ≅ SU(2)."""

from functools import lru_cache

import torch
import clifford

from su2eqnn.utils.einsum import cached_einsum

MV_DIM = 8  # dim of Cl(3,0): 2^3 basis elements


@lru_cache()
def _load_geometric_product_tensor(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Loads Cl(3,0) geometric product tensor.

    Cached. Shape (8, 8, 8): gmt[i, j, k] used in einsum
    'i j k, ...j, ...k -> ...i' to compute the geometric product.

    Returns
    -------
    gmt : torch.Tensor with shape (8, 8, 8)
    """
    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        gmt = _load_geometric_product_tensor()
    else:
        layout, _ = clifford.Cl(3, 0)          # Cl(3,0): e_i^2 = +1
        gmt = torch.tensor(layout.gmt, dtype=torch.float32)
        gmt = torch.transpose(gmt, 1, 0)        # match einsum convention
        gmt = gmt.to_dense()
    return gmt.to(device=device, dtype=dtype)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the Cl(3,0) geometric product f(x, y) = xy.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 8)
    y : torch.Tensor with shape (..., 8)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 8)
    """
    gp = _load_geometric_product_tensor(device=x.device, dtype=x.dtype)
    return cached_einsum("i j k, ... j, ... k -> ... i", gp, x, y)
