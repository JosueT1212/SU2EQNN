"""Embedding of 4-momenta into Cl(3,0) multivectors for jet tagging.

Particle MV layout: [E, px, py, pz, 0, 0, 0, 0]
  - E  → index 0 (grade-0, Spin(3)-invariant)
  - px → index 1 (grade-1 e1)
  - py → index 2 (grade-1 e2)
  - pz → index 3 (grade-1 e3)
"""

import torch
from torch import Tensor

from su2eqnn.interface.spurions import embed_spurions

UNITS = 20.0  # GeV — all momenta divided by this before embedding


def get_batch_from_ptr(ptr: Tensor) -> Tensor:
    """Convert ptr to batch index tensor.

    Parameters
    ----------
    ptr : Tensor shape (batchsize+1,) — cumulative token counts per jet

    Returns
    -------
    batch : Tensor shape (n_total,) — jet index for each token
    """
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1]
    )


def embed_tagging_data(
    fourmomenta: Tensor,
    ptr: Tensor,
    two_beams: bool = True,
    rescale: bool = True,
) -> dict:
    """Embed sparse 4-momenta as Cl(3,0) MVs and prepend beam spurion tokens.

    Parameters
    ----------
    fourmomenta : Tensor shape (n_particles, 4) — sparse, zero-padded rows already removed
    ptr         : Tensor shape (batchsize+1,) — cumulative particle counts per jet
    two_beams   : bool — whether to prepend both +z and -z beam tokens
    rescale     : bool — divide fourmomenta by UNITS before embedding

    Returns
    -------
    dict with keys:
        "mv"      : Tensor (n_total, 1, 8)
        "batch"   : Tensor (n_total,) — jet index per token
        "is_beam" : Tensor (n_total,) bool — True for beam tokens
    """
    batchsize = len(ptr) - 1

    if rescale:
        fourmomenta = fourmomenta / UNITS

    # Build particle MVs: (n_particles, 1, 8)
    mv = torch.zeros(
        fourmomenta.shape[0], 1, 8,
        dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    mv[:, 0, 0] = fourmomenta[:, 0]  # E  → grade-0
    mv[:, 0, 1] = fourmomenta[:, 1]  # px → e1
    mv[:, 0, 2] = fourmomenta[:, 2]  # py → e2
    mv[:, 0, 3] = fourmomenta[:, 3]  # pz → e3

    # Get beam spurions: (n_spurions, 1, 8)
    spurions = embed_spurions(
        two_beams=two_beams,
        device=fourmomenta.device,
        dtype=fourmomenta.dtype,
    )
    n_spurions = spurions.shape[0]

    # Compute output positions for beam tokens after interleaving.
    # Each jet's slice gets n_spurions beam tokens prepended.
    # Beam token i of jet j lands at: ptr[j] + j*n_spurions + i
    arange = torch.arange(batchsize, device=ptr.device)
    spurion_idxs = torch.stack(
        [ptr[:-1] + i for i in range(n_spurions)], dim=0
    ) + n_spurions * arange          # (n_spurions, batchsize)
    spurion_idxs = spurion_idxs.permute(1, 0).flatten()  # (n_spurions*batchsize,)

    n_total = mv.shape[0] + n_spurions * batchsize
    is_beam = torch.zeros(n_total, dtype=torch.bool, device=mv.device)
    is_beam[spurion_idxs] = True

    mv_out = torch.empty(n_total, 1, 8, dtype=mv.dtype, device=mv.device)
    mv_out[~is_beam] = mv
    mv_out[is_beam] = spurions.repeat(batchsize, 1, 1)  # (n_spurions*batchsize, 1, 8)

    # Update ptr: each jet j gains (j+1)*n_spurions cumulative extra tokens
    ptr_out = ptr.clone()
    ptr_out[1:] = ptr[1:] + (arange + 1) * n_spurions

    batch = get_batch_from_ptr(ptr_out)

    return {"mv": mv_out, "batch": batch, "is_beam": is_beam}
