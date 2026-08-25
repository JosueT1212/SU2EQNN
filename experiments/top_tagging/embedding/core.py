"""Shared mechanics for every embedding layout: wedges, spurions, dropout.

Multivector slot layout: [scalar, e1, e2, e3, e12, e13, e23, e123].
"""

import torch
from torch import Tensor

from su2eqnn.interface.spurions import embed_spurions

UNITS = 20.0  # GeV — all momenta divided by this before embedding


def get_batch_from_ptr(ptr: Tensor) -> Tensor:
    """ptr (batchsize+1,) -> per-token jet index (n_total,)."""
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1]
    )


def wedge_into(mv: Tensor, chan: slice, p3: Tensor, ax: Tensor) -> None:
    """Write the pure grade-2 wedge p3 ∧ ax into slots 4/5/6 of `chan`.

    Parameters
    ----------
    mv   : (n_particles, C, 8) — written in place
    chan : channel slice receiving the wedge
    p3   : (n_particles, 1, 3) 3-momenta, broadcast over axes
    ax   : (n_particles, K, 3) unit axes, K matching the width of `chan`
    """
    mv[:, chan, 4] = p3[..., 0] * ax[..., 1] - p3[..., 1] * ax[..., 0]  # e12
    mv[:, chan, 5] = p3[..., 0] * ax[..., 2] - p3[..., 2] * ax[..., 0]  # e13
    mv[:, chan, 6] = p3[..., 1] * ax[..., 2] - p3[..., 2] * ax[..., 1]  # e23


def attach_spurions(mv: Tensor, ptr: Tensor, two_beams: bool) -> dict:
    """Interleave beam spurion tokens into `mv` and rebuild the batch index.

    Spurions carry the beam multivector in channel 0 and zeros elsewhere.

    Returns
    -------
    dict with "mv" (n_total, C, 8), "batch" (n_total,), "is_beam" (n_total,)
    """
    batchsize = len(ptr) - 1
    n_channels = mv.shape[1]

    spurions = embed_spurions(
        two_beams=two_beams, device=mv.device, dtype=mv.dtype,
    )
    n_spurions = spurions.shape[0]
    spurions_full = torch.zeros(
        n_spurions, n_channels, 8, dtype=mv.dtype, device=mv.device
    )
    spurions_full[:, 0, :] = spurions[:, 0, :]

    arange = torch.arange(batchsize, device=ptr.device)
    spurion_idxs = torch.stack(
        [ptr[:-1] + i for i in range(n_spurions)], dim=0
    ) + n_spurions * arange
    spurion_idxs = spurion_idxs.permute(1, 0).flatten()

    n_total = mv.shape[0] + n_spurions * batchsize
    is_beam = torch.zeros(n_total, dtype=torch.bool, device=mv.device)
    is_beam[spurion_idxs] = True

    mv_out = torch.empty(n_total, n_channels, 8, dtype=mv.dtype, device=mv.device)
    mv_out[~is_beam] = mv
    mv_out[is_beam] = spurions_full.repeat(batchsize, 1, 1)

    ptr_out = ptr.clone()
    ptr_out[1:] = ptr[1:] + (arange + 1) * n_spurions

    return {"mv": mv_out, "batch": get_batch_from_ptr(ptr_out), "is_beam": is_beam}


def apply_axis_channel_dropout(embedding: dict, p: float) -> dict:
    """Zero each axis channel (1..C-1) independently per jet with prob p.

    Training-time only (caller's responsibility). Channel 0 is never touched.
    Beam tokens already carry zeros in channels >= 1, so masking every token
    by its jet index is safe. No 1/(1-p) rescaling — ModDrop convention: a
    modality is present or absent, and bivector magnitudes are physical.
    Equivariance is preserved: zeroing a channel commutes with the rotor
    action (w · 0 · w^{-1} = 0).
    """
    mv = embedding["mv"]
    n_axis_channels = mv.shape[1] - 1
    if p <= 0.0 or n_axis_channels == 0:
        return embedding
    batch = embedding["batch"]
    n_jets = int(batch.max().item()) + 1
    keep = (
        torch.rand(n_jets, n_axis_channels, device=mv.device) >= p
    ).to(mv.dtype)                                     # (B, C-1)
    mv[:, 1:, :] = mv[:, 1:, :] * keep[batch][:, :, None]
    return embedding
