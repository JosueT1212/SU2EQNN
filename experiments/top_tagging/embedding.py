"""Embedding of 4-momenta into Cl(3,0) multivectors for jet tagging.

Particle MV layout, channel 0: [E, px, py, pz, 0, 0, 0, 0]
  - E  → index 0 (grade-0, Spin(3)-invariant)
  - px → index 1 (grade-1 e1)
  - py → index 2 (grade-1 e2)
  - pz → index 3 (grade-1 e3)

Optional axis channels 1..K (R4 multi-axis bivector encoding):
  channel 1+k = p⃗ ∧ n̂_k, pure grade-2 at indices 4 (e12), 5 (e13), 6 (e23),
  where n̂_k are per-jet unit reference axes (jet axis + exclusive-kT subjets).

R8 single-channel layout (see embed_tagging_data_r8): everything lives in
channel 0 — momentum at indices 0-3, the p⃗ ∧ n̂_jet bivector at indices 4-6,
and an optional pseudoscalar I = p⃗ · (n̂_jet × p̂_lead) at index 7 (zero when
p_lead is not supplied, i.e. the R8a ablation).
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
    axes: Tensor = None,
) -> dict:
    """Embed sparse 4-momenta as Cl(3,0) MVs and prepend beam spurion tokens.

    Parameters
    ----------
    fourmomenta : Tensor shape (n_particles, 4) — sparse, zero-padded rows already removed
    ptr         : Tensor shape (batchsize+1,) — cumulative particle counts per jet
    two_beams   : bool — whether to prepend both +z and -z beam tokens
    rescale     : bool — divide fourmomenta by UNITS before embedding
    axes        : Tensor (batchsize, K, 3) or None — per-jet unit reference axes.
                  When given, channels 1..K hold the pure grade-2 wedge p⃗_i ∧ n̂_k.

    Returns
    -------
    dict with keys:
        "mv"      : Tensor (n_total, C, 8) with C = 1 + K (K=0 when axes is None)
        "batch"   : Tensor (n_total,) — jet index per token
        "is_beam" : Tensor (n_total,) bool — True for beam tokens
    """
    batchsize = len(ptr) - 1
    n_channels = 1 if axes is None else 1 + axes.shape[1]

    if rescale:
        fourmomenta = fourmomenta / UNITS

    # Build particle MVs: (n_particles, C, 8)
    mv = torch.zeros(
        fourmomenta.shape[0], n_channels, 8,
        dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    mv[:, 0, 0] = fourmomenta[:, 0]  # E  → grade-0
    mv[:, 0, 1] = fourmomenta[:, 1]  # px → e1
    mv[:, 0, 2] = fourmomenta[:, 2]  # py → e2
    mv[:, 0, 3] = fourmomenta[:, 3]  # pz → e3

    if axes is not None:
        if axes.shape[0] != batchsize:
            raise ValueError(
                f"axes batch size {axes.shape[0]} != number of jets {batchsize}"
            )
        # per-particle jet index (pre-spurion): (n_particles,)
        particle_batch = get_batch_from_ptr(ptr)
        ax = axes.to(fourmomenta.dtype)[particle_batch]     # (n_particles, K, 3)
        p = fourmomenta[:, 1:4].unsqueeze(1)                # (n_particles, 1, 3)
        # wedge p∧n in repo layout: idx4 = e12, idx5 = e13, idx6 = e23
        mv[:, 1:, 4] = p[..., 0] * ax[..., 1] - p[..., 1] * ax[..., 0]
        mv[:, 1:, 5] = p[..., 0] * ax[..., 2] - p[..., 2] * ax[..., 0]
        mv[:, 1:, 6] = p[..., 1] * ax[..., 2] - p[..., 2] * ax[..., 1]

    # Get beam spurions: (n_spurions, 1, 8) → zero-padded to C channels
    spurions = embed_spurions(
        two_beams=two_beams,
        device=fourmomenta.device,
        dtype=fourmomenta.dtype,
    )
    n_spurions = spurions.shape[0]
    spurions_full = torch.zeros(
        n_spurions, n_channels, 8, dtype=mv.dtype, device=mv.device
    )
    spurions_full[:, 0, :] = spurions[:, 0, :]

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

    mv_out = torch.empty(n_total, n_channels, 8, dtype=mv.dtype, device=mv.device)
    mv_out[~is_beam] = mv
    mv_out[is_beam] = spurions_full.repeat(batchsize, 1, 1)

    # Update ptr: each jet j gains (j+1)*n_spurions cumulative extra tokens
    ptr_out = ptr.clone()
    ptr_out[1:] = ptr[1:] + (arange + 1) * n_spurions

    batch = get_batch_from_ptr(ptr_out)

    return {"mv": mv_out, "batch": batch, "is_beam": is_beam}


def embed_tagging_data_r8(
    fourmomenta: Tensor,
    ptr: Tensor,
    jet_axis: Tensor,
    two_beams: bool = True,
    rescale: bool = True,
    p_lead: Tensor = None,
) -> dict:
    """Embed sparse 4-momenta as single-channel Cl(3,0) MVs (R8 layout).

    Channel 0 per particle: (E, px, py, pz, b23, b31, b12, I) where
    (b12, b13, b23) = p⃗ ∧ n̂_jet (same wedge as embed_tagging_data's axes
    branch, restricted to K=1) and I = p⃗ · (n̂_jet × p̂_lead) is the
    pseudoscalar volume, present only when p_lead is given.

    Parameters
    ----------
    fourmomenta : Tensor shape (n_particles, 4)
    ptr         : Tensor shape (batchsize+1,)
    jet_axis    : Tensor shape (batchsize, 3) — per-jet unit axis
    two_beams   : bool
    rescale     : bool
    p_lead      : Tensor shape (batchsize, 4) or None — per-jet leading
                  constituent 4-momentum (unrescaled). When given, index 7
                  (pseudoscalar) is filled; otherwise it stays 0 (R8a).

    Returns
    -------
    dict with keys "mv" (n_total, 1, 8), "batch" (n_total,), "is_beam" (n_total,)
    """
    batchsize = len(ptr) - 1

    if rescale:
        fourmomenta = fourmomenta / UNITS

    mv = torch.zeros(
        fourmomenta.shape[0], 1, 8,
        dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    mv[:, 0, 0] = fourmomenta[:, 0]
    mv[:, 0, 1:4] = fourmomenta[:, 1:4]

    particle_batch = get_batch_from_ptr(ptr)
    ax = jet_axis.to(fourmomenta.dtype)[particle_batch]   # (n_particles, 3)
    p = fourmomenta[:, 1:4]                                # (n_particles, 3)
    mv[:, 0, 4] = p[:, 0] * ax[:, 1] - p[:, 1] * ax[:, 0]   # e12
    mv[:, 0, 5] = p[:, 0] * ax[:, 2] - p[:, 2] * ax[:, 0]   # e13
    mv[:, 0, 6] = p[:, 1] * ax[:, 2] - p[:, 2] * ax[:, 1]   # e23

    if p_lead is not None:
        pl = p_lead[:, 1:4].to(fourmomenta.dtype)
        pl_norm = pl.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        pl_hat = (pl / pl_norm)[particle_batch]             # (n_particles, 3)
        cross = torch.cross(ax, pl_hat, dim=-1)
        mv[:, 0, 7] = (p * cross).sum(dim=-1)

    spurions = embed_spurions(
        two_beams=two_beams, device=fourmomenta.device, dtype=fourmomenta.dtype,
    )   # (n_spurions, 1, 8)
    n_spurions = spurions.shape[0]

    arange = torch.arange(batchsize, device=ptr.device)
    spurion_idxs = torch.stack(
        [ptr[:-1] + i for i in range(n_spurions)], dim=0
    ) + n_spurions * arange
    spurion_idxs = spurion_idxs.permute(1, 0).flatten()

    n_total = mv.shape[0] + n_spurions * batchsize
    is_beam = torch.zeros(n_total, dtype=torch.bool, device=mv.device)
    is_beam[spurion_idxs] = True

    mv_out = torch.empty(n_total, 1, 8, dtype=mv.dtype, device=mv.device)
    mv_out[~is_beam] = mv
    mv_out[is_beam] = spurions.repeat(batchsize, 1, 1)

    ptr_out = ptr.clone()
    ptr_out[1:] = ptr[1:] + (arange + 1) * n_spurions
    batch = get_batch_from_ptr(ptr_out)

    return {"mv": mv_out, "batch": batch, "is_beam": is_beam}


def apply_axis_channel_dropout(embedding: dict, p: float) -> dict:
    """Zero each axis channel (1..C-1) independently per jet with prob p.

    Training-time only (caller's responsibility). Channel 0 is never touched.
    Beam tokens already carry zeros in channels >= 1, so masking every token
    by its jet index is safe. No 1/(1-p) rescaling — ModDrop convention: a
    modality is present or absent, and bivector magnitudes are physical.
    Equivariance is preserved: zeroing a channel commutes with the rotor
    action (w · 0 · w^{-1} = 0).

    Parameters
    ----------
    embedding : dict with keys "mv" (n_total, C, 8) and "batch" (n_total,)
                as returned by embed_tagging_data
    p         : float — per-jet, per-channel drop probability

    Returns
    -------
    The same dict, with embedding["mv"] masked in place.
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
