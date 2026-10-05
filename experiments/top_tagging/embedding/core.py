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


def boost_to_rest(v: Tensor, P: Tensor) -> Tensor:
    """Apply the inverse standard boost L(P)^-1 to four-vectors v.

    Parameters
    ----------
    v : (..., 4) four-vectors (E, px, py, pz)
    P : (..., 4) timelike reference momentum, broadcastable to v

    Returns
    -------
    (..., 4) v in the rest frame of P, in v's dtype. Computed in float64:
    real jets reach gamma ~ 105 and float32 loses the rest-frame components.
    (gamma-1)/|P|^2 = 1/(m(P0+m)), so there is no division by |P| and a jet
    already at rest maps to the identity.
    """
    v64, P64 = v.double(), P.double()
    P0, Pv = P64[..., :1], P64[..., 1:]
    v0, vv = v64[..., :1], v64[..., 1:]
    m = torch.sqrt(P0**2 - (Pv**2).sum(-1, keepdim=True))
    Pdotv = (Pv * vv).sum(-1, keepdim=True)
    out0 = (P0 * v0 - Pdotv) / m
    outv = vv + Pv * (Pdotv / (m * (P0 + m)) - v0 / m)
    return torch.cat([out0, outv], dim=-1).to(v.dtype)


def jet_momenta(fourmomenta: Tensor, ptr: Tensor) -> Tensor:
    """Total four-momentum per jet, (batchsize, 4), in float64."""
    batch = get_batch_from_ptr(ptr)
    P = torch.zeros(len(ptr) - 1, 4, dtype=torch.float64, device=fourmomenta.device)
    return P.index_add_(0, batch, fourmomenta.double())


def rest_frame_beams(P: Tensor, two_beams: bool, dtype) -> Tensor:
    """Beam tokens seen from each jet's rest frame, (batchsize, S, 8).

    Boosts the null beam directions n = (1, 0, 0, +-1) with L(P)^-1 and keeps
    the unit spatial direction (grade 1, slots 1-3). A z-boost only rescales a
    null vector, so the token is exactly invariant under z-boosts x z-rotations;
    the lab token's spacelike e3 would not be (its stabilizer is SO(1,2)).
    """
    signs = [1.0, -1.0] if two_beams else [1.0]
    n = torch.tensor([[1.0, 0.0, 0.0, s] for s in signs],
                     dtype=torch.float64, device=P.device)
    nb = boost_to_rest(n.unsqueeze(0), P.unsqueeze(1))           # (B, S, 4)
    tokens = torch.zeros(P.shape[0], len(signs), 8, dtype=torch.float64, device=P.device)
    tokens[..., 1:4] = nb[..., 1:] / nb[..., :1]
    return tokens.to(dtype)


def rest_frame_xyplane(P: Tensor, dtype) -> Tensor:
    """The xy-plane bivector gamma1 gamma2 (L-GATr beam_reference "xyplane") seen from
    each jet's rest frame, (batchsize, 1, 8).

    F^{xy} = -F^{yx} = 1 transforms as F' = L F L^T with L = L(P)^-1, built column
    by column from boost_to_rest. Rest-frame split with the connector table (even
    half, D = diag(1,-1,...,-1,+1)): electric part F'^{0i} -> -e_i (slots 1-3),
    magnetic part F'^{xy,xz,yz} -> -e12, -e13, -e23 (slots 4-6). F is fixed by
    z-boosts and z-rotations, so the token is exactly covariant under them. Its
    Euclidean norm sqrt(cosh 2y_T) grows with the jet's transverse rapidity
    (sinh y_T = p_T/m), so it is scaled to unit norm like the null beams; p_T/m stays
    readable as |E|/|B| = tanh y_T. No time token: L(P)^-1 gamma0 would hand the head
    the jet's lab velocity.
    """
    n_jets = P.shape[0]
    eye = torch.eye(4, dtype=torch.float64, device=P.device).expand(n_jets, 4, 4)
    L = boost_to_rest(eye, P.double().unsqueeze(1)).transpose(1, 2)   # L[:, :, b] = L e_b
    F = torch.zeros(4, 4, dtype=torch.float64, device=P.device)
    F[1, 2], F[2, 1] = 1.0, -1.0
    Fp = L @ F @ L.transpose(1, 2)
    tokens = torch.zeros(n_jets, 1, 8, dtype=torch.float64, device=P.device)
    tokens[:, 0, 1:4] = -Fp[:, 0, 1:4]
    tokens[:, 0, 4] = -Fp[:, 1, 2]
    tokens[:, 0, 5] = -Fp[:, 1, 3]
    tokens[:, 0, 6] = -Fp[:, 2, 3]
    tokens = tokens / tokens.norm(dim=-1, keepdim=True)   # |E|^2+|B|^2 = cosh 2y_T >= 1
    return tokens.to(dtype)


def rest_subjets(x_rest: Tensor, ptr: Tensor, subjet_idx: Tensor, k: int = 3) -> dict:
    """Rest-frame Durham subjets from per-constituent labels, all float64.

    Parameters
    ----------
    x_rest     : (n_particles, 4) constituents already boosted with L(P)^-1, float64
    ptr        : (batchsize+1,) cumulative particle counts per jet
    subjet_idx : (n_particles,) long labels 0..k-1 (scripts/precompute_features.py --features durham)

    Returns
    -------
    dict with
      "axes"   (B, k, 3): unit spatial direction of each subjet; zero for an empty subjet
               (a fixed fallback direction would act as a spurion in the rest frame),
      "masses" (B, k):    sqrt(max(p_s^2, 0)) in GeV,
      "p_lead" (B, 4):    the constituent with the highest rest-frame energy.
    Summing after the boost makes every output follow the constituents under any
    Lorentz transformation: the subjets rotate by the same Wigner rotation.
    """
    if (subjet_idx < 0).any() or (subjet_idx >= k).any():
        raise ValueError(
            f"subjet_idx outside 0..{k - 1}: padding/mask mismatch between the "
            "precomputed labels and the dataset"
        )
    batch = get_batch_from_ptr(ptr)
    n_jets = len(ptr) - 1
    S = torch.zeros(n_jets * k, 4, dtype=torch.float64, device=x_rest.device)
    S = S.index_add_(0, batch * k + subjet_idx, x_rest.double()).view(n_jets, k, 4)
    p3 = S[..., 1:]
    norm = p3.norm(dim=-1, keepdim=True)
    axes = torch.where(norm > 1e-9, p3 / norm.clamp_min(1e-9), torch.zeros_like(p3))
    masses = torch.sqrt((S[..., 0] ** 2 - (p3 ** 2).sum(-1)).clamp_min(0.0))

    E = x_rest[:, 0].double()
    e_max = torch.full((n_jets,), -torch.inf, dtype=torch.float64, device=E.device)
    e_max = e_max.scatter_reduce(0, batch, E, "amax")
    is_max = E == e_max[batch]
    pos = torch.arange(len(E), device=E.device)
    first = torch.full((n_jets,), len(E), dtype=torch.long, device=E.device)
    first = first.scatter_reduce(0, batch[is_max], pos[is_max], "amin")
    return {"axes": axes, "masses": masses, "p_lead": x_rest.double()[first]}


def random_z_boost(fourmomenta: Tensor, ptr: Tensor, max_rapidity: float,
                   generator: torch.Generator) -> Tensor:
    """Boost each jet along z by a rapidity drawn from U[-max_rapidity, max_rapidity].

    Out-of-distribution evaluation only. Returns float64; the caller casts the
    embedding to the model dtype after canonicalization.
    """
    y = (torch.rand(len(ptr) - 1, generator=generator, dtype=torch.float64) * 2 - 1)
    y = (y * max_rapidity).to(fourmomenta.device)[get_batch_from_ptr(ptr)]
    x = fourmomenta.double()
    E, pz = x[:, 0], x[:, 3]
    out = x.clone()
    out[:, 0] = torch.cosh(y) * E + torch.sinh(y) * pz
    out[:, 3] = torch.sinh(y) * E + torch.cosh(y) * pz
    return out


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


def attach_spurions(mv: Tensor, ptr: Tensor, two_beams: bool,
                    tokens: Tensor = None) -> dict:
    """Interleave beam spurion tokens into `mv` and rebuild the batch index.

    Spurions carry the beam multivector in channel 0 and zeros elsewhere.
    `tokens` (batchsize, S, 8) replaces the shared lab spurions with per-jet
    ones (rest-frame beams); S = 0 means no beam tokens at all.

    Returns
    -------
    dict with "mv" (n_total, C, 8), "batch" (n_total,), "is_beam" (n_total,)
    """
    batchsize = len(ptr) - 1
    n_channels = mv.shape[1]

    if tokens is None:
        spurions = embed_spurions(
            two_beams=two_beams, device=mv.device, dtype=mv.dtype,
        )
        tokens = spurions[:, 0, :].expand(batchsize, -1, -1)
    n_spurions = tokens.shape[1]
    if n_spurions == 0:
        is_beam = torch.zeros(mv.shape[0], dtype=torch.bool, device=mv.device)
        return {"mv": mv, "batch": get_batch_from_ptr(ptr), "is_beam": is_beam}

    spurions_full = torch.zeros(
        batchsize, n_spurions, n_channels, 8, dtype=mv.dtype, device=mv.device
    )
    spurions_full[:, :, 0, :] = tokens

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
    mv_out[is_beam] = spurions_full.reshape(-1, n_channels, 8)

    ptr_out = ptr.clone()
    ptr_out[1:] = ptr[1:] + (arange + 1) * n_spurions

    return {"mv": mv_out, "batch": get_batch_from_ptr(ptr_out), "is_beam": is_beam}


def scatter_particles(values: Tensor, is_beam: Tensor) -> Tensor:
    """Place per-particle rows into a spurion-interleaved tensor, zeros on beams.

    Parameters
    ----------
    values  : (n_particles, C, 8)
    is_beam : (n_total,) bool from attach_spurions
    """
    out = torch.zeros(
        is_beam.shape[0], values.shape[1], values.shape[2],
        dtype=values.dtype, device=values.device,
    )
    out[~is_beam] = values
    return out


def apply_patch_channel_dropout(embedding: dict, p: float) -> dict:
    """Zero each patch channel independently per jet with prob p.

    Same ModDrop convention as apply_axis_channel_dropout, applied to the
    separate patch_mv stream instead of to input channels.
    """
    patch = embedding.get("patch_mv")
    if patch is None or p <= 0.0:
        return embedding
    batch = embedding["batch"]
    n_jets = int(batch.max().item()) + 1
    keep = (
        torch.rand(n_jets, patch.shape[1], device=patch.device) >= p
    ).to(patch.dtype)
    embedding["patch_mv"] = patch * keep[batch][:, :, None]
    return embedding


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
