"""Slot injectors. Each writes one slot of an existing mv, in place.

Signature: (mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None

Injectors never allocate, reshape, or change the channel count. Physics
bodies are copied verbatim from the branch that introduced them.
"""

import torch

from experiments.top_tagging.embedding.core import UNITS  # noqa: F401  (used by later injectors)


def subjet_mass(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Broadcast [m_jet, m_s1..s3] into slot 0 of channels 1-4 (r5d).

    Masses arrive in GeV and are divided by UNITS when rescale is set, the
    same convention as the momenta in channel 0.
    """
    masses = aux["masses"].to(mv.dtype)
    if rescale:
        masses = masses / UNITS
    mv[:, 1:5, 0] = masses[particle_batch]


def triad_handedness(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Signed subjet triads into slot 7 (r5a).

    Channel 1 carries the jet-level triad handedness broadcast to every
    particle; channels 2..K carry per-particle signed volumes built from the
    jet axis and each subjet axis. chi_triad is a product of unit vectors
    (O(1)) and is deliberately not rescaled. The cross_jk signed volumes are
    built from fourmomenta, which the source branch's local variable had
    already been rescaled by UNITS at that point in the function, so this
    injector rescales its own copy the same way when `rescale` is set.
    """
    if rescale:
        fourmomenta = fourmomenta / UNITS

    axes_f = aux["axes"].to(mv.dtype)                        # (B, K, 3)
    n_jet = axes_f[:, 0]                                     # (B, 3)
    subs = axes_f[:, 1:]                                     # (B, K-1, 3)

    chi_triad = (torch.cross(subs[:, 0], subs[:, 1], dim=-1) * subs[:, 2]).sum(-1)
    mv[:, 1, 7] = chi_triad[particle_batch]

    cross_jk = torch.cross(n_jet.unsqueeze(1).expand_as(subs), subs, dim=-1)
    mv[:, 2:, 7] = torch.einsum(
        "nd,nkd->nk", fourmomenta[:, 1:4], cross_jk[particle_batch]
    )


def local_handedness(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Efremov-style local handedness into slot 7 of every axis channel (r5b).

    v_i = normalized sum_{j!=i} (|p_j|/max(|p_i-p_j|,1e-3)) p_j within the
    jet; chi_i^(k) = n_k . (p_i x v_i). Neighborhood average (not
    nearest-neighbor) for IRC smoothness. O(n^2) per jet.

    NOTE: deviates from the brief's eta/phi-and-pT ("DeltaR") weighting.
    eta, phi and DeltaR are only invariant under boosts/rotations about the
    beam (z) axis, not under a general Spin(3) rotation -- using them here
    made v (and hence chi) fail the full-rotation equivariance test.
    Replacing pT with |p| and DeltaR with the Euclidean momentum separation
    |p_i-p_j| (the direct SO(3)-invariant analog of DeltaR) keeps the same
    "nearby, high-momentum neighbors dominate" idea, makes v a genuinely
    SO(3)-equivariant function of the momenta (so chi = n . (p x v) is
    invariant as required), and avoids the unbounded gradient that an
    arccos-based opening-angle metric would have for collimated jets.

    NOTE: the 1e-3 floor is applied to |p_i-p_j| in *rescaled* momentum space
    when rescale=True (GeV/UNITS=20), not to an eta-phi DeltaR. A separation
    of 1e-3 there is ~0.02 GeV of 3-momentum difference -- below realistic
    constituent spacing, so it only guards the coincident-momentum edge case.
    """
    ax = aux["axes"].to(mv.dtype)[particle_batch]      # (n_particles, K, 3)
    p3 = fourmomenta[:, 1:4]
    if rescale:
        p3 = p3 / UNITS
    pnorm = p3.norm(dim=-1).clamp_min(1e-12)
    v = torch.zeros_like(p3)
    for j0, j1 in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        if j1 - j0 < 2:
            continue
        pj = p3[j0:j1]
        nj = pnorm[j0:j1]
        dist = torch.cdist(pj, pj)
        w = nj[None, :] / dist.clamp_min(1e-3)
        w = w - torch.diag_embed(torch.diagonal(w))          # zero self-weight
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        v[j0:j1] = w @ pj
    cross_pv = torch.linalg.cross(p3, v)                      # (n, 3)
    mv[:, 1:, 7] = torch.einsum("nkd,nd->nk", ax, cross_pv)
