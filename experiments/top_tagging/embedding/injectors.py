"""Slot injectors. Each writes one slot of an existing mv, in place.

Signature: (mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None

Injectors never allocate, reshape, or change the channel count. Physics
bodies are copied verbatim from the branch that introduced them.
"""

import torch

from experiments.top_tagging.embedding.core import UNITS


def subjet_mass(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Broadcast the M aux masses into slot 0 of axis channels 1..M.

    Lab: M = 4, [m_jet, m_s1..s3] on channels 1-4. Rest frame: M = 3, the Durham
    subjet masses (m_jet is Gram-determined from channel 0 and dropped). Masses
    arrive in GeV and are divided by UNITS when rescale is set, the same
    convention as the momenta in channel 0.
    """
    masses = aux["masses"].to(mv.dtype)
    if masses.shape[1] != mv.shape[1] - 1:
        raise ValueError(
            f"subjet_mass: {masses.shape[1]} masses for {mv.shape[1] - 1} axis channels"
        )
    if rescale:
        masses = masses / UNITS
    mv[:, 1:1 + masses.shape[1], 0] = masses[particle_batch]


def triad_handedness(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Signed subjet triads into slot 7.

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
    """Efremov-style local handedness into slot 7 of every axis channel.

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


def ca_phase(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """C-A splitting-plane phases into slot 7 of channels 1-4.

    sin(dpsi) between successive splitting planes on the hard branch, broadcast
    per jet. O(1) by construction, so never rescaled.
    """
    phases = aux["phases"].to(mv.dtype)
    mv[:, 1:5, 7] = phases[particle_batch]


def lead_pseudoscalar(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Pseudoscalar volume I = p⃗ · (n̂_jet × p̂_lead) into slot 7 of channel 0.

    p_lead is the highest-pT constituent's 4-momentum; only its direction is
    used, so the value is scale-free apart from p⃗ itself.
    """
    ax = aux["axes"].to(mv.dtype)[:, 0, :][particle_batch]   # (n_particles, 3)
    p = fourmomenta[:, 1:4].to(mv.dtype)
    if rescale:
        p = p / UNITS
    pl = aux["p_lead"][:, 1:4].to(mv.dtype)
    pl_norm = pl.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pl_hat = (pl / pl_norm)[particle_batch]                  # (n_particles, 3)
    cross = torch.cross(ax, pl_hat, dim=-1)
    mv[:, 0, 7] = (p * cross).sum(dim=-1)


def cos_theta(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Invariant cos(theta) against the highest-pT subjet, into ch0 slot 4.

    cos_theta = p̂ · n̂_sub_highest, where n̂_sub_highest is axes[:, 1]. Scale
    free by construction: the momentum is normalized before the dot product,
    so rescale does not change the value.
    """
    p = fourmomenta[:, 1:4].to(mv.dtype)
    if rescale:
        p = p / UNITS
    p_norm = p.norm(dim=-1).clamp_min(1e-12)
    ax_highest = aux["axes"][:, 1, :].to(mv.dtype)[particle_batch]
    mv[:, 0, 4] = (p * ax_highest).sum(dim=-1) / p_norm


def plane_normal(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Out-of-decay-plane pseudoscalar into slot 7 of axis channels 1-3 (frame: rest).

    Channel k carries p⃗·(n̂_k × n̂_{k+1 mod 3}), the rest-frame port of
    triad_handedness's per-particle p⃗·(n̂_jet × n̂_k): the jet axis is zero in the
    rest frame and the jet-level triad (n̂1×n̂2)·n̂3 is identically zero (subjets sum
    to zero, so their axes are coplanar). Each cross product is normal to the decay
    plane, so the three channels are proportional per jet (factor sin θ_k,k+1).
    Momenta are rescaled by UNITS when `rescale` is set, like triad_handedness.
    """
    p3 = fourmomenta[:, 1:4].to(mv.dtype)
    if rescale:
        p3 = p3 / UNITS
    ax = aux["axes"].to(mv.dtype)                                   # (B, 3, 3)
    normals = torch.linalg.cross(ax, ax.roll(-1, dims=1), dim=-1)   # n_k x n_{k+1}
    mv[:, 1:4, 7] = torch.einsum("nd,nkd->nk", p3, normals[particle_batch])
