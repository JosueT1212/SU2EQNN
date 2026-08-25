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
