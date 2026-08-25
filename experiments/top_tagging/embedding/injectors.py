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
