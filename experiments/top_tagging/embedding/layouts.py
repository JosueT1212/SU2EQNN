"""Channel layouts. Each owns channel count and any extra output streams.

Signature: (fourmomenta, ptr, aux, two_beams, rescale) -> (mv, extras)

`mv` is pre-spurion (n_particles, C, 8); build_embedding attaches spurions
after injectors have run.
"""

import torch

from experiments.top_tagging.embedding.core import (
    UNITS, get_batch_from_ptr, wedge_into,
)


def multi_channel(fourmomenta, ptr, aux, two_beams=True, rescale=True) -> dict:
    """R4/R5 layout: channel 0 kinematics, channels 1..K the wedge p⃗ ∧ n̂_k.

    Requires aux["axes"] of shape (batchsize, K, 3). Without it the layout
    degenerates to a single kinematic channel (the R1 baseline).
    """
    batchsize = len(ptr) - 1
    axes = aux.get("axes")
    n_channels = 1 if axes is None else 1 + axes.shape[1]

    if rescale:
        fourmomenta = fourmomenta / UNITS

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
        particle_batch = get_batch_from_ptr(ptr)
        ax = axes.to(fourmomenta.dtype)[particle_batch]     # (n_particles, K, 3)
        p = fourmomenta[:, 1:4].unsqueeze(1)                # (n_particles, 1, 3)
        wedge_into(mv, slice(1, None), p, ax)

    return mv, {}


def single_channel(fourmomenta, ptr, aux, two_beams=True, rescale=True):
    """r8 layout: everything in channel 0.

    Slots 0-3 hold (E, px, py, pz); slots 4-6 hold p⃗ ∧ n̂_jet, the K=1
    restriction of the multi_channel wedge; slot 7 is left for an injector.
    """
    if aux.get("axes") is None:
        raise ValueError("single_channel layout requires aux['axes']")
    jet_axis = aux["axes"][:, 0, :]

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

    return mv, {}
