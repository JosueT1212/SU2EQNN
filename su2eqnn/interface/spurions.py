"""Cl(3,0) beam spurion embeddings for Spin(3) symmetry breaking.

Injects z-axis reference vectors to break SO(3) → SO(2).
"""

import torch


def get_num_spurions(two_beams: bool = True) -> int:
    """Return number of beam spurion tokens."""
    return 2 if two_beams else 1


def embed_spurions(
    two_beams: bool = True,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Return z-axis beam spurion MV tokens for Cl(3,0).

    Cl(3,0) layout: index 3 = e3 (grade-1 z-direction).

    Returns
    -------
    spurions : Tensor shape (n_spurions, 1, 8)
        spurions[0, 0] = [0, 0, 0, +1, 0, 0, 0, 0]  # +z beam
        spurions[1, 0] = [0, 0, 0, -1, 0, 0, 0, 0]  # -z beam (if two_beams)
    """
    kwargs = {"device": device, "dtype": dtype}
    beam = torch.zeros(1, 1, 8, **kwargs)
    beam[0, 0, 3] = 1.0
    if two_beams:
        beam_neg = beam.clone()
        beam_neg[0, 0, 3] = -1.0
        return torch.cat([beam, beam_neg], dim=0)  # (2, 1, 8)
    return beam  # (1, 1, 8)
