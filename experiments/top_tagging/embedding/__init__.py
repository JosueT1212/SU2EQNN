"""Public embedding API: one entry point, variant chosen by config."""

from experiments.top_tagging.embedding.core import (  # noqa: F401
    UNITS, apply_axis_channel_dropout, apply_patch_channel_dropout,
    get_batch_from_ptr,
)
from experiments.top_tagging.embedding.core import attach_spurions, scatter_particles
from experiments.top_tagging.embedding.registry import check_aux, resolve

__all__ = [
    "build_embedding", "apply_axis_channel_dropout", "apply_patch_channel_dropout",
    "get_batch_from_ptr", "UNITS",
]


def build_embedding(cfg, fourmomenta, ptr, aux) -> dict:
    """Embed 4-momenta as Cl(3,0) multivectors per the configured variant.

    Parameters
    ----------
    cfg  : config exposing cfg.data.embedding (layout name), cfg.data.inject
           (list of injector names), cfg.data.two_beams, cfg.data.rescale_data
    fourmomenta : (n_particles, 4) sparse, zero-padded rows already removed
    ptr         : (batchsize+1,) cumulative particle counts per jet
    aux         : dict of per-jet tensors — "axes", "masses", "phases", "p_lead"

    Returns
    -------
    dict with "mv" (n_total, C, 8), "batch" (n_total,), "is_beam" (n_total,),
    plus "patch_mv" for the patch layout.
    """
    layout_name = getattr(cfg.data, "embedding", "multi_channel")
    inject_names = list(getattr(cfg.data, "inject", []) or [])
    two_beams = cfg.data.two_beams
    rescale = cfg.data.rescale_data

    layout_fn, chosen = resolve(layout_name, inject_names)
    check_aux(chosen, aux, layout_name)

    mv, extras = layout_fn(fourmomenta, ptr, aux, two_beams, rescale)

    if chosen:
        particle_batch = get_batch_from_ptr(ptr)
        for inj in chosen:
            inj.fn(mv, fourmomenta, ptr, particle_batch, aux, rescale)

    out = attach_spurions(mv, ptr, two_beams)
    for key, value in extras.items():
        if key == "_patch_particles":
            out["patch_mv"] = scatter_particles(value, out["is_beam"])
        else:
            out[key] = value
    return out
