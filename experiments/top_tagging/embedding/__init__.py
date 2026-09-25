"""Public embedding API: one entry point, variant chosen by config."""

from experiments.top_tagging.embedding.core import (  # noqa: F401
    UNITS, apply_axis_channel_dropout, apply_patch_channel_dropout,
    get_batch_from_ptr,
)
from experiments.top_tagging.embedding.core import (
    attach_spurions, boost_to_rest, jet_momenta, rest_frame_beams, scatter_particles,
)
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
           (list of injector names), cfg.data.two_beams, cfg.data.rescale_data,
           and optionally cfg.data.frame ("lab" | "rest") and cfg.data.beams
           ("lab" | "none" | "boosted"). frame: rest boosts every jet to its
           rest frame with L(P)^-1 before embedding (little-group
           canonicalization); it needs beams none or boosted.
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

    frame = getattr(cfg.data, "frame", "lab")
    beams = getattr(cfg.data, "beams", "lab")

    if frame == "rest":
        check_rest_frame(layout_name, inject_names, aux, beams)
    layout_fn, chosen = resolve(layout_name, inject_names)
    check_aux(chosen, aux, layout_name)

    tokens = None
    if frame == "rest":
        P = jet_momenta(fourmomenta, ptr)
        m2 = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(-1)
        if (m2 <= 0).any():
            raise ValueError(
                f"{int((m2 <= 0).sum())} jets have non-timelike P; rest frame undefined"
            )
        fourmomenta = boost_to_rest(fourmomenta, P[get_batch_from_ptr(ptr)])
        if beams == "boosted":
            tokens = rest_frame_beams(P, two_beams, fourmomenta.dtype)
        else:
            tokens = fourmomenta.new_zeros(len(ptr) - 1, 0, 8)
    elif frame != "lab":
        raise ValueError(f"unknown frame {frame!r}; known: 'lab', 'rest'")
    elif beams != "lab":
        raise ValueError(f"beams {beams!r} requires frame: rest; frame: lab uses beams: lab")

    mv, extras = layout_fn(fourmomenta, ptr, aux, two_beams, rescale)

    if chosen:
        particle_batch = get_batch_from_ptr(ptr)
        for inj in chosen:
            inj.fn(mv, fourmomenta, ptr, particle_batch, aux, rescale)

    out = attach_spurions(mv, ptr, two_beams, tokens)
    for key, value in extras.items():
        if key == "_patch_particles":
            out["patch_mv"] = scatter_particles(value, out["is_beam"])
        else:
            out[key] = value
    return out


def check_rest_frame(layout_name, inject_names, aux, beams):
    """frame: rest supports the plain baseline only; raise on anything else.

    Lab-frame aux (subjet axes, masses, phases) is not boosted, and the jet
    axis degenerates in the rest frame (sum of rest-frame 3-momenta is zero).
    """
    if beams not in ("none", "boosted"):
        raise ValueError(f"frame: rest needs beams 'none' or 'boosted', got beams {beams!r}")
    if layout_name != "multi_channel" or inject_names:
        raise ValueError(
            "frame: rest supports only the plain multi_channel baseline with no "
            f"injectors; got layout {layout_name!r}, inject {inject_names}"
        )
    present = sorted(k for k, v in aux.items() if v is not None)
    if present:
        raise ValueError(
            f"frame: rest got lab-frame aux keys {present}; use a dataset "
            "without aux features (e.g. data/toptagging_full.npz)"
        )
