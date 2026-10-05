"""Public embedding API: one entry point, variant chosen by config."""

from experiments.top_tagging.embedding.core import (  # noqa: F401
    UNITS, apply_axis_channel_dropout, apply_patch_channel_dropout,
    get_batch_from_ptr,
)
from experiments.top_tagging.embedding.core import (
    attach_spurions, boost_to_rest, jet_momenta, rest_frame_beams, rest_frame_xyplane,
    rest_subjets,
    scatter_particles,
)
from experiments.top_tagging.embedding.registry import check_aux, resolve

__all__ = [
    "build_embedding", "apply_axis_channel_dropout", "apply_patch_channel_dropout",
    "get_batch_from_ptr", "UNITS",
]

# frame: rest support. Lab-only pieces need the jet axis (zero in the rest frame),
# eta-phi trees, or are identically zero (triad: rest-frame subjets are coplanar).
REST_LAYOUTS = ("multi_channel", "patch", "single_channel")
REST_INJECTORS = ("subjet_mass", "local_handedness", "lead_pseudoscalar", "plane_normal")
REST_ONLY_INJECTORS = ("plane_normal",)


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
           cfg.data.beam_reference ("lightlike" default | "xyplane", frame: rest
           with beams: boosted only) picks the boosted beam token(s).
    fourmomenta : (n_particles, 4) sparse, zero-padded rows already removed
    ptr         : (batchsize+1,) cumulative particle counts per jet
    aux         : dict of per-jet tensors — "axes", "masses", "phases", "p_lead" (lab),
                  or per-constituent "subjet_idx" (frame: rest, Durham labels)

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
    beam_reference = getattr(cfg.data, "beam_reference", None)

    rest_only = [n for n in inject_names if n in REST_ONLY_INJECTORS]
    if frame != "rest" and rest_only:
        raise ValueError(f"injectors {rest_only} require frame: rest")

    if frame == "rest":
        check_rest_frame(layout_name, inject_names, aux, beams, beam_reference)
    layout_fn, chosen = resolve(layout_name, inject_names)

    tokens = None
    if frame == "rest":
        P = jet_momenta(fourmomenta, ptr)
        m2 = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(-1)
        if (m2 <= 0).any():
            raise ValueError(
                f"{int((m2 <= 0).sum())} jets have non-timelike P; rest frame undefined"
            )
        x_rest = boost_to_rest(fourmomenta.double(), P[get_batch_from_ptr(ptr)])
        subjet_idx = aux.get("subjet_idx")
        aux = rest_subjets(x_rest, ptr, subjet_idx) if subjet_idx is not None else {}
        fourmomenta = x_rest.to(fourmomenta.dtype)
        if beams == "boosted":
            if beam_reference == "xyplane":
                tokens = rest_frame_xyplane(P, fourmomenta.dtype)
            else:
                tokens = rest_frame_beams(P, two_beams, fourmomenta.dtype)
        else:
            tokens = fourmomenta.new_zeros(len(ptr) - 1, 0, 8)
    elif frame != "lab":
        raise ValueError(f"unknown frame {frame!r}; known: 'lab', 'rest'")
    elif beams != "lab":
        raise ValueError(f"beams {beams!r} requires frame: rest; frame: lab uses beams: lab")
    elif beam_reference is not None:
        raise ValueError(f"beam_reference {beam_reference!r} is a frame: rest option")
    elif aux.get("subjet_idx") is not None:
        raise ValueError("subjet_idx (rest-frame Durham labels) requires frame: rest")
    check_aux(chosen, aux, layout_name)

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


def check_rest_frame(layout_name, inject_names, aux, beams, beam_reference=None):
    """Validate a frame: rest request; raise on anything unsupported.

    Lab-frame aux (subjet axes, masses, phases, p_lead) is not boosted, so it is
    rejected. Anything beyond the plain baseline needs per-constituent Durham
    labels (aux["subjet_idx"]); the subjets are built after the boost.
    """
    if beams not in ("none", "boosted"):
        raise ValueError(f"frame: rest needs beams 'none' or 'boosted', got beams {beams!r}")
    if beam_reference not in (None, "lightlike", "xyplane"):
        raise ValueError(f"unknown beam_reference {beam_reference!r}")
    if beam_reference == "xyplane" and beams != "boosted":
        raise ValueError("beam_reference xyplane needs beams: boosted")
    lab_keys = sorted(k for k, v in aux.items() if v is not None and k != "subjet_idx")
    if lab_keys:
        raise ValueError(
            f"frame: rest got lab-frame aux keys {lab_keys}; use a dataset without "
            "lab aux (toptagging_full.npz or toptagging_full_durham.npz)"
        )
    if layout_name not in REST_LAYOUTS or not set(inject_names) <= set(REST_INJECTORS):
        raise ValueError(
            f"frame: rest supports layouts {REST_LAYOUTS} and injectors {REST_INJECTORS}; "
            f"got layout {layout_name!r}, inject {inject_names}"
        )
    plain = layout_name == "multi_channel" and not inject_names
    if not plain and aux.get("subjet_idx") is None:
        raise ValueError(
            f"frame: rest layout {layout_name!r} with inject {inject_names} needs Durham "
            "labels aux['subjet_idx'] (scripts/precompute_features.py --features durham)"
        )
