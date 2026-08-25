"""Config name -> (layout, injectors), with build-time validation.

Collision detection keys on (channel range, slot), not on slot alone: r5a's
triad injector legitimately writes slot 7 on channel 1 and on channels 2..K,
and must not be reported as colliding with itself.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from experiments.top_tagging.embedding import injectors, layouts

# Channel ranges an injector may claim, and which pairs overlap.
CH0 = "ch0"
AXIS_ALL = "axis_all"        # channels 1..K
AXIS_FIRST4 = "axis_first4"  # channels 1..4

_OVERLAPS = {
    (CH0, CH0),
    (AXIS_ALL, AXIS_ALL), (AXIS_ALL, AXIS_FIRST4),
    (AXIS_FIRST4, AXIS_ALL), (AXIS_FIRST4, AXIS_FIRST4),
}


@dataclass(frozen=True)
class Injector:
    name: str
    fn: Callable
    slot: int
    channels: str
    requires_aux: Tuple[str, ...] = ()
    requires_layout: Optional[str] = None
    requires_k: Optional[int] = None


LAYOUTS = {
    "multi_channel": layouts.multi_channel,
    "single_channel": layouts.single_channel,
    "hybrid_r9": layouts.hybrid_r9,
}

INJECTORS = {
    "subjet_mass": Injector(
        name="subjet_mass",
        fn=injectors.subjet_mass,
        slot=0,
        channels=AXIS_FIRST4,
        requires_aux=("masses",),
        requires_layout="multi_channel",
        requires_k=4,
    ),
    "triad_handedness": Injector(
        name="triad_handedness",
        fn=injectors.triad_handedness,
        slot=7,
        channels=AXIS_ALL,
        requires_aux=("axes",),
        requires_layout="multi_channel",
        requires_k=4,
    ),
    "local_handedness": Injector(
        name="local_handedness",
        fn=injectors.local_handedness,
        slot=7,
        channels=AXIS_ALL,
        requires_aux=("axes",),
        requires_layout="multi_channel",
    ),
    "ca_phase": Injector(
        name="ca_phase",
        fn=injectors.ca_phase,
        slot=7,
        channels=AXIS_FIRST4,
        requires_aux=("phases",),
        requires_layout="multi_channel",
        requires_k=4,
    ),
    "lead_pseudoscalar": Injector(
        name="lead_pseudoscalar",
        fn=injectors.lead_pseudoscalar,
        slot=7,
        channels=CH0,
        requires_aux=("axes", "p_lead"),
        requires_layout="single_channel",
    ),
    "cos_theta": Injector(
        name="cos_theta",
        fn=injectors.cos_theta,
        slot=4,
        channels=CH0,
        requires_aux=("axes",),
        requires_layout="hybrid_r9",
        requires_k=4,
    ),
}


def resolve(layout_name: str, inject_names):
    """Look up a layout and its injectors, validating compatibility.

    Raises ValueError on an unknown name, a layout mismatch, or two injectors
    claiming overlapping (channel range, slot) regions.
    """
    if layout_name not in LAYOUTS:
        raise ValueError(
            f"unknown embedding layout {layout_name!r}; known: {sorted(LAYOUTS)}"
        )
    chosen = []
    for name in inject_names:
        if name not in INJECTORS:
            raise ValueError(
                f"unknown injector {name!r}; known: {sorted(INJECTORS)}"
            )
        inj = INJECTORS[name]
        if inj.requires_layout is not None and inj.requires_layout != layout_name:
            raise ValueError(
                f"injector {name!r} requires layout {inj.requires_layout!r}, "
                f"got {layout_name!r}"
            )
        chosen.append(inj)

    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            if a.slot == b.slot and (a.channels, b.channels) in _OVERLAPS:
                raise ValueError(
                    f"injectors {a.name!r} and {b.name!r} both write slot "
                    f"{a.slot} on overlapping channels "
                    f"({a.channels}, {b.channels})"
                )
    return LAYOUTS[layout_name], chosen


def check_aux(injectors_chosen, aux, layout_name):
    """Raise if a chosen injector's aux key or K requirement is unmet."""
    for inj in injectors_chosen:
        for key in inj.requires_aux:
            if aux.get(key) is None:
                raise ValueError(
                    f"injector {inj.name!r} needs aux key {key!r}; regenerate "
                    f"the dataset with: scripts/precompute_features.py "
                    f"--features {key if key != 'p_lead' else 'leading'}"
                )
        if inj.requires_k is not None:
            k = aux["axes"].shape[1] if aux.get("axes") is not None else 0
            if k != inj.requires_k:
                raise ValueError(
                    f"injector {inj.name!r} requires K={inj.requires_k} axis "
                    f"channels, dataset has K={k}"
                )
