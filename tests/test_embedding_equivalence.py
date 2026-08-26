"""Unified embedding must reproduce each branch's output bit-for-bit.

Fixtures were captured from the branches in Task 1 and are the acceptance
gate for the refactor. A mismatch here means behaviour changed.
"""

from types import SimpleNamespace

import pytest
import torch

from experiments.top_tagging.embedding import build_embedding

FIXTURE_DIR = "tests/fixtures"

# variant -> (layout name, injector list)
VARIANTS = {
    "multi_axis_bivectors": ("multi_channel", []),
    "subjet_mass": ("multi_channel", ["subjet_mass"]),
    "triad_handedness": ("multi_channel", ["triad_handedness"]),
    "local_handedness": ("multi_channel", ["local_handedness"]),
    "ca_phase": ("multi_channel", ["ca_phase"]),
    "single_channel_pseudoscalar": ("single_channel", ["lead_pseudoscalar"]),
    "invariant_subjet_axis": ("invariant_subjet_axis", ["cos_theta"]),
    "bivector_patch": ("patch", []),
}


def _cfg(layout, inject):
    return SimpleNamespace(
        data=SimpleNamespace(
            embedding=layout, inject=inject, two_beams=True, rescale_data=True,
        )
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_matches_branch_fixture(variant):
    layout, inject = VARIANTS[variant]
    fx = torch.load(f"{FIXTURE_DIR}/{variant}.pt", weights_only=False)
    inputs, expected = fx["inputs"], fx["outputs"]

    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}
    got = build_embedding(
        _cfg(layout, inject), inputs["fourmomenta"], inputs["ptr"], aux
    )

    assert set(got) >= set(expected)
    for key, want in expected.items():
        assert torch.equal(got[key], want), f"{variant}: {key} differs"


def test_single_channel_without_injector_leaves_slot7_zero():
    fx = torch.load(f"{FIXTURE_DIR}/single_channel_pseudoscalar.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}

    got = build_embedding(
        _cfg("single_channel", []), inputs["fourmomenta"], inputs["ptr"], aux
    )
    assert got["mv"].shape[1] == 1
    assert torch.equal(got["mv"][:, 0, 7], torch.zeros_like(got["mv"][:, 0, 7]))
    # slots 0-6 are unaffected by the injector's presence
    assert torch.equal(got["mv"][..., :7], fx["outputs"]["mv"][..., :7])


def test_invariant_subjet_axis_without_cos_theta_leaves_ch0_slot4_zero():
    fx = torch.load(f"{FIXTURE_DIR}/invariant_subjet_axis.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}

    got = build_embedding(
        _cfg("invariant_subjet_axis", []), inputs["fourmomenta"], inputs["ptr"], aux
    )
    assert torch.equal(got["mv"][:, 0, 4], torch.zeros_like(got["mv"][:, 0, 4]))
    assert torch.equal(got["mv"][:, 1:], fx["outputs"]["mv"][:, 1:])
