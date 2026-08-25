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
    "r4": ("multi_channel", []),
    "r5d": ("multi_channel", ["subjet_mass"]),
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
