"""Capture a branch's embedding output as a regression fixture.

Run from inside a worktree checked out to the variant's branch:

    .venv/bin/python scripts/capture_embedding_fixtures.py r5d /abs/path/to/tests/fixtures

Writes <out_dir>/<variant>.pt containing both the inputs the branch consumed
and the outputs it produced, so the unified implementation can be checked
against it without re-deriving aux tensors.
"""

import sys

import torch

from experiments.top_tagging import embedding as emb

N_JETS = 3
COUNTS = [7, 4, 9]          # particles per jet
K = 4                        # jet axis + 3 subjet axes
SEED = 20260825


def make_inputs():
    """Deterministic (fourmomenta, ptr, aux) shared by every variant."""
    g = torch.Generator().manual_seed(SEED)
    n = sum(COUNTS)
    p3 = torch.randn(n, 3, generator=g, dtype=torch.float32) * 30.0
    e = p3.norm(dim=-1, keepdim=True) + torch.rand(n, 1, generator=g) * 5.0
    fourmomenta = torch.cat([e, p3], dim=-1)

    ptr = torch.tensor([0] + COUNTS, dtype=torch.long).cumsum(0)

    axes = torch.randn(N_JETS, K, 3, generator=g, dtype=torch.float32)
    axes = axes / axes.norm(dim=-1, keepdim=True)

    masses = torch.rand(N_JETS, 4, generator=g, dtype=torch.float32) * 80.0
    phases = torch.rand(N_JETS, 4, generator=g, dtype=torch.float32) * 2.0 - 1.0
    # Synthetic and deliberately NOT pT-ordered: only its direction is used,
    # and both branch and unified code consume this same tensor. Do not
    # "fix" it to a real leading constituent — that invalidates the fixtures.
    p_lead = fourmomenta[[0, COUNTS[0], COUNTS[0] + COUNTS[1]]].clone()

    return fourmomenta, ptr, {
        "axes": axes, "masses": masses, "phases": phases, "p_lead": p_lead,
    }


def call_branch(variant, fourmomenta, ptr, aux):
    """Invoke the branch's embedding entry point for this variant."""
    if variant in ("r4", "r5a", "r5b"):
        return emb.embed_tagging_data(fourmomenta, ptr, axes=aux["axes"])
    if variant == "r5c":
        return emb.embed_tagging_data(
            fourmomenta, ptr, axes=aux["axes"], phases=aux["phases"]
        )
    if variant == "r5d":
        return emb.embed_tagging_data(
            fourmomenta, ptr, axes=aux["axes"], masses=aux["masses"]
        )
    if variant == "r7":
        return emb.embed_tagging_data_bivector_patch(fourmomenta, ptr, axes=aux["axes"])
    if variant == "r8":
        return emb.embed_tagging_data_single_channel_axis(
            fourmomenta, ptr, aux["axes"][:, 0, :], p_lead=aux["p_lead"]
        )
    if variant == "r9":
        return emb.embed_tagging_data_r9(
            fourmomenta, ptr,
            jet_axis=aux["axes"][:, 0, :],
            subjet_axes_2=aux["axes"][:, 2:4, :],
            subjet_axis_highest=aux["axes"][:, 1, :],
        )
    raise SystemExit(f"unknown variant {variant}")


def main(variant, out_dir):
    fourmomenta, ptr, aux = make_inputs()
    out = call_branch(variant, fourmomenta, ptr, aux)
    payload = {
        "inputs": {"fourmomenta": fourmomenta, "ptr": ptr, **aux},
        "outputs": {k: v for k, v in out.items()},
    }
    path = f"{out_dir}/{variant}.pt"
    torch.save(payload, path)
    shapes = {k: tuple(v.shape) for k, v in payload["outputs"].items()}
    print(f"wrote {path}  outputs={shapes}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: capture_embedding_fixtures.py <variant> <out_dir>")
    main(sys.argv[1], sys.argv[2])
