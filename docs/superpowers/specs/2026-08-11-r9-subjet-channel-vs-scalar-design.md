# R9: subjet channel-vs-scalar ablation

## Motivation

R4 mini (jet_axis + 3 exclusive-kT subjets, each its own bivector channel,
5 MV channels total) ties or beats R8a mini (jet_axis only, 1 channel,
scalar-free) and near-ties L-GATr. Open question: does the gain come from
*channel format* (separate p⃗∧n̂ₖ bivector per axis) or just *information
content* (subjet direction, in any encoding)?

R9 demotes the single most informative subjet (highest-pT, from exclusive-kT
clustering) to a per-particle scalar invariant, while keeping jet_axis and
the other 2 subjets as bivector channels — same info, one axis re-encoded.
If test accuracy tracks R4 mini closely, channel format isn't doing the
work. If it drops, channel format matters.

Only one variable changes vs R4's axis set (highest-pT subjet: bivector
channel → scalar feature); jet_axis and the other 2 subjets stay untouched,
so any accuracy delta attributes to that one change.

## Layout

4 total MV channels (`in_mv_channels: 4`), built from the same
`(1 + N_SUBJETS, 3) = (4, 3)` axes array `precompute_axes.py` already
produces (`axes[:,0]` = jet_axis, `axes[:,1..3]` = subjets sorted pT
descending — no new precompute needed):

- **ch0**: momentum (idx 0-3, unchanged) + scalar cosθ = p̂ᵢ·n̂_sub_highest
  at idx 4 (new — channel0 never uses grade-2 slots in the existing
  `embed_tagging_data`, so this slot is free)
- **ch1**: bivector p⃗∧n̂_jet_axis (`axes[:,0,:]`)
- **ch2**: bivector p⃗∧n̂_subjet_mid (2nd-highest pT, `axes[:,2,:]`)
- **ch3**: bivector p⃗∧n̂_subjet_soft (3rd-highest pT, `axes[:,3,:]`)

Highest-pT subjet (`axes[:,1,:]`) supplies the scalar only, excluded from
the bivector-channel set.

## Implementation

**New function** `embed_tagging_data_r9` in
`experiments/top_tagging/embedding.py`:

```
embed_tagging_data_r9(
    fourmomenta: Tensor,       # (n_particles, 4)
    ptr: Tensor,                # (batchsize+1,)
    jet_axis: Tensor,           # (batchsize, 3)  = axes[:,0]
    subjet_axes_2: Tensor,      # (batchsize, 2, 3) = axes[:,2:4]  (mid, soft)
    subjet_axis_highest: Tensor,# (batchsize, 3)  = axes[:,1]
    two_beams: bool = True,
    rescale: bool = True,
) -> dict  # {"mv": (n_total, 4, 8), "batch": ..., "is_beam": ...}
```

Built by composing `embed_tagging_data`'s existing axes-branch wedge logic
(stack `jet_axis` + `subjet_axes_2` into a `(batchsize, 3, 3)` axes tensor,
K=3, reuse the same bivector-fill code for channels 1-3), plus one extra
line computing the scalar for channel0 idx4:

```
p_hat = fourmomenta[:, 1:4] / fourmomenta[:, 1:4].norm(dim=-1, keepdim=True).clamp_min(1e-12)
ax_highest = subjet_axis_highest.to(fourmomenta.dtype)[particle_batch]
mv[:, 0, 4] = (p_hat * ax_highest).sum(dim=-1)
```

Beam-token and ptr-update logic mirrors `embed_tagging_data` exactly
(no change needed beyond `n_channels = 4` fixed instead of derived from
`axes.shape[1]`).

**Config**: new `config/top_tagging_r9.yaml`, copy of
`config/top_tagging_r4.yaml` with:

```yaml
model:
  in_mv_channels: 4   # was 5
```

Same `data.path: data/toptagging_mini_axes.npz`, same training block
(30 epochs, batch_size 32, lr 1e-3).

**Wiring**: `experiments/top_tagging/experiment.py` currently calls
`embed_tagging_data(..., axes=getattr(batch, "axes", None))` (lines 65,
151). Add an R9 branch (config-driven, e.g. `embedding: r9` field, mirroring
however the R8 worktree switched to `embed_tagging_data_r8`) that slices
the loaded `axes` tensor into `jet_axis`/`subjet_axes_2`/
`subjet_axis_highest` and calls `embed_tagging_data_r9` instead.

## Testing

- Unit test in `tests/test_top_tagging.py` (or new
  `tests/test_embedding_r9.py`): shape check (`mv.shape[1] == 4`), verify
  ch0 idx4 scalar equals manual cosθ computation on a synthetic jet, verify
  ch1-3 bivectors match `embed_tagging_data`'s output when fed the same
  3-axis subset (regression-test against the existing K=3 path).
- Equivariance test: rotate synthetic constituents + axes by a random
  SO(3) rotor, confirm ch0 scalar is invariant and ch1-3 bivectors rotate
  correctly (same pattern as `tests/test_equivariance.py` in the R8
  worktree).

## Run plan

5 seeds (0-4), mini dataset (`toptagging_mini_axes.npz`), same eval protocol
as R4/R8. Report mean±std test accuracy. Compare against R4 mini (0.9788)
and R8a mini (0.9804) in the running results table (see
[[r8-single-channel-ablation]] memory).

## Out of scope

- Full-dataset run — mini only, pending signal from this ablation.
- Sweeping *which* subjet gets demoted (lowest-pT, random) — deferred;
  this run tests only the hardest case (highest-pT demoted).
- Any change to `precompute_axes.py` or the underlying exclusive-kT
  clustering — reuses existing precomputed axes unchanged.
