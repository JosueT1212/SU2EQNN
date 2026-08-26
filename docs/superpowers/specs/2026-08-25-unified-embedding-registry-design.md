# Unified Embedding Registry — Design

Date: 2026-08-25
Status: approved for planning

## Problem

Eight experiment branches (`r4-multiaxis-bivector`, `r5a`, `r5b`, `r5c`, `r5d`,
`r7`, `r8`, `r9`) each carry a full copy of the same ~150 lines of shared
infrastructure plus a small variant delta. Running two variants means checking
out two branches. Diffs against `main` are dominated by duplicated base code, so
real deltas are invisible and every port conflicts in the same five files.

The stated goal — "only the embedding changes, so make each version a function" —
is true for six of the eight branches. `r7` also changes model internals
(`nets/gatr.py`, `wrapper.py`), and `r8`/`r9` change the channel *layout*, not
just a slot fill. The design accounts for all three cases.

## What actually differs between branches

Every branch keeps a byte-identical core `embed_tagging_data`: channel 0 holds
`(E, px, py, pz)`, channels `1..K` hold the pure grade-2 wedge `p⃗ ∧ n̂_k`
(indices 4/5/6), beam spurions are zero-padded to `C` channels and interleaved
by `ptr`, and `apply_axis_channel_dropout` masks axis channels per jet.

Variant deltas on top of that core:

| Variant | Kind | Delta |
| --- | --- | --- |
| r4 | — | none; the core *is* R4 |
| r5a | injector | slot 7: triad handedness (ch 1) + per-particle signed volumes (ch 2..K) |
| r5b | injector | slot 7: local handedness `χ = n̂·(p⃗×v⃗)`, O(n²) neighborhood average |
| r5c | injector | slot 7: C-A splitting-plane phases, per-jet broadcast (needs `utils/catree.py`) |
| r5d | injector | slot 0: subjet masses, per-jet broadcast, divided by `UNITS` when `rescale` |
| r8 | layout | single channel: momentum + wedge (K=1) + optional pseudoscalar all in ch 0 |
| r9 | layout | ch 0 gains invariant `cosθ = p̂·n̂_sub_highest` at slot 4; ch 1-3 wedges |
| r7 | layout + model | wedges returned in a separate `patch_mv` tensor for per-block residual re-injection; `gatr.py`/`wrapper.py` consume it |

`r4-explore` and the `r10*` rotor-attention branches are out of scope (see
Non-goals).

## Architecture

New package `experiments/top_tagging/embedding/` replacing the single module:

```
embedding/
  __init__.py    # public: build_embedding, apply_axis_channel_dropout
  core.py        # wedge(), spurion padding, ptr/batch bookkeeping, dropout
  layouts.py     # multi_channel, single_channel, hybrid_r9, patch
  injectors.py   # subjet_mass, triad_handedness, local_handedness,
                 # ca_phase, lead_pseudoscalar, cos_theta
  registry.py    # name -> (layout_fn, [injector names])
```

Data flow:

```
build_embedding(cfg, fourmomenta, ptr, aux)
  -> layout_fn(fourmomenta, ptr, aux, two_beams, rescale) -> dict with "mv", "batch", "is_beam"[, "patch_mv"]
  -> for each injector: injector(mv, fourmomenta, ptr, particle_batch, aux, rescale)   # writes its slot in place
  -> returns the dict
```

Contracts:

- **Layout function** owns channel count, spurion padding and interleaving,
  and any extra output tensors. Signature
  `(fourmomenta, ptr, aux, two_beams, rescale) -> dict`.
- **Injector** never allocates or reshapes; it writes into slots of an existing
  `mv` and returns `None`. Signature
  `(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None`.
- **`aux`** is one dict of per-jet precomputed tensors (`axes`, `masses`,
  `phases`, `leading`), passed through from the dataset. Adding a feature does
  not change any function signature.
- An injector raises if a key it needs is absent from `aux`, naming the
  `precompute_features.py --features` flag that produces it.

Injectors compose: `inject: [subjet_mass, ca_phase]` is valid as long as the
two write disjoint `(channel range, slot)` regions. `registry.py` validates at
build time and raises rather than silently overwriting. Two rules:

- **Collision keys on `(channel_range, slot)`, not on slot alone.** `r5a` writes
  slot 7 twice — triad handedness on channel 1, signed volumes on channels
  2..K — and must not be reported as colliding with itself. It is registered as
  a single injector owning slot 7 across all axis channels.
- **Injectors declare layout compatibility.** `subjet_mass` and `ca_phase`
  hardcode channels 1-4 and require per-jet aux of shape `(B, 4)`, so both
  declare `requires_layout=multi_channel, requires_K=4`. Selecting them with
  `single_channel` (K=1) is a config error raised at build time, not a tensor
  shape error at the first forward pass.

## Config surface

```yaml
data:
  embedding: multi_channel     # multi_channel | single_channel | hybrid_r9 | patch
  inject: []                   # e.g. [subjet_mass] or [triad_handedness]
  features: [axes]             # aux keys required from the npz
model:
  patch_injection: false       # true only for the r7 layout
```

Each ported variant becomes a config file differing from R4's by two or three
lines. `config/top_tagging_r4_explore.yaml` is not carried forward.

## Data pipeline

- `scripts/precompute_features.py --features axes,masses,phases,leading`
  replaces `precompute_axes.py`, `precompute_masses.py`,
  `precompute_phases.py`, `precompute_leading.py`. Each feature is an
  independent writer into the npz; re-running with a subset preserves existing
  keys.
- **The merged writer must be byte-identical to the scripts it replaces.** On
  the mini dataset, `precompute_features.py --features masses` must produce the
  same npz keys and values as `precompute_masses.py` did, and likewise for
  `phases` and `leading`. Without this check the embedding gate below still
  passes — it is fed fixture aux tensors — while the real pipeline silently
  drifts. Verified once per feature, at the commit that introduces it.
- `dataset.py` loads every recognised aux key present in the npz into the
  `Data` object with a generic loop, instead of the hardcoded `axes_{mode}`
  lookup. That hardcode is why `r5d` and `r8` carry a divergent `dataset.py`.
- `experiment.py` passes `aux` through to `build_embedding`; it keeps the Lion
  optimizer, the LR scheduler, and `in_mv_channels` from config that every
  branch already added.

## Landing order

Each step is one commit on `main`, verified before the next starts.

0. **Capture fixtures.** Before any branch is touched, check out each ported
   branch and dump its embedding output *and* its input aux tensors on a
   fixed-seed input to `tests/fixtures/<variant>.pt`. These are the reference tensors for gate 1
   and must exist before step 1 starts, not before step 5.
1. **Shared base.** `su2eqnn/utils/subjets.py`, `scripts/precompute_features.py`,
   generic `dataset.py`, `experiment.py` (Lion, scheduler, aux passing,
   configurable `in_mv_channels`), `tests/test_subjets.py`,
   `tests/test_top_tagging.py`. Identical across all branches today, so it
   lands once and removes the bulk of every later conflict.
2. **`embedding/` package with `multi_channel` only.** Must reproduce the R4
   mini AUC.
3. **Injectors**, one commit each: `subjet_mass` (r5d), `triad_handedness`
   (r5a), `local_handedness` (r5b), `ca_phase` (r5c, plus
   `su2eqnn/utils/catree.py`).
4. **Layouts**, one commit each: `single_channel` (r8), `hybrid_r9` (r9), then
   `patch` (r7) together with the `gatr.py`/`wrapper.py` changes gated behind
   `model.patch_injection`.
5. **Archive.** After a variant reproduces its AUC: `git tag archive/<branch>
   <branch>`, push tags, delete the branch.

## Testing

Three gates, in increasing cost:

1. **Equivalence test** (CI, per variant): unified output tensors equal the
   branch implementation's output on a fixed-seed input, fed the same aux
   tensors. Fixtures store both the embedding output *and* the aux tensors the
   branch consumed, so the aux producer is pinned separately by the byte-identity
   check above. Catches refactor
   errors immediately. Reference tensors are generated once per branch before
   the branch is modified (landing order step 0) and committed as `.pt` fixtures.
2. **Equivariance test** (CI): the per-branch `test_equivariance.py` files are
   near-duplicates; they collapse into one parametrized test that sweeps the
   registry, asserting Spin(3) equivariance for every layout/injector combination.
3. **AUC reproduction** (manual, before archiving a branch): run the variant's
   mini config on unified `main` and compare to the AUC recorded in that
   branch's `outputs/` or `experiment.log`. This is the acceptance bar — tests
   passing is not sufficient.

## Non-goals

- `r10a-wedge-basis`, `r10b-frozen-basis`, `r10-rotor-primitive` stay as live
  branches, unported. They change attention internals, not embeddings; they get
  rebased onto unified `main` in later work.
- `r4-explore` is not ported. It is archived as a tag and deleted; the R4
  variant is taken from `r4-multiaxis-bivector`.
- No plugin/entry-point discovery mechanism. The registry is a dict in one file.
- Committed run artifacts are not propagated to `main`: r5c's
  `outputs/pseudoscalar_phase_mini/` and its logs stay in the archive tag only.
  The `README.md` deletion riding in `r4-explore`/`r5a`/`r5b` is not carried
  forward either.

## Risks

- **Silent numerical drift.** Extracting the shared wedge/spurion code could
  change a variant's output by a rounding-level amount that the AUC gate would
  not catch. Gate 1 (exact tensor equivalence against pre-deletion fixtures)
  exists for this reason and must run before any branch is deleted.
- **AUC non-reproducibility from seeding.** If a branch's recorded AUC came
  from an unseeded run, exact reproduction is impossible. In that case the
  acceptance bar for that variant is agreement within the seed-to-seed spread,
  recorded explicitly in the plan for that variant rather than assumed.
- **`r5b`'s O(n²) loop** is a Python `for` over jets, carried over verbatim.
  A vectorized rewrite is out of scope but not forbidden by gate 1 — what breaks
  bit-exactness is reordering the `clamp_min(1e-3)` and `clamp_min(1e-12)`
  guards, which are load-bearing.
