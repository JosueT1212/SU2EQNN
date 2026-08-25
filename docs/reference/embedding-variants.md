# Embedding variant index

Each of the eight branches that this refactor unified injected one physics
observable into the Cl(3,0) embedding differently. After Task 1-11, every
variant is reachable from `main` (or, on this working branch, from
`unify-embeddings`) as `data.embedding` / `data.inject` on a single Hydra
config — no branch checkout required. This table is the index from "the R5b
experiment" to "how do I run that now."

Run any variant with:

```bash
PYTHONPATH=$(pwd) .venv/bin/python -m experiments.top_tagging.experiment --config-name <config>
```

## Table

| old branch | `data.embedding` (layout) | `data.inject` (injector) | aux features needed | config file |
|---|---|---|---|---|
| `r4-multiaxis-bivector` | `multi_channel` | `[]` | `axes` | `config/top_tagging_grade2_bivectors.yaml` |
| `r5a-pseudoscalar-triad` | `multi_channel` | `[triad_handedness]` | `axes` (K=4) | `config/top_tagging_pseudoscalar_triad.yaml` |
| `r5b-pseudoscalar-hand` | `multi_channel` | `[local_handedness]` | `axes` | `config/top_tagging_pseudoscalar_handedness.yaml` |
| `r5c-pseudoscalar-phase` | `multi_channel` | `[ca_phase]` | `phases` (K=4) | `config/top_tagging_pseudoscalar_phase.yaml` |
| `r5d-subjet-mass` | `multi_channel` | `[subjet_mass]` | `masses` (K=4) | `config/top_tagging_subjet_mass.yaml` |
| `r7-patch-injection` | `patch` | `[]` | `axes` | `config/top_tagging_bivector_patch.yaml` |
| `r8-single-channel-ablation` (variant a, bivector only) | `single_channel` | `[]` | `axes` | `config/top_tagging_single_channel_axis.yaml` (full-dataset hyperparameters: Lion optimizer, cosine warm restarts — see Notes) |
| `r8-single-channel-ablation` (variant b, + oriented-volume pseudoscalar) | `single_channel` | `[lead_pseudoscalar]` | `axes`, `p_lead` | `config/top_tagging_single_channel_axis_b.yaml` |
| `r9-subjet-channel-vs-scalar` (scalar arm) | `hybrid_r9` | `[cos_theta]` | `axes` (K=4) | `config/top_tagging_r9_scalar.yaml` |
| `r9-subjet-channel-vs-scalar` (channel arm, ablation twin of r4) | `multi_channel` | `[]` | `axes` | `config/top_tagging_r9_channel.yaml` |

The `r4-explore` and `r10-*` (rotor-attention) branches are exploratory work
not covered by this registry; `r10a-wedge-basis`, `r10b-frozen-basis`, and
`r10-rotor-primitive` remain live branches (not archived — see the
project's Task 12 report for why).

## Precomputing aux features

The `axes`, `masses`, `phases`, and `p_lead` (leading-constituent momentum)
aux features are written by `scripts/precompute_features.py`, which
replaces the four branch-era scripts `precompute_axes.py`,
`precompute_masses.py`, `precompute_phases.py`, and `precompute_leading.py`.
Each feature is an independent writer, so a subset can be added to an
existing file:

```bash
.venv/bin/python scripts/precompute_features.py \
  --features axes,masses,phases,leading \
  data/toptagging_mini.npz data/toptagging_mini_aux.npz
```

`config/top_tagging_grade2_bivectors.yaml`, `top_tagging_bivector_patch.yaml`,
and `top_tagging_r9_*.yaml` only need the `axes` feature and read
`data/toptagging_mini_axes.npz` (the branch-era filename, kept as-is since
it predates the merged aux file). The `r5*` and `r8` variant-b configs read
the merged `data/toptagging_mini_aux.npz` produced by the command above.

## Notes

- **`r5c-pseudoscalar-phase` has a committed AUC baseline that has not yet
  been reproduced.** `outputs/pseudoscalar_phase_mini/experiment.log` on
  that branch is tracked in git history and ends `Test AUC=0.9770 |
  accuracy=0.9260` (30 epochs, `data.path=data/toptagging_mini_axes_phases.npz`).
  This is a real, verifiable baseline — stronger than r4's untracked 0.9786 —
  and Task 12's paired-run table only exercises r5c with a 1-epoch,
  2000-jet subset (0.7519 vs 0.7603), which is not comparable to 0.9770.
  Treat r5c the same as r4: not yet safe to archive/delete against a
  reproduced number. See `docs/reference/variant-auc-baselines.md`'s
  addendum for the full 30-epoch reproduction result once it exists.

- **`scripts/capture_embedding_fixtures.py` is a historical artifact.** It
  calls each branch's pre-refactor entry-point names (e.g. branch-specific
  `embed_tagging_data_*` functions) to capture the `tests/fixtures/*.pt`
  bit-exactness fixtures checked into this repo. It was run once, against
  the (now-archived) source branches, before this refactor landed. It is
  **not runnable** against the unified `main`/`unify-embeddings` tree (the
  functions it imports no longer exist there), and it cannot be re-run once
  the source branches are gone. The fixtures it produced remain the
  regression gate (`tests/test_embedding_equivalence.py`); the script itself
  is kept only for provenance.
- **`config/top_tagging_single_channel_axis.yaml` still carries r8's
  original `checkpoint_path`** (`checkpoints/top_tagging_single_channel_axis_a_full21.pt`).
  Re-running this config as checked in will overwrite that checkpoint file
  if it still exists on disk from the original branch run. Override
  `training.checkpoint_path` if you need to preserve it.
- **r7's ported design is the branch TIP, not the original R7.** Branch
  `r7-patch-injection` at its tip (`100e9b5`) adds a shared `EquiLinear`
  bottleneck to the patch-injection pathway that the earliest R7 design did
  not have. The embedding fixture (`tests/fixtures/r7.pt`) was captured
  from the tip and cannot distinguish the two designs, so any AUC number
  from an early R7 run is not comparable to the unified `patch` layout —
  they are different architectures under the same name. See Task 12's
  report for the paired short-run numbers actually measured.
- Every unified config now sets `training.seed` (added in Task 12) so mini
  runs are reproducible; it was previously absent from every branch-era
  config, including the run that produced r4's untracked 0.9786. Only
  `r9-subjet-channel-vs-scalar`'s `experiment.py` has an internal
  `set_seed` function among the eight source branches (unified's came from
  that file, ported verbatim in Task 2) — the other seven cannot be made
  reproducible by config alone. **This was closed by an external-seeding
  parity sweep, not left as a limitation**: seeding `random`/`numpy`/`torch`
  from outside each branch's `main()` (verified equivalent to internal
  seeding — nothing on either side consumes RNG state earlier) and re-running
  every variant produced bit-for-bit identical unified-vs-branch training
  runs for seven of eight variants (every param count, every validation
  step's AUC/accuracy, epoch loss, and test AUC/accuracy — not just the
  final number), with r9 already verified via true internal seeding on
  both sides. Full methodology and every log line compared:
  `.superpowers/sdd/2026-08-25-unified-embedding-registry/parity-report.md`
  and `docs/reference/variant-auc-baselines.md`'s "Training-path parity
  sweep" section. This is direct verification of the training path itself,
  not an inference from matching embeddings plus a shared `experiment.py`.
- **`config/top_tagging_single_channel_axis.yaml` (r8, variant a) cannot be
  run as checked in, in this environment.** It needs `data/toptagging_full_axes.npz`
  (the 1.2M-jet full dataset with axes precomputed — not distributed, and
  not feasible to precompute in a short session) and the `lion_pytorch`
  package (not installed in `.venv`). Task 12's r8 paired short run
  substituted the mini dataset and the Adam optimizer (matching the
  branch's own mini-scale `top_tagging_single_channel_axis_a.yaml`
  variant) via runtime overrides — the committed config was not changed.
- **`r8-single-channel-ablation`'s branch code infers the pseudoscalar
  channel from data presence, not from config — a real behavior difference
  from the unified registry, discovered during the parity sweep above.**
  The branch's `dataset.py` loads `p_lead_{mode}` into the batch whenever
  that key exists in the `.npz` file, and its embedding dispatch turns the
  pseudoscalar channel on whenever `p_lead` is present on the batch —
  there is no config-level gate on the branch side. The unified registry's
  `data.inject: [...]` list is the *only* thing that controls
  `lead_pseudoscalar`. Practically: r8's original branch dataset
  (`data/toptagging_mini_axes.npz`) has no `p_lead` key, so this never
  fired in the branch's own historical runs — but any dataset file that
  does carry `p_lead` (e.g. the merged aux file, or the parity sweep's
  shared subset file) will silently turn pseudoscalar injection on for the
  branch's "variant a" config regardless of what that config requests.
  This caused an initial parity-sweep mismatch that was fully explained
  once found (see the baselines doc); it is not a unified-side bug, and
  the unified `single_channel` + explicit `inject: [lead_pseudoscalar]`
  path reproduces the branch's pseudoscalar-on behavior exactly once both
  sides are told to use it.
- **r5c and r5d's aux features (`phases`, `masses`) are the slow part of
  `precompute_features.py`.** `ca_hard_branch_phases` is a pure-Python,
  roughly-cubic-per-jet pairwise scan; over the full 24000-jet mini set
  this does not finish in minutes. Task 12's r5c/r5d paired short runs used
  a 2000/500/500-jet subset (`data/toptagging_subset_aux.npz`, not
  committed) instead of the full `data/toptagging_mini_aux.npz` — see the
  baselines doc for why that still answers the training-path question the
  pairing exists to ask.
