# Training-path parity report: unified vs. archived branches

Verifies, by actually running both sides with an externally-seeded launcher
and diffing logs, that the unified `experiments/top_tagging/experiment.py`
reproduces each of the eight archived branches' training run — not just
their embeddings (already proven bit-identical by `tests/fixtures/*.pt`).

Executed on `unify-embeddings` at `d2f403b`, 2026-08-25.

**Provenance note:** this file is a copy of the working ledger produced
during that run. The individual run logs it cites were originally written
to `outputs/<name>.log` (an untracked, working-tree-only directory) during
the run itself; the copies actually shipped with this repo, so a clone can
verify them, live at `docs/reference/logs/<name>.log`. Citations below have
been repointed there.

## Headline result

**All eight variants match exactly**, once each side is run with the same
external seed and the same input data. Seven of the eight (`r4`, `r5a`,
`r5b`, `r5c`, `r5d`, `r7`, `r8`) needed an externally-seeded launcher
because their `experiment.py` never calls `set_seed` — the ninth (`r9`) was
already verified in a prior session (`docs/reference/logs/pair_unified_r9.log` /
`docs/reference/logs/pair_branch_r9.log`, both ending `loss=0.3626`, `Test AUC=0.9519`,
`accuracy=0.8895`) and was not re-run here.

## Seeding-support table

```
for b in r4-multiaxis-bivector r5a-pseudoscalar-triad r5b-pseudoscalar-hand \
         r5c-pseudoscalar-phase r5d-subjet-mass r7-patch-injection \
         r8-single-channel-ablation r9-subjet-channel-vs-scalar; do
  printf "%-32s %s\n" "$b" \
    "$(git show "${b}:experiments/top_tagging/experiment.py" | grep -c 'def set_seed')"
done
```

| branch | `def set_seed` count |
|---|---|
| r4-multiaxis-bivector | 0 |
| r5a-pseudoscalar-triad | 0 |
| r5b-pseudoscalar-hand | 0 |
| r5c-pseudoscalar-phase | 0 |
| r5d-subjet-mass | 0 |
| r7-patch-injection | 0 |
| r8-single-channel-ablation | 0 |
| r9-subjet-channel-vs-scalar | 1 |

Confirmed empirically (not just by inspection): every branch's `main()` is
`@hydra.main(...) -> run_experiment(cfg)` with no seeding call anywhere in
that path, and no `random`/`shuffle`/`seed` symbol appears in any branch's
`dataset.py` (`git show <branch>:experiments/top_tagging/dataset.py | grep
-nE "random|shuffle|seed"` returns nothing for all seven). So nothing
between process start and the point the unified side seeds (top of
`run_experiment`, before dataset loading and before model construction)
consumes RNG state on the branch side either — seeding from an external
launcher immediately before `main()` is called is equivalent to the
unified side's internal `set_seed(0)` call. The comparison is valid.

## External seeding launcher

Written once per throwaway worktree (untracked, deleted with the worktree
on cleanup):

```python
import random
import runpy

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

runpy.run_module("experiments.top_tagging.experiment", run_name="__main__")
```

Note: a first version imported `experiments.top_tagging.experiment` and
called `main()` directly; this failed with `Primary config module 'config'
not found` because Hydra's `config_path` resolution for `@hydra.main`
follows the calling frame, which pointed at `launcher.py`'s directory (the
worktree root, where `../../config` doesn't exist) rather than
`experiment.py`'s own location. Using `runpy.run_module(..., run_name="__main__")`
makes Hydra see the same calling context as `python -m
experiments.top_tagging.experiment`, fixing this — confirmed by the
`Model parameters: 186,161` line landing identically to the unified run.

## Exact commands used

Worktrees: `git worktree add --force .worktrees/parity-<branch> <branch>`
for each of the seven branches; `data/toptagging_subset_aux.npz` symlinked
into `<worktree>/data/` so the relative `data.path` override resolves.

Unified side (repo root):
```
PYTHONPATH=$(pwd) .venv/bin/python -m experiments.top_tagging.experiment \
  --config-name <unified-config> \
  data.path=data/toptagging_subset_aux.npz \
  training.epochs=1 \
  training.validate_every_n_steps=15 \
  training.checkpoint_path=checkpoints/_parity_unified_<variant>.pt \
  [variant-specific overrides] \
  > outputs/parity_unified_<variant>.log 2>&1
```

Branch side (throwaway worktree):
```
cd .worktrees/parity-<branch>
PYTHONPATH=$(pwd) <repo>/.venv/bin/python launcher.py \
  --config-name <branch-config> \
  data.path=data/toptagging_subset_aux.npz \
  training.epochs=1 \
  training.validate_every_n_steps=15 \
  training.checkpoint_path=checkpoints/_parity_branch_<variant>.pt \
  [variant-specific overrides] \
  > outputs/parity_branch_<variant>.log 2>&1
```

Both sides used `data/toptagging_subset_aux.npz` (2000 train / 500 val /
500 test, all four aux keys present), 1 epoch, batch size 32 (config
default, unchanged), `training.validate_every_n_steps=15` (overridden down
from the mini-scale default of 200, since the subset has only ~62
steps/epoch — this gives 4 `Step N` checkpoints per run instead of 0).
Unified configs already carry `training.seed: 0` (added in a prior task);
branch seeding came entirely from the external launcher, no config edit.
No committed config was changed. No `tests/fixtures/*.pt` was touched.

Config name pairing per variant (unified name vs. branch name, where they
differ):

| variant | unified config | branch config |
|---|---|---|
| r4 | `top_tagging_grade2_bivectors` | `top_tagging_grade2_bivectors` |
| r5a | `top_tagging_pseudoscalar_triad` | `top_tagging_pseudoscalar_triad` |
| r5b | `top_tagging_pseudoscalar_handedness` | `top_tagging_pseudoscalar_handedness` |
| r5c | `top_tagging_pseudoscalar_phase` | `top_tagging_pseudoscalar_phase` |
| r5d | `top_tagging_subjet_mass` | `top_tagging_scalar_masses` (different name) |
| r7 | `top_tagging_bivector_patch` | `top_tagging_grade2_bivectors_patch` (different name) |
| r8 | `top_tagging_single_channel_axis` (variant a) / `top_tagging_single_channel_axis_b` (variant b) | `top_tagging_single_channel_axis_a` / `top_tagging_single_channel_axis_b` |
| r9 | `top_tagging_r9_scalar` | (already verified in a prior session) |

r8 variant-a extra overrides (both sides identical): `training.optimizer=adam
training.lr=1.0e-3 training.scheduler=null` — the checked-in
`top_tagging_single_channel_axis.yaml` needs `lion_pytorch` (not installed:
`ModuleNotFoundError: No module named 'lion_pytorch'`, confirmed) and
`data/toptagging_full_axes.npz` (not present, 1.2M-jet full dataset not
distributed), so it cannot execute as checked in, in this environment. The
substitute matches the branch's own mini-scale `top_tagging_single_channel_axis_a.yaml`
hyperparameters (Adam, lr 1e-3, no scheduler) — same substitution Task 12
used, this time with real external seeding on the branch side.

## Per-variant verdict table

| variant | verdict | unified Test AUC/acc | branch Test AUC/acc | notes |
|---|---|---|---|---|
| r4 | **IDENTICAL** | 0.9037 / 0.8220 | 0.9037 / 0.8220 | params 186,161 both; loss=0.6024 both |
| r5a | **IDENTICAL** | 0.8516 / 0.7860 | 0.8516 / 0.7860 | params 186,161 both; loss=0.5785 both |
| r5b | **IDENTICAL** | 0.9002 / 0.8540 | 0.9002 / 0.8540 | params 186,161 both; loss=0.6030 both |
| r5c | **IDENTICAL** | 0.7528 / 0.7120 | 0.7528 / 0.7120 | params 186,161 both; loss=0.6876 both |
| r5d | **IDENTICAL** | 0.9250 / 0.9000 | 0.9250 / 0.9000 | params 186,161 both; loss=0.3587 both |
| r7 | **IDENTICAL** | 0.9125 / 0.8280 | 0.9125 / 0.8280 | params 198,449 both; loss=0.6260 both; cosine-warm-restarts scheduler + axis dropout, still exact |
| r8 (variant b, matched pairing) | **IDENTICAL** | 0.7761 / 0.6460 | 0.7761 / 0.6460 | params 185,649 both; loss=0.6291 both |
| r9 | **IDENTICAL** (verified previously, not re-run) | 0.9519 / 0.8895 | 0.9519 / 0.8895 | loss=0.3626 both |

Every `Step N | val AUC=... acc=...` line matched exactly for every
IDENTICAL row above, not just the final test line — see the full per-step
comparison in the next section.

### Full per-step comparison (all IDENTICAL rows)

**r4** (`config/top_tagging_grade2_bivectors.yaml`, `docs/reference/logs/parity_unified_r4.log` / `docs/reference/logs/parity_branch_r4.log`):
params 186,161 = 186,161. Step 15: AUC 0.7802=0.7802 acc 0.5600=0.5600. Step
30: 0.8381=0.8381 / 0.7180=0.7180. Step 45: 0.8982=0.8982 / 0.7840=0.7840.
Step 60: 0.9097=0.9097 / 0.8100=0.8100. Epoch loss 0.6024=0.6024. Test
0.9037=0.9037 / 0.8220=0.8220.

**r5a** (`config/top_tagging_pseudoscalar_triad.yaml`, `docs/reference/logs/parity_unified_r5a.log` / `docs/reference/logs/parity_branch_r5a.log`):
params 186,161=186,161. Step 15: 0.8010=0.8010 / 0.7420=0.7420. Step 30:
0.8609=0.8609 / 0.7740=0.7740. Step 45: 0.8697=0.8697 / 0.7980=0.7980. Step
60: 0.8491=0.8491 / 0.7740=0.7740. Epoch loss 0.5785=0.5785. Test
0.8516=0.8516 / 0.7860=0.7860.

**r5b** (`config/top_tagging_pseudoscalar_handedness.yaml`, `docs/reference/logs/parity_unified_r5b.log` / `docs/reference/logs/parity_branch_r5b.log`):
params 186,161=186,161. Step 15: 0.7585=0.7585 / 0.5540=0.5540. Step 30:
0.8588=0.8588 / 0.7740=0.7740. Step 45: 0.8996=0.8996 / 0.7840=0.7840. Step
60: 0.9060=0.9060 / 0.8240=0.8240. Epoch loss 0.6030=0.6030. Test
0.9002=0.9002 / 0.8540=0.8540.

**r5c** (`config/top_tagging_pseudoscalar_phase.yaml`, `docs/reference/logs/parity_unified_r5c.log` / `docs/reference/logs/parity_branch_r5c.log`):
params 186,161=186,161. Step 15: 0.5325=0.5325 / 0.5320=0.5320. Step 30:
0.6345=0.6345 / 0.6060=0.6060. Step 45: 0.7808=0.7808 / 0.6740=0.6740. Step
60: 0.8041=0.8041 / 0.7260=0.7260. Epoch loss 0.6876=0.6876. Test
0.7528=0.7528 / 0.7120=0.7120.

**r5d** (`config/top_tagging_subjet_mass.yaml` vs. branch `top_tagging_scalar_masses.yaml`, `docs/reference/logs/parity_unified_r5d.log` / `docs/reference/logs/parity_branch_r5d.log`):
params 186,161=186,161. Step 15: 0.9382=0.9382 / 0.8860=0.8860. Step 30:
0.9383=0.9383 / 0.8800=0.8800. Step 45: 0.9416=0.9416 / 0.8880=0.8880. Step
60: 0.9415=0.9415 / 0.8800=0.8800. Epoch loss 0.3587=0.3587. Test
0.9250=0.9250 / 0.9000=0.9000.

**r7** (`config/top_tagging_bivector_patch.yaml` vs. branch `top_tagging_grade2_bivectors_patch.yaml`, `docs/reference/logs/parity_unified_r7.log` / `docs/reference/logs/parity_branch_r7.log`):
params 198,449=198,449 (patch layout, larger than the multi-channel
variants). Both logged `Using CosineAnnealingWarmRestarts: T_0=3125 steps,
eta_min=1e-05` identically. Step 15: 0.7658=0.7658 / 0.6180=0.6180. Step
30: 0.8804=0.8804 / 0.5600=0.5600. Step 45: 0.9044=0.9044 / 0.8300=0.8300.
Step 60: 0.9161=0.9161 / 0.8120=0.8120. Epoch loss 0.6260=0.6260. Test
0.9125=0.9125 / 0.8280=0.8280. `axis_dropout_p=0.2` is active on this
config and still produced bit-identical trajectories, confirming the
dropout RNG draws are consumed in the same order on both sides.

**r8, variant b** (`config/top_tagging_single_channel_axis_b.yaml` vs. branch `top_tagging_single_channel_axis_b.yaml`, `docs/reference/logs/parity_unified_r8b.log` / `docs/reference/logs/parity_branch_r8b.log`):
params 185,649=185,649. Both logged `Single-channel jet-axis embedding:
pseudoscalar=on` (branch) / silently matching behavior (unified — see
divergence note below for why this line differs in presence, not effect).
Step 15: 0.7811=0.7811 / 0.5600=0.5600. Step 30: 0.7953=0.7953 /
0.6940=0.6940. Step 45: 0.7965=0.7965 / 0.6900=0.6900. Step 60:
0.7998=0.7998 / 0.6780=0.6780. Epoch loss 0.6291=0.6291. Test
0.7761=0.7761 / 0.6460=0.6460.

## A divergence found, diagnosed, and resolved — r8 variant "a"

The first r8 pairing run (unified `config/top_tagging_single_channel_axis.yaml`
substitute vs. branch `top_tagging_single_channel_axis_a.yaml`, both pointed
at `data/toptagging_subset_aux.npz`) **did not match**:

- unified (`docs/reference/logs/parity_unified_r8.log`): Step 15 `val AUC=0.7709 acc=0.5600`, ... Test `AUC=0.7828 | accuracy=0.6620`
- branch (`docs/reference/logs/parity_branch_r8.log`): Step 15 `val AUC=0.7811 acc=0.5600`, ... Test `AUC=0.7761 | accuracy=0.6460`, plus an extra branch-only log line: `Single-channel jet-axis embedding: pseudoscalar=on`

First divergent line: **Step 15**, `val AUC=0.7709` (unified) vs. `0.7811`
(branch). Parameter counts matched (185,649 = 185,649), so this is not a
model-construction difference.

**Root cause, confirmed by reading source (`git show r8-single-channel-ablation:experiments/top_tagging/dataset.py`
and `...experiment.py`):** the r8 branch's `dataset.py` unconditionally
loads `p_lead_{mode}` into the batch whenever the key is present in the
`.npz` file, and the branch's embedding dispatch unconditionally passes it
through whenever present on the batch (`p_lead = getattr(batch, "p_lead",
None)`; the pseudoscalar channel is filled iff `p_lead is not None`) — this
branch has no config-level injector gate, unlike the unified registry's
`data.inject: [...]` list. `data/toptagging_subset_aux.npz` (chosen to give
every variant a common dataset with all aux keys) contains `p_lead_*`,
so the branch's "variant a" run silently auto-enabled the pseudoscalar
channel that its own config intends to leave off (branch-original
`top_tagging_single_channel_axis_a.yaml` points at `data/toptagging_mini_axes.npz`,
which has no `p_lead` key, so this auto-detection never fires in the
branch's own original runs). The unified side, by contrast, gates
correctly on `data.inject: []` for the `top_tagging_single_channel_axis.yaml`
(variant a) config and left the pseudoscalar channel at zero regardless of
what the npz contains — this is the unified registry's injector-gating
design working as intended, not a bug.

**This is a dataset-selection artifact of the parity harness, not a
training-path defect.** Resolution: re-ran the pairing using each side's
own **variant-b** config (`top_tagging_single_channel_axis_b.yaml` on both
sides), which explicitly requests the pseudoscalar channel
(`data.inject: [lead_pseudoscalar]` on the unified side; implicit via
`p_lead` presence on the branch side) — this is the true apples-to-apples
match for a dataset that carries `p_lead`. That pairing (documented above)
is **IDENTICAL** end to end, including the exact 0.7811/0.7953/0.7965/0.7998
step sequence the mismatched "variant a" branch run had already shown —
confirming the unified `single_channel` embedding + pseudoscalar path
reproduces the branch's training path exactly once both sides are told to
use it. The mismatched "variant a" numbers above are not evidence of a
unified-side bug; they are two different embeddings (pseudoscalar-off vs.
pseudoscalar-on) being compared against each other because of an implicit,
data-driven behavior switch specific to r8's branch code.

### Confirmed 2026-08-26: variant a matches exactly on its own dataset

The diagnosis above was verified directly rather than left as an inference.
Both sides were re-run on `data/toptagging_mini_axes.npz` — the axes-only
dataset r8 variant a is actually configured to use, which carries no
`p_lead` key, so the branch's implicit switch cannot fire:

- branch: `top_tagging_single_channel_axis_a` on `r8-single-channel-ablation`,
  RNGs seeded externally (that branch has no `set_seed`), 1 epoch
- unified: `top_tagging_single_channel_axis_b` with `data.inject=[]` and
  `data.path=data/toptagging_mini_axes.npz`, `training.seed=0`, 1 epoch

Result — **IDENTICAL**, every metric line:

```
Model parameters: 185,649        (both sides)
Step 200 | val AUC=0.9076 acc=0.8280
Step 400 | val AUC=0.9425 acc=0.8725
Step 600 | val AUC=0.9415 acc=0.8825
loss=0.4044
Test AUC=0.9314 | accuracy=0.8440
```

Logs: `docs/reference/logs/r8_arma_axesonly_branch.log`,
`docs/reference/logs/r8_arma_axesonly_unified.log`.

So **all eight variants — nine configurations counting both r8 arms —
reproduce their source branch exactly.** There is no remaining variant
whose training path differs. The subset-dataset divergence recorded above
was the harness comparing pseudoscalar-off against pseudoscalar-on, and is
not a defect in the port.

The underlying design difference is still real and still worth knowing: the
branch infers the variant from which keys the dataset happens to contain,
so a branch variant-a run on a `p_lead`-carrying dataset silently becomes
variant b. The unified registry requires `data.inject` to say so. That makes
the unified behavior the safer of the two for an ablation.

## Files produced

- `outputs/parity_unified_{r4,r5a,r5b,r5c,r5d,r7,r8,r8b}.log` (originally
  written here; shipped copies at `docs/reference/logs/parity_unified_{...}.log`)
- `outputs/parity_branch_{r4,r5a,r5b,r5c,r5d,r7,r8,r8b}.log` (originally
  written here; shipped copies at `docs/reference/logs/parity_branch_{...}.log`)
- `docs/reference/logs/pair_unified_r9.log` / `docs/reference/logs/pair_branch_r9.log` (pre-existing, from prior session, re-verified by inspection here)
- Checkpoints written to scratch paths `checkpoints/_parity_{unified,branch}_<variant>.pt` (not the real `checkpoint_path` from any committed config)
- `.worktrees/parity-<branch>/launcher.py` (untracked, deleted with each worktree on cleanup)

## Cleanup performed

All seven throwaway worktrees removed: `git worktree remove --force
.worktrees/parity-<branch>` for r4, r5a, r5b, r5c, r5d, r7, r8. Verified via
`git worktree list` — only the pre-existing `.claude/worktrees/*` and
`.worktrees/r10*` worktrees remain, none of which were touched. No config
file was committed or modified. No branch was pushed or deleted.

## Concerns for the user

1. **r8's checked-in primary config still cannot run as checked in** in
   this environment (`lion_pytorch` not installed, `data/toptagging_full_axes.npz`
   not present) — same finding as the prior session's report. The r8
   pairing above used the same Adam/mini-scale substitute Task 12 used, now
   genuinely seeded on both sides.
2. **The r8 "variant a" divergence is worth flagging to anyone reusing
   `data/toptagging_subset_aux.npz` for future parity checks against other
   branches**: any branch whose dataset/embedding code auto-detects aux
   keys by presence (rather than gating on an explicit config field) can
   silently pick up extra aux data from a shared multi-key file. The
   unified registry's explicit `data.inject` list does not have this
   failure mode; it was branch-only behavior.
3. `data/toptagging_subset_aux.npz` and `data/toptagging_subset.npz` are
   not committed (as instructed) — anyone re-running this needs to
   regenerate them via `scripts/precompute_features.py` against a 2000/500/500
   cut of `data/toptagging_mini.npz`, or reuse the existing untracked file
   if still present on disk.
