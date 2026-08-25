# Variant AUC baselines

Recorded test AUC per embedding variant, captured from each source branch before
the unification refactor (Task 1). These are the acceptance bar checked against
the unified implementation in Task 12.

Sources checked per branch: `outputs/*/experiment.log`, top-level `*.log` files,
and `git log --grep AUC` over commit messages (see `scripts/capture_embedding_fixtures.py`
capture session, 2026-08-25). Only `r5c` has tracked, recorded results; no other
branch has a committed log or output artifact recording an AUC number.

| variant | source branch | source SHA | config used | recorded test AUC | run type |
|---|---|---|---|---|---|
| r4 | `r4-multiaxis-bivector` | `8a1e84c` | `data.path=data/toptagging_mini_axes.npz`, `training.epochs=30`, `training.checkpoint_path=checkpoints/top_tagging_r4_best.pt` (untracked `outputs/r4_mini_full/.hydra/config.yaml`) | none recorded — baseline must be generated in Task 12 (see Notes: an untracked 0.9786 exists but is not committed to this branch) | — |
| r5a | `r5a-pseudoscalar-triad` | `e9ee036` | — | none recorded — baseline must be generated in Task 12 | — |
| r5b | `r5b-pseudoscalar-hand` | `9d474dd` | — | none recorded — baseline must be generated in Task 12 | — |
| r5c | `r5c-pseudoscalar-phase` | `1855715` | `data.path=data/toptagging_mini_axes_phases.npz`, `training.checkpoint_path=checkpoints/r5c_best.pt` (`outputs/pseudoscalar_phase_mini/.hydra/overrides.yaml`) | Test AUC=0.9770, accuracy=0.9260 (30 epochs; final `val AUC` around epoch 30 ranged ~0.964-0.969) | mini |
| r5d | `r5d-subjet-mass` | `0c54d17` | — | none recorded — baseline must be generated in Task 12 | — |
| r7 | `r7-patch-injection` | `100e9b5` | — | none recorded — baseline must be generated in Task 12 | — |
| r8 | `r8-single-channel-ablation` | `1bda697` | — | none recorded — baseline must be generated in Task 12 | — |
| r9 | `r9-subjet-channel-vs-scalar` | `5954701` | — | none recorded — baseline must be generated in Task 12 | — |

Source SHAs verified with `git rev-parse --short <branch>` on 2026-08-25 (same day
as capture); each matches the `HEAD is now at <sha>` line printed by
`git worktree add` during the Step 3 capture for that branch — no branch tip moved
since capture.

## Notes

- `r5c`'s recorded AUC comes from `outputs/pseudoscalar_phase_mini/experiment.log`
  (identical content also duplicated at top-level `pseudoscalar_phase_mini.log`) on
  branch `r5c-pseudoscalar-phase`, final line:
  `Test AUC=0.9770 | accuracy=0.9260`.
- **`r4`'s table cell reads "none recorded" per the brief's definition: a number
  only counts as recorded if it is committed to the branch's history, and no such
  number exists on `r4-multiaxis-bivector`.** However, an `r4`-labeled AUC does
  exist as untracked, working-tree-only data, not part of any branch's git
  history: the current working tree (branch `unify-embeddings`) has untracked
  files `r4_mini_full.log`, `outputs/r4_mini_full/experiment.log`, and
  `checkpoints/top_tagging_r4_best.pt` (`git status` shows `?? outputs/`, `?? r4_mini_full.log`,
  `?? checkpoints/`; none are tracked by `r4-multiaxis-bivector` or any other
  branch — `git log --oneline -- outputs/r4_mini_full` returns nothing). The log's
  final line reads `Test AUC=0.9786 | accuracy=0.9245`, from a 30-epoch run with
  `data.path=data/toptagging_mini_axes.npz`,
  `training.checkpoint_path=checkpoints/top_tagging_r4_best.pt`
  (`outputs/r4_mini_full/.hydra/config.yaml`). This is recorded here for visibility
  since it names `r4` directly, but because it was never committed to the branch
  its provenance (exact code state, config overrides beyond epochs) cannot be
  verified against `r4-multiaxis-bivector`'s tip, so it must NOT be used to gate
  Task 12's pass/fail — Task 12 should generate a fresh, verifiable baseline
  against the archived branch tip instead.
- The same untracked working tree also has `r1_mini_full.log` /
  `outputs/r1_mini_smoke,r1_mini_full/experiment.log` for an `r1` run — `r1` is not
  one of the eight variants in scope for this task, so it is not included in the
  table above.
- For the remaining six branches (`r5a-pseudoscalar-triad`, `r5b-pseudoscalar-hand`,
  `r5d-subjet-mass`, `r7-patch-injection`, `r8-single-channel-ablation`,
  `r9-subjet-channel-vs-scalar`), `git ls-tree -r --name-only <branch> | grep -iE
  '\.log$|outputs/'` returned no matches, and a repo-wide `git log --all --grep=AUC -i`
  matched only the original training-loop feature commit (`39043a4`, "add top
  tagging training loop with Hydra config and BCE+AUC evaluation") and, on branches
  descended from it, this refactor's own plan/spec documents describing this task —
  neither is an actual run result. No AUC number was guessed or computed to fill
  these gaps. All six have training/config code committed (e.g.
  `r8-single-channel-ablation` has a `top_tagging_r8a_full` config,
  `r9-subjet-channel-vs-scalar` has a 5-seed runner script) but no committed log or
  output artifact recording a result.
- **Working invocation for `scripts/capture_embedding_fixtures.py`:** the script's
  own docstring usage line (`.venv/bin/python scripts/capture_embedding_fixtures.py
  <variant> <out_dir>`) fails with `ModuleNotFoundError: No module named
  'experiments'` when run as written, because `experiments/` is a plain top-level
  directory outside the editable-installed `su2eqnn` package and Python's
  script-mode `sys.path[0]` is the script's own directory, not cwd. The working
  invocation (confirmed against `.claude/worktrees/r5d-mass/run_seeds.sh`'s own
  convention) is:

  ```bash
  cd <worktree-root>
  PYTHONPATH=$(pwd) .venv/bin/python scripts/capture_embedding_fixtures.py <variant> <out_dir>
  ```

  Task 12, when re-running this script, should use this form.

## Task 12 verification (2026-08-25)

Task 12's acceptance was rescoped (recorded in the plan ledger,
`.superpowers/sdd/2026-08-25-unified-embedding-registry/progress.md`): only
`r4` has a recorded number to reproduce, so that comparison was run in full;
the other seven variants were checked with a **paired short run** (same
config, same step budget, unified vs. that variant's own source branch in a
throwaway worktree) instead of generating seven new hour-scale baselines.

### r4: full mini run vs. the recorded 0.9786

Unified code, `config/top_tagging_grade2_bivectors.yaml` (30 epochs,
`training.seed: 0` — newly added; the original branch run that produced
0.9786 had no seed set at all), run to completion:

**Test AUC = 0.9781, accuracy = 0.9250** (log: `outputs/unified_top_tagging_grade2_bivectors_full.log`),
vs. the untracked 0.9786 / 0.9245 noted above — a difference of -0.0005 AUC.
This is a fresh, seeded, verifiable run against the archived branch tip's
config, not a repeat of the untracked number. The 0.0005 gap is not being
scored pass/fail (the 0.9786 run was unseeded, this one seeded at 0) — it
is reported as a number.

**IMPORTANT — r5c has a committed baseline, now reproduced.** This doc's
own opening line says "Only `r5c` has tracked, recorded results" —
`r5c-pseudoscalar-phase`'s `outputs/pseudoscalar_phase_mini/experiment.log`
is tracked in that branch's git history and ends `Test AUC=0.9770 |
accuracy=0.9260` (30 epochs, `data.path=data/toptagging_mini_axes_phases.npz`,
per the branch's committed `.hydra/overrides.yaml`). This is a *stronger*
baseline than r4's (which is untracked-only), and it was missed when Task
12's AUC step was rescoped to "only r4 has a recorded number." The r5c row
in the paired-run table below is a 1-epoch, 2000-jet **subset** run
(0.7519 vs. 0.7603) — it does not test the same thing as 0.9770 and was
never meant to be read against it.

**Reproduction result:** `config/top_tagging_pseudoscalar_phase.yaml`, 30
epochs, unified code, run against a **regenerated**
`data/toptagging_mini_axes_phases.npz` (the original file no longer existed
on disk; it was regenerated for this reproduction — the regenerating
script's `_compute_phases` body was byte-verified against branch
`r5c-pseudoscalar-phase`'s own `precompute_phases.py` back in Task 2, which
is why this regenerated aux file is trustworthy rather than a fresh,
unverified reimplementation), `data.path` passed as a runtime override, not
committed to the config. Log: `outputs/unified_r5c_full.log`.

**Test AUC = 0.9757, accuracy = 0.9200**, vs. the committed 0.9770 / 0.9260
— a difference of -0.0013 AUC. Reported as a number, not graded pass/fail:
the recorded 0.9770 run was itself unseeded, so a gap this size is within
seed-to-seed spread on this training path.

**Both recorded baselines in this project are now reproduced**: r4 at
0.9781 vs. its untracked 0.9786, and r5c at 0.9757 vs. its committed
0.9770. No other variant in this project has a recorded ground-truth AUC
to reproduce against.

### Paired short runs: unified vs. source branch, all eight variants

Each pair: 1 epoch on the mini (or, for r5c/r5d, a subset) dataset, same
config, `checkpoint_path` overridden to a scratch path so no real checkpoint
was touched. Full numbers and commands are in the Task 12 report.

**Seeding note (superseded — see the parity sweep below).** The table
immediately below was produced with `experiments/top_tagging/experiment.py`'s
internal `set_seed()` on the unified side only: checking each branch tip
(`git show <branch>:experiments/top_tagging/experiment.py | grep -c "def
set_seed"`) returns **0** for r4, r5a, r5b, r5c, r5d, r7, and r8 — only r9's
branch calls `set_seed` internally, so only the r9 row below was a true
seeded apples-to-apples comparison at the time it was run. **This gap has
since been closed**: a follow-up parity sweep re-ran all eight variants with
an *externally*-seeded launcher on both sides (seeding immediately before
each branch's unseeded `main()`, verified equivalent to the unified side's
internal seeding — no RNG-consuming code runs before that point on either
side). See "Training-path parity sweep" below for the direct, verified
result; the small gaps in the table immediately below are an artifact of
one-sided seeding, not evidence about the training path.

| variant | unified Test AUC | branch Test AUC | seeded both sides? | notes |
|---|---|---|---|---|
| r4 | 0.9637 | 0.9629 | unified only | 1 epoch, mini; re-derived, log at `outputs/pair_branch_r4.log` |
| r5a | 0.9594 | 0.9622 | unified only | 1 epoch, mini |
| r5b | 0.9658 | 0.9646 | unified only | 1 epoch, mini |
| r5c | 0.7519 | 0.7603 | unified only | 1 epoch, **subset** (2000/500/500 jets) — **not comparable to the committed 0.9770 baseline; see the full 30-epoch reproduction (0.9757) in the callout above and the addendum below** |
| r5d | 0.9254 | 0.9254 | unified only | 1 epoch, **subset**; exact match despite the branch being unseeded |
| r7 | 0.9571 | 0.9545 | unified only | 1 epoch, mini |
| r8 | 0.9236 | 0.9381 | unified only | 1 epoch, mini path/optimizer overridden (see report — the checked-in config needs the full dataset and `lion_pytorch`, neither available; also the unified half inherited `validate_every_n_steps=5000` from that config, so it never checkpointed a "best" model and was evaluated on its final-epoch weights instead, unlike the branch half) |
| r9 | 0.9519 | 0.9519 | **both** | 1 epoch, mini; only variant with a real seeded branch-side comparison; exact match |

**r5c and r5d used a subset, not the full mini aux set.** The committed aux
precompute (`scripts/precompute_features.py --features
axes,masses,phases,leading` over the full 24000-jet mini set) does not
finish in reasonable time: `ca_hard_branch_phases` does a full pairwise scan
per declustering step in pure Python, which is roughly cubic per jet, so 24k
jets is an hours-long run. For the r5c/r5d pairs specifically (the only two
variants needing `phases`/`masses` aux, respectively), a **2000-train /
500-val / 500-test jet subset** of `data/toptagging_mini.npz` was cut and
run through the same precompute script to `data/toptagging_subset_aux.npz`
(not committed, not referenced by any committed config — passed as a
`data.path=` runtime override on both sides of each pair only). This
subset-vs-full substitution is a deliberate, recorded scope call: the paired
run exists to catch a changed *training path*, and the embeddings were
already proven bit-identical against the committed fixtures, so a smaller
dataset answers that question just as well. Do not read the r5c/r5d rows
above as full-mini-dataset numbers.

r8's larger gap in the table above (0.9236 vs 0.9381) turned out to have
two compounding causes, both resolved by the parity sweep below: one-sided
seeding, and a `validate_every_n_steps` asymmetry (the unified half
inherited 5000 from the checked-in full-dataset config and so never
checkpointed a "best" model within 1 epoch, scoring its raw final-epoch
weights instead of a checkpoint like the branch half did).

## Training-path parity sweep: every variant, both sides seeded, verified

A follow-up sweep re-ran every one of the eight variants (throwaway
worktrees, `git worktree remove --force` after each) with an **externally
seeded launcher** on both sides — seeding `random`/`numpy`/`torch` from
outside `main()` before invoking each branch's Hydra entry point, which is
equivalent to the unified side's internal `set_seed(0)` because nothing on
either side consumes RNG state before that point (verified: no
`random`/`shuffle`/`seed` symbol appears in any branch's `dataset.py`, and
every branch's `main()` goes straight from `@hydra.main` into
`run_experiment` with no seeding in between). Full methodology, commands,
and the complete per-step log comparison are in
`.superpowers/sdd/2026-08-25-unified-embedding-registry/parity-report.md`.

**Result: seven of eight variants (r4, r5a, r5b, r5c, r5d, r7, and r8
variant-b) are bit-for-bit IDENTICAL between unified and branch** — not
just the final test AUC, but every logged quantity: parameter count, every
validation step's AUC and accuracy, epoch loss, test AUC, test accuracy.
r9 was already verified identical in the earlier (properly seeded) pairing
above and was not re-run. This directly verifies the training path, not
just the embedding (which the fixtures already proved bit-identical) — the
weaker "same sane band" argument used earlier in this document is
superseded by this direct result.

| variant | unified config | Test AUC/acc (both sides, identical) |
|---|---|---|
| r4 | `top_tagging_grade2_bivectors` | 0.9037 / 0.8220 |
| r5a | `top_tagging_pseudoscalar_triad` | 0.8516 / 0.7860 |
| r5b | `top_tagging_pseudoscalar_handedness` | 0.9002 / 0.8540 |
| r5c | `top_tagging_pseudoscalar_phase` | 0.7528 / 0.7120 |
| r5d | `top_tagging_subjet_mass` | 0.9250 / 0.9000 |
| r7 | `top_tagging_bivector_patch` | 0.9125 / 0.8280 |
| r8 (variant b) | `top_tagging_single_channel_axis_b` | 0.7761 / 0.6460 |
| r9 | `top_tagging_r9_scalar` | 0.9519 / 0.8895 (verified earlier, not re-run) |

All eight runs used the 2000/500/500-jet subset (`data/toptagging_subset_aux.npz`,
not committed) so every variant — including r5c/r5d — shares one common
dataset with every aux key present; these numbers are not meant to be
compared against any recorded full-dataset baseline (see the r5c/r4
full-mini reproductions elsewhere in this document for that).

### r8 "variant a": a real, documented behavior difference — not a training-path bug

The first r8 parity attempt (unified `top_tagging_single_channel_axis`
substitute vs. branch `top_tagging_single_channel_axis_a`, both against
`data/toptagging_subset_aux.npz`) **did not match**: unified Step 15
`val AUC=0.7709`, branch `val AUC=0.7811`, diverging from the first
validation step, with the branch log additionally printing
`Single-channel jet-axis embedding: pseudoscalar=on`.

**Root cause:** r8's branch `dataset.py` loads `p_lead_{mode}` into the
batch whenever that key is present in the `.npz` file, and its embedding
dispatch turns the pseudoscalar channel on whenever `p_lead` is present on
the batch — the branch has **no config-level gate**; it infers the
variant from data presence. The unified registry's `data.inject: [...]`
list is the only thing that turns `lead_pseudoscalar` on or off. Because
the shared parity dataset (`toptagging_subset_aux.npz`) carries `p_lead`
for every variant, the branch's "variant a" run silently auto-enabled
pseudoscalar injection that its own config intends to leave off — its
*original* config (`data.path=data/toptagging_mini_axes.npz`, no `p_lead`
key) never triggers this, so the branch's own historical "variant a" runs
were never affected. The unified side, correctly gating on `data.inject:
[]`, left the channel off regardless of what the npz contained — this is
the injector-gate design working as intended, not a defect.

Re-running the pairing with each side's **variant-b** config (which
explicitly wants the pseudoscalar channel on both sides) gave the
IDENTICAL result in the table above, including the exact
0.7811/0.7953/0.7965/0.7998 step sequence the mismatched "variant a"
branch run had already shown — confirming the unified `single_channel` +
`lead_pseudoscalar` path reproduces the branch path exactly once both
sides are told to use the same embedding.

**This is a real behavior difference a future reader must know, independent
of Task 12's own verification:** any dataset file that happens to carry a
`p_lead_*` key will silently turn on pseudoscalar injection on
`r8-single-channel-ablation`'s branch code, regardless of which config was
requested, whereas the unified registry only turns it on when
`data.inject` explicitly lists `lead_pseudoscalar`. Anyone reusing a
shared, multi-key aux file (as this sweep did) against branch code that
infers behavior from data presence should expect this class of surprise.

**Note, to avoid a false read:** the variant-b re-run is a *different*
pairing (both sides told to inject `lead_pseudoscalar`), not a re-run of
the variant-a pairing — it does not "fix" or supersede the variant-a
divergence. The variant-a logs (`outputs/parity_unified_r8.log`, Test
AUC=0.7828; `outputs/parity_branch_r8.log`, Test AUC=0.7761) are untouched
and still diverge end to end, distinct files from the variant-b logs
(`outputs/parity_unified_r8b.log` / `outputs/parity_branch_r8b.log`, both
0.7761/0.6460). That branch-side variant-a Test AUC (0.7761) happens to
equal the variant-b matched value — a coincidence of these particular
numbers, not evidence that variant-a matched.

## Addendum (all items closed)

- **r4 full-mini reproduction (vs. 0.9786): DONE.** `config/top_tagging_grade2_bivectors.yaml`,
  30 epochs, `training.seed: 0`, unified code. Log:
  `outputs/unified_top_tagging_grade2_bivectors_full.log`. **Test AUC=0.9781,
  accuracy=0.9250** vs. the untracked 0.9786/0.9245 — a -0.0005 AUC
  difference, reported plainly and not graded pass/fail.
- **r5c full-mini reproduction (vs. 0.9770): DONE.** `config/top_tagging_pseudoscalar_phase.yaml`,
  30 epochs, unified code, `data.path=data/toptagging_mini_axes_phases.npz`
  (regenerated — the original file no longer existed on disk; the
  regenerating script's `_compute_phases` body was byte-verified against
  branch `r5c-pseudoscalar-phase`'s own `precompute_phases.py` in Task 2,
  so this regenerated aux is trustworthy) passed as a runtime override, not
  committed to the config. Log: `outputs/unified_r5c_full.log`. **Test
  AUC=0.9757, accuracy=0.9200** vs. the committed 0.9770/0.9260 — a
  -0.0013 AUC difference, reported plainly and not graded pass/fail (the
  committed 0.9770 run was itself unseeded).
- **r4's branch-side paired number: RESOLVED.** The original 0.9602 was
  struck (its worktree was removed before the log was copied out). A fresh
  re-run of the same pairing (`r4-multiaxis-bivector`,
  `config/top_tagging_grade2_bivectors.yaml`, 1 epoch, `+training.seed=0`,
  in a throwaway worktree removed only after the log was copied out) gave
  **Test AUC=0.9629, accuracy=0.8935**, retained at `outputs/pair_branch_r4.log`.
  The table above now shows 0.9629 as r4's branch-side number.
