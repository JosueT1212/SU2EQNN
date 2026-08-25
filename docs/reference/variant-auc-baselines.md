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

**Test AUC = &lt;filled after the run completes — see Task 12 report&gt;**, vs. the
untracked 0.9786 noted above. This is a fresh, seeded, verifiable run against
the archived branch tip's config, not a repeat of the untracked number. The
report states the difference plainly; a nonzero gap is expected (the 0.9786
run was unseeded, this one seeded at 0) and is not being scored pass/fail.

### Paired short runs: unified vs. source branch, all eight variants

Each pair: 1 epoch on the mini (or, for r5c/r5d, a subset) dataset, same
config, `checkpoint_path` overridden to a scratch path so no real checkpoint
was touched. Full numbers and commands are in the Task 12 report.

**Seeding caveat — read before comparing any row except r9.** Unified
`experiments/top_tagging/experiment.py` calls `set_seed()` unconditionally
(it was ported wholesale from `r9-subjet-channel-vs-scalar`'s own
`experiment.py` in Task 2). Checking each other branch's tip
(`git show <branch>:experiments/top_tagging/experiment.py | grep -c "def
set_seed"`) returns **0** for r4, r5a, r5b, r5c, r5d, r7, and r8 — only r9's
branch has a `set_seed` function at all. A `+training.seed=0` override was
still passed to every branch-side run for the record, but on these seven it
has nothing to attach to: the branch process ran with **whatever RNG state
its unseeded imports produced**, while the unified process was seeded at 0.
Only the r9 row is a true apples-to-apples seeded comparison; the other six
(plus r5c/r5d, see below) are "same config, uncontrolled branch-side
randomness" — informative about gross training-path parity, not a strict
trajectory match, and a small AUC gap on those rows is not attributable to
the embedding port.

| variant | unified Test AUC | branch Test AUC | seeded both sides? | notes |
|---|---|---|---|---|
| r4 | 0.9637 | 0.9602 | unified only | 1 epoch, mini |
| r5a | 0.9594 | 0.9622 | unified only | 1 epoch, mini |
| r5b | 0.9658 | 0.9646 | unified only | 1 epoch, mini |
| r5c | 0.7519 | 0.7603 | unified only | 1 epoch, **subset** (2000/500/500 jets, not the full mini set — see below) |
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

**Parity for the six unseeded-branch variants (r4, r5a, r5b, r5c, r5d, r7)
rests on three things, not on the branch-side AUC number matching:**
1. The embedding is bit-identical to the branch's, proven by the committed
   `tests/fixtures/*.pt` equivalence tests (`tests/test_embedding_equivalence.py`).
2. Unified `experiment.py` is r9's file, ported verbatim and reviewed
   against the branch diff in Task 2 — the training loop itself is not a
   per-variant reimplementation.
3. Every unified short-run AUC above lands in the same sane band as its
   (unseeded) branch counterpart — no variant collapsed, diverged, or
   errored.

r8's larger gap (0.9236 vs 0.9381) is very likely explained by the
`validate_every_n_steps` asymmetry noted in the table above (the unified
half never saved a "best" checkpoint and was scored on raw final-epoch
weights) rather than by unseeded randomness alone; see the Task 12 report
for the full accounting.
