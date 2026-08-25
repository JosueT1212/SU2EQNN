# Variant AUC baselines

Recorded test AUC per embedding variant, captured from each source branch before
the unification refactor (Task 1). These are the acceptance bar checked against
the unified implementation in Task 12.

Sources checked per branch: `outputs/*/experiment.log`, top-level `*.log` files,
and `git log --grep AUC` over commit messages (see `scripts/capture_embedding_fixtures.py`
capture session, 2026-08-25). Only `r5c` has tracked, recorded results; no other
branch has a committed log or output artifact recording an AUC number.

| variant | source branch | config used | recorded test AUC | run type |
|---|---|---|---|---|
| r4 | `r4-multiaxis-bivector` | `data.path=data/toptagging_mini_axes.npz`, `training.epochs=30`, `training.checkpoint_path=checkpoints/top_tagging_r4_best.pt` (untracked `outputs/r4_mini_full/.hydra/config.yaml`) | 0.9786 (accuracy=0.9245) — **untracked, not committed to `r4-multiaxis-bivector`**; see Notes | mini |
| r5a | `r5a-pseudoscalar-triad` | — | none recorded — baseline must be generated in Task 12 | — |
| r5b | `r5b-pseudoscalar-hand` | — | none recorded — baseline must be generated in Task 12 | — |
| r5c | `r5c-pseudoscalar-phase` | `data.path=data/toptagging_mini_axes_phases.npz`, `training.checkpoint_path=checkpoints/r5c_best.pt` (`outputs/pseudoscalar_phase_mini/.hydra/overrides.yaml`) | Test AUC=0.9770, accuracy=0.9260 (30 epochs; final `val AUC` around epoch 30 ranged ~0.964-0.969) | mini |
| r5d | `r5d-subjet-mass` | — | none recorded — baseline must be generated in Task 12 | — |
| r7 | `r7-patch-injection` | — | none recorded — baseline must be generated in Task 12 | — |
| r8 | `r8-single-channel-ablation` | — | none recorded — baseline must be generated in Task 12 | — |
| r9 | `r9-subjet-channel-vs-scalar` | — | none recorded — baseline must be generated in Task 12 | — |

## Notes

- `r5c`'s recorded AUC comes from `outputs/pseudoscalar_phase_mini/experiment.log`
  (identical content also duplicated at top-level `pseudoscalar_phase_mini.log`) on
  branch `r5c-pseudoscalar-phase`, final line:
  `Test AUC=0.9770 | accuracy=0.9260`.
- **`r4`'s number is untracked, working-tree-only data, not part of any branch's git
  history.** The current working tree (branch `unify-embeddings`) has untracked
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
  verified against `r4-multiaxis-bivector`'s tip. Task 12 should treat this as an
  unverified reference point, not a substitute for a baseline regenerated against
  the actual branch tip.
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
