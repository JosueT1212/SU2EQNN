# SU2EQNN

Spin(3) ≅ SU(2) equivariant transformer on Cl(3,0) for top-jet tagging, with
physics observables injected into unused grades of the input multivector.
Companion code for the draft "Equivariant Feature Injection for Top-Jet
Tagging" (`Draft/main.tex`).

Each experimental variant below lives on its own git branch. `main` holds the
architecture and the unmodified Cl(3,0) baseline only; checkout the branch
for the variant you want to run or inspect.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python ≥3.9, PyTorch ≥2.0. See `pyproject.toml` for the full
dependency list (torch-geometric, hydra-core, clifford, etc).

## Data

Experiments use the standard top-tagging reference dataset
(Kasieczka et al., *The Machine Learning Landscape of Top Taggers*, SciPost
Phys. 7 (2019) 014, [doi:10.5281/zenodo.2603256](https://doi.org/10.5281/zenodo.2603256)).

- **Reduced (mini) dataset** — 20k train / 2k val / 2k test jets, used for
  every model in the paper's mini comparison table. Place the raw file at
  `data/toptagging_mini.npz` (schema: sparse 4-momenta per jet, `ptr`
  offsets, binary label).
- **Full dataset** — 1.2M training jets, used only for the full-dataset rows
  of the results table (subjet-axis bivectors, ± invariant masses). ~6.4GB;
  not distributed in this repo.

Branches that inject subjet-axis or subjet-mass features require a
precompute pass before training (see each branch's section below) —
`scripts/precompute_axes.py`, `scripts/precompute_masses.py`,
`scripts/precompute_phases.py` read the raw `.npz` and write an augmented
copy with the extra per-jet fields.

## Running an experiment

Every branch is run the same way, from the repo root on that branch:

```bash
python -m experiments.top_tagging.experiment --config-name <config>
```

`<config>` is a filename (without `.yaml`) under `config/`. Hydra writes
logs and checkpoints under `outputs/<date>/<time>/` by default; override
`training.checkpoint_path` in the config to control where the best
checkpoint is written.

Run tests on any branch with:

```bash
pytest
```

## Branches and paper sections

| Branch | Paper section | What it tests | Config |
|---|---|---|---|
| `main` | Sec. III.C (architecture) | Cl(3,0) transformer baseline, no injected features | `config/top_tagging.yaml` |
| `r4-multiaxis-bivector` | Sec. III.D.1, "Grade 2: subjet-axis bivectors" | 4-channel wedge injection: jet axis + 3 exclusive-$k_T$ subjet axes | `config/top_tagging_r4.yaml` |
| `r7-patch-injection` | Table I, "+ subjet-axis bivectors (patch, 189k)" | Same bivector signal re-injected at every block via an independent equivariant linear map, instead of only at the input embedding | `config/top_tagging_r7_patch.yaml` |
| `r8-single-channel-ablation` | Sec. III.D.4 (`sec:r8`), "Grades 2 and 3: single-channel jet-axis injection" | Packs jet-axis bivector (variant a) or jet-axis bivector + pseudoscalar volume (variant b) into the single existing channel instead of adding new channels | `config/top_tagging_r8a.yaml` (a), `config/top_tagging_r8b.yaml` (b), `config/top_tagging_r8a_full.yaml` (a, full dataset) |
| `r5a-pseudoscalar-triad` | Sec. III.D.2, "signed subjet triads" | Grade-3 injection: oriented volume $\chi=(\hat n_2\times\hat n_3)\cdot\hat n_4$ + per-constituent determinants | `config/top_tagging_r4.yaml` (pseudoscalar mode is compiled into `embedding.py` on this branch, not config-selected) |
| `r5b-pseudoscalar-hand` | Sec. III.D.2, "local handedness" | Grade-3 injection: per-constituent screw-sense observable $\chi_i^{(k)}$ | `config/top_tagging_r4.yaml` |
| `r5c-pseudoscalar-phase` | Sec. III.D.2, "splitting-plane phases" | Grade-3 injection: signed dihedral angle between successive Cambridge–Aachen splitting planes, $\sin(2\Delta\psi)$ and $\sin(\Delta\psi)$ variants | `config/top_tagging_r4.yaml` |
| `r5d-subjet-mass` | Sec. III.D.3, "Grade 0: subjet invariant masses" | Scalar-slot injection: $[m_\mathrm{jet}, m_{s_1}, m_{s_2}, m_{s_3}]$ Casimir invariants | `config/top_tagging_r4.yaml` |

Branches `r4-explore`, `r9-subjet-channel-vs-scalar`, and the `r10-*` family
(rotor-attention) are exploratory/in-progress work not yet reported in the
draft; they are not covered by the table above.

## Per-branch run commands

Run each from a fresh checkout: `git checkout <branch>`.

### `main` — baseline
```bash
# data/toptagging_mini.npz required, no precompute
python -m experiments.top_tagging.experiment --config-name top_tagging
```

### `r4-multiaxis-bivector` — subjet-axis bivectors (grade 2)
```bash
python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
python -m experiments.top_tagging.experiment --config-name top_tagging_r4
```

### `r7-patch-injection` — subjet-axis bivectors, patch variant (189k)
```bash
python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
python -m experiments.top_tagging.experiment --config-name top_tagging_r7_patch
```

### `r8-single-channel-ablation` — single-channel jet-axis injection
```bash
python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
# variant (a): bivector only, empty pseudoscalar slot
python -m experiments.top_tagging.experiment --config-name top_tagging_r8a
# variant (b): bivector + oriented-volume pseudoscalar, needs leading-constituent axis
python scripts/precompute_leading.py data/toptagging_mini_axes.npz data/toptagging_mini_axes_leading.npz
python -m experiments.top_tagging.experiment --config-name top_tagging_r8b
# variant (a), full dataset (data/toptagging_full_axes.npz, not distributed here)
python -m experiments.top_tagging.experiment --config-name top_tagging_r8a_full
```

### `r5a-pseudoscalar-triad` — signed subjet triads (grade 3)
```bash
python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
python -m experiments.top_tagging.experiment --config-name top_tagging_r4
```

### `r5b-pseudoscalar-hand` — local handedness (grade 3)
```bash
python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
python -m experiments.top_tagging.experiment --config-name top_tagging_r4
```

### `r5c-pseudoscalar-phase` — splitting-plane phases (grade 3)
```bash
python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
python scripts/precompute_phases.py data/toptagging_mini_axes.npz data/toptagging_mini_axes.npz  # adds phases_* keys in place; config reads this same file
python -m experiments.top_tagging.experiment --config-name top_tagging_r4
```

### `r5d-subjet-mass` — subjet invariant masses (grade 0)
```bash
python scripts/precompute_axes.py data/toptagging_mini.npz data/toptagging_mini_axes.npz
python scripts/precompute_masses.py data/toptagging_mini_axes.npz data/toptagging_mini_axes.npz  # adds masses_* keys in place; config reads this same file
python -m experiments.top_tagging.experiment --config-name top_tagging_r4
```

Training protocol for every mini-dataset run above: 30 epochs, Adam,
lr = 1e-3, batch size 32 (Sec. IV, `sec:exp`). Full-dataset runs use 21
epochs, the Lion optimizer, and a cosine warm-restart schedule.
