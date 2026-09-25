# SU2EQNN

Spin(3) ≅ SU(2) equivariant transformer on Cl(3,0) for top-jet tagging, with
physics observables injected into unused grades of the input multivector.
Companion code for the draft "Equivariant Feature Injection for Top-Jet
Tagging" (`Draft/main.tex`).

Every experimental variant runs from `main`. A variant is a **config file**,
not a branch: pick a channel layout with `data.embedding` and, optionally, a
physics observable to write into a spare slot with `data.inject`.

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
  of the results table. ~6.4GB; not distributed in this repo.

### Auxiliary features

Every layout except the plain baseline needs per-jet quantities precomputed
into the `.npz` first. One script produces all of them:

```bash
.venv/bin/python scripts/precompute_features.py \
  --features axes,masses,phases,leading \
  data/toptagging_mini.npz data/toptagging_mini_aux.npz
```

Available features: `axes`, `masses`, `phases`, `leading`. Each is an
independent writer — re-running with a subset preserves keys already present.

`--features axes` is fast. **`phases` is not**: it is a pure-Python,
roughly-cubic-per-jet computation and takes on the order of hours over the
20k-jet mini training split. `data/toptagging_mini_axes.npz` (axes only)
already exists in the repo, so configs that need only `axes` can skip this
step entirely.

## Running an experiment

```bash
PYTHONPATH=$(pwd) .venv/bin/python -m experiments.top_tagging.experiment \
  --config-name <config>
```

`<config>` is a filename (without `.yaml`) under `config/`. Hydra writes logs
and checkpoints under `outputs/<date>/<time>/`; override
`training.checkpoint_path` to control where the best checkpoint lands.

Run the test suite with `pytest`.

## Variants

| Config | Layout / inject | Aux needed | What it tests |
|---|---|---|---|
| `top_tagging` | *(baseline)* | — | Cl(3,0) transformer, no injected features |
| `top_tagging_grade2_bivectors` | `multi_channel` | `axes` | 4-channel wedge injection: jet axis + 3 exclusive-$k_T$ subjet axes |
| `top_tagging_bivector_patch` | `patch` | `axes` | Same bivector signal re-injected at every block via an independent equivariant linear map, instead of only at the input |
| `top_tagging_subjet_mass` | `multi_channel` + `subjet_mass` | `axes`, `masses` | Grade 0: $[m_\mathrm{jet}, m_{s_1}, m_{s_2}, m_{s_3}]$ Casimir invariants in the scalar slot |
| `top_tagging_pseudoscalar_triad` | `multi_channel` + `triad_handedness` | `axes` | Grade 3: oriented volume $\chi=(\hat n_2\times\hat n_3)\cdot\hat n_4$ + per-constituent determinants |
| `top_tagging_pseudoscalar_handedness` | `multi_channel` + `local_handedness` | `axes` | Grade 3: per-constituent screw-sense observable $\chi_i^{(k)}$ |
| `top_tagging_pseudoscalar_phase` | `multi_channel` + `ca_phase` | `axes`, `phases` | Grade 3: signed dihedral angle between successive Cambridge–Aachen splitting planes |
| `top_tagging_single_channel_axis` | `single_channel` | `axes` | Packs the jet-axis bivector into the single existing channel instead of adding channels (full dataset) |
| `top_tagging_single_channel_axis_b` | `single_channel` + `lead_pseudoscalar` | `axes`, `leading` | As above, plus the pseudoscalar volume $I = \vec p\cdot(\hat n_\mathrm{jet}\times\hat p_\mathrm{lead})$ |
| `top_tagging_subjet_axis_as_channel` | `multi_channel` | `axes` | Highest-$p_T$ subjet axis carried as a 5th wedge channel … |
| `top_tagging_subjet_axis_as_invariant` | `invariant_subjet_axis` + `cos_theta` | `axes` | … versus the same axis reduced to the invariant $\cos\theta = \hat p\cdot\hat n$ in a spare slot of channel 0 |

The last two are a matched pair: same physical information, once as an
equivariant channel and once as an invariant scalar.

### Rest-frame (little-group) canonicalization

`data.frame: rest` boosts every jet to its own rest frame with the inverse
standard boost of its total momentum, $L(P)^{-1}$, before the unchanged
embedding. A lab Lorentz transform then reaches the network only as the
Wigner rotation $W(\Lambda, P)\in SU(2)$, which the Spin(3) network already
respects (`Draft/little-group-canonicalization.tex`, Thm 5.2).
`data.beams` picks the spurions:

| Config | Frame / beams | Symmetry of the tagger |
|---|---|---|
| `top_tagging_r1_full_b12` | `lab` / `lab` | $O(2)$ about the beam; not boost-invariant |
| `top_tagging_rest_frame_full_b12` | `rest` / `none` | exact $SO^+(1,3)$ invariance |
| `top_tagging_rest_frame_boosted_beams_full_b12` | `rest` / `boosted` | invariant under $z$-boosts × $z$-rotations; transverse boosts broken by the beam |

`boosted` tokens are the null beam directions $(1,0,0,\pm1)$ boosted to the
rest frame and normalized to a unit grade-1 vector. `frame: rest` supports
only the plain baseline: injectors, other layouts and lab-frame aux keys
raise. These three configs are full-dataset only and read the raw
`data/toptagging_full.npz` (no aux keys).

**Boost-robustness evaluation.** `evaluation.random_boost_rapidity=Y` boosts
each test jet along $z$ by a rapidity from $U[-Y, Y]$. With
`training.epochs=0` the run skips training, loads `training.checkpoint_path`
and only evaluates:

```bash
PYTHONPATH=$(pwd) .venv/bin/python -m experiments.top_tagging.experiment \
  --config-name top_tagging_rest_frame_full_b12 \
  training.epochs=0 evaluation.random_boost_rapidity=2.0
```

The rest-frame models must give the same AUC at every $Y$; the lab model need not.

For the physics behind each variant see `Draft/main.tex`; this README does not
restate it.

## The embedding registry

`experiments/top_tagging/embedding/` holds the whole variant surface:

```
embedding/
  core.py        # wedge, spurion padding, ptr/batch bookkeeping, dropout
  layouts.py     # multi_channel, single_channel, invariant_subjet_axis, patch
  injectors.py   # subjet_mass, triad_handedness, local_handedness,
                 # ca_phase, lead_pseudoscalar, cos_theta
  registry.py    # name -> (layout, injectors), with build-time validation
```

- A **layout** owns the channel count, spurion padding and interleaving, and
  any extra output tensors.
- An **injector** never allocates or reshapes; it writes into slots of an
  existing `mv`.
- Injectors compose as long as they write disjoint `(channel range, slot)`
  regions. `registry.py` validates this at build time and raises rather than
  silently overwriting, and it raises on a layout mismatch or a missing aux
  key — naming the `precompute_features.py --features` flag that produces it.

Adding a variant means adding a layout or injector and a config file; no
function signature changes, because all per-jet precomputed tensors travel in
one `aux` dict.

### Regression fixtures

`tests/fixtures/*.pt` pin each variant's embedding output bit-for-bit against
the original per-branch implementation it replaced. `test_embedding_equivalence.py`
is the acceptance gate — a mismatch there means behaviour changed, not that a
fixture needs regenerating.

`scripts/capture_embedding_fixtures.py` regenerates them from the archived
branches. It still takes the old branch shorthand (`r4`, `r5d`, …) as its
argument, because those name real git branches and their old embedding APIs;
it writes to the current fixture filenames via its `FIXTURE_NAMES` map.

## Training protocol

Mini-dataset runs: 30 epochs, Adam, lr = 1e-3, batch size 32 (Sec. IV,
`sec:exp`). Full-dataset runs: 21 epochs, Lion optimizer, cosine warm-restart
schedule.
