# R9: Subjet Channel-vs-Scalar Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether R4's accuracy gain over R8a comes from giving each subjet axis its own bivector channel, or just from the subjet-direction information itself, by demoting the highest-pT subjet to a per-particle scalar invariant while keeping jet_axis and the other 2 subjets as bivector channels — then comparing 5-seed mini-dataset accuracy against R4 (0.9788) and R8a (0.9804).

**Architecture:** New `embed_tagging_data_r9` function in `experiments/top_tagging/embedding.py` builds a 4-channel Cl(3,0) multivector per particle (momentum + one scalar invariant in channel 0, three bivector channels for jet_axis/subjet_mid/subjet_soft), reusing the axes array `precompute_axes.py` already produces — no new precompute step. `experiments/top_tagging/experiment.py` gains a small embedding-variant dispatcher (`cfg.data.embedding == "r9"`) and a seed-setting helper (currently absent on this branch, needed for reproducible multi-seed runs). A new Hydra config and a shell runner execute 5 seeds on the mini dataset.

**Tech Stack:** PyTorch, PyTorch Geometric, Hydra/OmegaConf, `clifford` (Cl(3,0) basis), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-r9-subjet-channel-vs-scalar-design.md`
- Only one variable changes vs R4's axis set: highest-pT subjet moves from bivector channel to scalar feature. jet_axis and the other 2 subjets are untouched.
- Reuses existing `data/toptagging_mini_axes.npz` (axes shape `(N, 4, 3)`: `axes[:,0]`=jet_axis, `axes[:,1..3]`=subjets sorted pT descending). No changes to `scripts/precompute_axes.py` or `su2eqnn/utils/subjets.py`.
- Mini dataset only (`config/top_tagging_r4.yaml`'s data path), 30 epochs, batch_size 32, lr 1e-3 — same training hyperparameters as R4 mini, so the only difference in the final comparison is the embedding.
- All new code must pass the existing equivariance-test conventions in `tests/test_equivariance.py` (Spin(3) sandwich `w x w^-1` via `random_spin3`/`spin3_action`/`spin3_action_channels` helpers already defined there — reuse them, don't redefine).

---

### Task 1: `embed_tagging_data_r9` embedding function

**Files:**
- Modify: `experiments/top_tagging/embedding.py`
- Test: `tests/test_top_tagging.py`

**Interfaces:**
- Produces: `embed_tagging_data_r9(fourmomenta: Tensor, ptr: Tensor, jet_axis: Tensor, subjet_axes_2: Tensor, subjet_axis_highest: Tensor, two_beams: bool = True, rescale: bool = True) -> dict` with keys `"mv"` `(n_total, 4, 8)`, `"batch"` `(n_total,)`, `"is_beam"` `(n_total,)` — same dict shape as `embed_tagging_data`/`embed_tagging_data_r8`.
  - `jet_axis`: `(batchsize, 3)`
  - `subjet_axes_2`: `(batchsize, 2, 3)` — order is [subjet_mid, subjet_soft] (2nd/3rd-pT)
  - `subjet_axis_highest`: `(batchsize, 3)` — 1st-pT (highest) subjet

- [ ] **Step 1: Write the failing shape/value test**

Append to `tests/test_top_tagging.py` (near the other `embed_tagging_data*` tests, after `test_embed_tagging_data_batch_indices`):

```python
def test_embed_tagging_data_r9_shape_and_channels():
    from experiments.top_tagging.embedding import embed_tagging_data_r9, UNITS

    fourmomenta = torch.tensor([
        [40.0, 3.0, 0.0, 4.0],
        [10.0, 0.0, 1.0, 2.0],
        [20.0, 1.0, 1.0, 5.0],
    ])
    ptr = torch.tensor([0, 2, 3])  # jet0 = 2 particles, jet1 = 1
    jet_axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    subjet_axes_2 = torch.zeros(2, 2, 3)
    subjet_axes_2[:, :, 2] = 1.0  # both +z
    subjet_axis_highest = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    out = embed_tagging_data_r9(
        fourmomenta, ptr, jet_axis, subjet_axes_2, subjet_axis_highest,
        two_beams=True, rescale=True,
    )
    mv = out["mv"]
    assert mv.shape[1] == 4, f"expected 4 channels, got {mv.shape[1]}"

    particles = mv[~out["is_beam"]]  # (3, 4, 8)
    p = fourmomenta / UNITS
    # channel 0: momentum in 0-3, scalar cos-theta at idx4 (n=+z -> cos = pz/|p|)
    assert torch.allclose(particles[:, 0, 0], p[:, 0], atol=1e-6)
    assert torch.allclose(particles[:, 0, 1:4], p[:, 1:4], atol=1e-6)
    p3 = p[:, 1:4]
    expected_cos = p3[:, 2] / p3.norm(dim=-1).clamp_min(1e-12)
    assert torch.allclose(particles[:, 0, 4], expected_cos, atol=1e-5)
    assert particles[:, 0, 5:8].abs().max() < 1e-6, "only idx4 used for the scalar"

    # channels 1-3: pure grade-2 wedges against +z (e12=0, e13=px, e23=py, matching
    # the existing embed_tagging_data axes-branch convention)
    for k in range(1, 4):
        assert particles[:, k, 4].abs().max() < 1e-6
        assert torch.allclose(particles[:, k, 5], p3[:, 0], atol=1e-6)
        assert torch.allclose(particles[:, k, 6], p3[:, 1], atol=1e-6)
        assert particles[:, k, 0:4].abs().max() < 1e-6, "wedge channels must be pure grade-2"
        assert particles[:, k, 7].abs().max() < 1e-6

    beams = mv[out["is_beam"]]
    assert beams[:, 1:].abs().max() < 1e-6


def test_embed_tagging_data_r9_matches_k3_wedges():
    """ch1-3 must equal embed_tagging_data's axes-branch output for the same 3 axes."""
    from experiments.top_tagging.embedding import embed_tagging_data, embed_tagging_data_r9

    torch.manual_seed(0)
    fourmomenta = torch.randn(6, 4)
    ptr = torch.tensor([0, 3, 6])
    jet_axis = torch.randn(2, 3); jet_axis /= jet_axis.norm(dim=-1, keepdim=True)
    subjet_mid = torch.randn(2, 3); subjet_mid /= subjet_mid.norm(dim=-1, keepdim=True)
    subjet_soft = torch.randn(2, 3); subjet_soft /= subjet_soft.norm(dim=-1, keepdim=True)
    subjet_highest = torch.randn(2, 3); subjet_highest /= subjet_highest.norm(dim=-1, keepdim=True)
    subjet_axes_2 = torch.stack([subjet_mid, subjet_soft], dim=1)

    out_r9 = embed_tagging_data_r9(
        fourmomenta, ptr, jet_axis, subjet_axes_2, subjet_highest,
        two_beams=False, rescale=False,
    )
    axes_k3 = torch.stack([jet_axis, subjet_mid, subjet_soft], dim=1)  # (2, 3, 3)
    out_ref = embed_tagging_data(fourmomenta, ptr, two_beams=False, rescale=False, axes=axes_k3)

    assert torch.allclose(out_r9["mv"][:, 1:4, :], out_ref["mv"][:, 1:4, :], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k "r9" -v`
Expected: FAIL with `ImportError: cannot import name 'embed_tagging_data_r9'`

- [ ] **Step 3: Implement `embed_tagging_data_r9`**

Add to `experiments/top_tagging/embedding.py`, after `embed_tagging_data_r8` and before `apply_axis_channel_dropout`:

```python
def embed_tagging_data_r9(
    fourmomenta: Tensor,
    ptr: Tensor,
    jet_axis: Tensor,
    subjet_axes_2: Tensor,
    subjet_axis_highest: Tensor,
    two_beams: bool = True,
    rescale: bool = True,
) -> dict:
    """Embed sparse 4-momenta as 4-channel Cl(3,0) MVs (R9 layout).

    Channel 0: (E, px, py, pz, cos_theta, 0, 0, 0) where cos_theta =
    p_hat . n_hat_sub_highest is the Spin(3)-invariant cosine between each
    particle's momentum direction and the highest-pT subjet axis.
    Channels 1-3: pure grade-2 wedges p x n_hat for jet_axis, subjet_mid,
    subjet_soft respectively (same convention as embed_tagging_data's axes
    branch).

    Parameters
    ----------
    fourmomenta : Tensor shape (n_particles, 4)
    ptr         : Tensor shape (batchsize+1,)
    jet_axis    : Tensor shape (batchsize, 3)
    subjet_axes_2 : Tensor shape (batchsize, 2, 3) — [subjet_mid, subjet_soft]
    subjet_axis_highest : Tensor shape (batchsize, 3)
    two_beams   : bool
    rescale     : bool

    Returns
    -------
    dict with keys "mv" (n_total, 4, 8), "batch" (n_total,), "is_beam" (n_total,)
    """
    batchsize = len(ptr) - 1
    n_channels = 4

    if rescale:
        fourmomenta = fourmomenta / UNITS

    mv = torch.zeros(
        fourmomenta.shape[0], n_channels, 8,
        dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    mv[:, 0, 0] = fourmomenta[:, 0]
    mv[:, 0, 1:4] = fourmomenta[:, 1:4]

    particle_batch = get_batch_from_ptr(ptr)
    p = fourmomenta[:, 1:4]                                 # (n_particles, 3)
    p_norm = p.norm(dim=-1).clamp_min(1e-12)

    ax_highest = subjet_axis_highest.to(fourmomenta.dtype)[particle_batch]
    mv[:, 0, 4] = (p * ax_highest).sum(dim=-1) / p_norm

    axes = torch.cat(
        [jet_axis.unsqueeze(1), subjet_axes_2], dim=1
    ).to(fourmomenta.dtype)                                 # (batchsize, 3, 3)
    ax = axes[particle_batch]                               # (n_particles, 3, 3)
    p_ = p.unsqueeze(1)                                     # (n_particles, 1, 3)
    mv[:, 1:, 4] = p_[..., 0] * ax[..., 1] - p_[..., 1] * ax[..., 0]
    mv[:, 1:, 5] = p_[..., 0] * ax[..., 2] - p_[..., 2] * ax[..., 0]
    mv[:, 1:, 6] = p_[..., 1] * ax[..., 2] - p_[..., 2] * ax[..., 1]

    spurions = embed_spurions(
        two_beams=two_beams, device=fourmomenta.device, dtype=fourmomenta.dtype,
    )
    n_spurions = spurions.shape[0]
    spurions_full = torch.zeros(
        n_spurions, n_channels, 8, dtype=mv.dtype, device=mv.device
    )
    spurions_full[:, 0, :] = spurions[:, 0, :]

    arange = torch.arange(batchsize, device=ptr.device)
    spurion_idxs = torch.stack(
        [ptr[:-1] + i for i in range(n_spurions)], dim=0
    ) + n_spurions * arange
    spurion_idxs = spurion_idxs.permute(1, 0).flatten()

    n_total = mv.shape[0] + n_spurions * batchsize
    is_beam = torch.zeros(n_total, dtype=torch.bool, device=mv.device)
    is_beam[spurion_idxs] = True

    mv_out = torch.empty(n_total, n_channels, 8, dtype=mv.dtype, device=mv.device)
    mv_out[~is_beam] = mv
    mv_out[is_beam] = spurions_full.repeat(batchsize, 1, 1)

    ptr_out = ptr.clone()
    ptr_out[1:] = ptr[1:] + (arange + 1) * n_spurions
    batch = get_batch_from_ptr(ptr_out)

    return {"mv": mv_out, "batch": batch, "is_beam": is_beam}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k "r9" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/top_tagging/embedding.py tests/test_top_tagging.py
git commit -m "feat: add embed_tagging_data_r9 (subjet channel-vs-scalar embedding)"
```

---

### Task 2: Equivariance test for `embed_tagging_data_r9`

**Files:**
- Modify: `tests/test_equivariance.py`

**Interfaces:**
- Consumes: `embed_tagging_data_r9` from Task 1; `random_spin3`, `spin3_action`, `spin3_action_channels` already defined in this file.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_equivariance.py`, after `test_apply_axis_channel_dropout` and before the `if __name__ == "__main__":` block:

```python
def test_embed_tagging_data_r9_equivariance():
    """Rotating momenta AND axes: ch0 scalar invariant, ch1-3 rotate by rho(w)."""
    from experiments.top_tagging.embedding import embed_tagging_data_r9
    from su2eqnn.interface.vector import embed_vector, extract_vector

    def rotate3(w, v):
        return torch.stack([extract_vector(spin3_action(w, embed_vector(v[i])))
                            for i in range(v.shape[0])])

    for _ in range(10):
        fourmomenta = torch.randn(6, 4)
        ptr = torch.tensor([0, 3, 6])
        jet_axis = torch.randn(2, 3); jet_axis /= jet_axis.norm(dim=-1, keepdim=True)
        subjet_axes_2 = torch.randn(2, 2, 3)
        subjet_axes_2 /= subjet_axes_2.norm(dim=-1, keepdim=True)
        subjet_highest = torch.randn(2, 3); subjet_highest /= subjet_highest.norm(dim=-1, keepdim=True)
        w = random_spin3()

        out = embed_tagging_data_r9(
            fourmomenta, ptr, jet_axis, subjet_axes_2, subjet_highest,
            two_beams=False, rescale=False,
        )

        rot_p = fourmomenta.clone()
        rot_p[:, 1:4] = rotate3(w, fourmomenta[:, 1:4])
        rot_jet_axis = rotate3(w, jet_axis)
        rot_subjet_axes_2 = torch.stack(
            [rotate3(w, subjet_axes_2[:, k, :]) for k in range(2)], dim=1
        )
        rot_subjet_highest = rotate3(w, subjet_highest)

        out_rot = embed_tagging_data_r9(
            rot_p, ptr, rot_jet_axis, rot_subjet_axes_2, rot_subjet_highest,
            two_beams=False, rescale=False,
        )

        expected = spin3_action_channels(w, out["mv"])

        # ch0 scalar (idx4) must be invariant: rotating both p and n together
        # leaves p_hat . n_hat unchanged, and spin3_action leaves a pure
        # grade-0 slot fixed too, so this is implied by the full check below
        # but assert it directly for a clearer failure message.
        assert torch.allclose(out_rot["mv"][:, 0, 4], out["mv"][:, 0, 4], atol=1e-4), \
            "R9 scalar channel not invariant under simultaneous rotation"

        assert torch.allclose(out_rot["mv"], expected, atol=1e-4), \
            f"R9 embedding equivariance failed: {(out_rot['mv'] - expected).abs().max():.2e}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_equivariance.py -k "r9" -v`
Expected: FAIL with `ImportError` (Task 1 not yet merged in this test run context) — if Task 1 already committed, this step instead confirms the test collects and passes immediately; in that case skip to Step 3 confirmation only.

- [ ] **Step 3: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_equivariance.py -k "r9" -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_equivariance.py
git commit -m "test: add Spin(3) equivariance check for embed_tagging_data_r9"
```

---

### Task 3: Seed support + embedding-variant dispatch in `experiment.py`

**Files:**
- Modify: `experiments/top_tagging/experiment.py`
- Test: `tests/test_top_tagging.py`

**Interfaces:**
- Produces: `set_seed(seed: int | None) -> None` (module-level function in `experiments/top_tagging/experiment.py`); `get_embedding(cfg, batch) -> dict` (module-level function, dispatches on `getattr(cfg.data, "embedding", "default")`, returns the same dict shape used everywhere else — `"default"`/unset → `embed_tagging_data(...)`, `"r9"` → `embed_tagging_data_r9(...)` using `batch.axes[:, 0, :]` as jet_axis, `batch.axes[:, 2:4, :]` as subjet_axes_2, `batch.axes[:, 1, :]` as subjet_axis_highest).
- Consumes: `embed_tagging_data_r9` from Task 1.

- [ ] **Step 1: Write the failing test for `set_seed`**

Append to `tests/test_top_tagging.py`, after `test_warm_restart_scheduler_resets_lr`:

```python
def test_set_seed_reproducible():
    from experiments.top_tagging.experiment import set_seed

    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_none_is_noop():
    from experiments.top_tagging.experiment import set_seed
    set_seed(None)  # must not raise


def test_get_embedding_default_matches_embed_tagging_data():
    from experiments.top_tagging.experiment import get_embedding
    from experiments.top_tagging.embedding import embed_tagging_data

    class Batch:
        x = torch.randn(6, 4)
        ptr = torch.tensor([0, 3, 6])
        axes = None

    class Cfg:
        class data:
            two_beams = True
            rescale_data = True

    out = get_embedding(Cfg(), Batch())
    ref = embed_tagging_data(Batch.x, Batch.ptr, two_beams=True, rescale=True, axes=None)
    assert torch.allclose(out["mv"], ref["mv"])


def test_get_embedding_r9_dispatch():
    from experiments.top_tagging.experiment import get_embedding
    from experiments.top_tagging.embedding import embed_tagging_data_r9

    class Batch:
        x = torch.randn(6, 4)
        ptr = torch.tensor([0, 3, 6])
        axes = torch.randn(2, 4, 3)

    class Cfg:
        class data:
            two_beams = True
            rescale_data = True
            embedding = "r9"

    out = get_embedding(Cfg(), Batch())
    ref = embed_tagging_data_r9(
        Batch.x, Batch.ptr,
        jet_axis=Batch.axes[:, 0, :],
        subjet_axes_2=Batch.axes[:, 2:4, :],
        subjet_axis_highest=Batch.axes[:, 1, :],
        two_beams=True, rescale=True,
    )
    assert torch.allclose(out["mv"], ref["mv"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k "set_seed or get_embedding" -v`
Expected: FAIL with `ImportError: cannot import name 'set_seed'` / `'get_embedding'`

- [ ] **Step 3: Implement `set_seed` and `get_embedding`, wire into `run_experiment`/`evaluate`**

In `experiments/top_tagging/experiment.py`:

1. Add imports at the top (alongside the existing `embed_tagging_data` import):

```python
from experiments.top_tagging.embedding import (
    embed_tagging_data,
    embed_tagging_data_r9,
    apply_axis_channel_dropout,
)
```

2. Add two new module-level functions, placed after the `log = logging.getLogger(__name__)` line and before `build_model`:

```python
def set_seed(seed) -> None:
    """Seed random/numpy/torch RNGs for reproducibility. No-op if seed is None."""
    if seed is None:
        return
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log.info(f"Seeded RNGs with seed={seed}")


def get_embedding(cfg, batch) -> dict:
    """Dispatch to the configured embedding variant.

    "r9" -> embed_tagging_data_r9 (highest-pT subjet as scalar, jet_axis +
    2 remaining subjets as bivector channels). Anything else (default/unset)
    -> embed_tagging_data (R1/R4/R5 multi-channel axes path, unchanged).
    """
    variant = getattr(cfg.data, "embedding", "default")
    if variant == "r9":
        axes = batch.axes
        return embed_tagging_data_r9(
            batch.x, batch.ptr,
            jet_axis=axes[:, 0, :],
            subjet_axes_2=axes[:, 2:4, :],
            subjet_axis_highest=axes[:, 1, :],
            two_beams=cfg.data.two_beams,
            rescale=cfg.data.rescale_data,
        )
    return embed_tagging_data(
        batch.x, batch.ptr,
        two_beams=cfg.data.two_beams,
        rescale=cfg.data.rescale_data,
        axes=getattr(batch, "axes", None),
    )
```

3. In `evaluate(...)`, replace the existing inline `embed_tagging_data(...)` call with:

```python
            embedding = get_embedding(cfg, batch)
```

4. In `run_experiment(...)`, at the very start of the function body (before `device = torch.device(...)`), add:

```python
    set_seed(getattr(cfg.training, "seed", None))
```

5. In the training loop inside `run_experiment(...)`, replace the existing inline `embed_tagging_data(...)` call with:

```python
            embedding = get_embedding(cfg, batch)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -k "set_seed or get_embedding" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full existing test_top_tagging.py suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/test_top_tagging.py -v`
Expected: all PASS (existing tests untouched by these edits, `embed_tagging_data` calls in `evaluate`/training loop are behavior-preserving for the default variant)

- [ ] **Step 6: Commit**

```bash
git add experiments/top_tagging/experiment.py tests/test_top_tagging.py
git commit -m "feat: add seed support and embedding-variant dispatch (r9) to experiment.py"
```

---

### Task 4: R9 Hydra config

**Files:**
- Create: `config/top_tagging_r9.yaml`

**Interfaces:**
- Consumes: `cfg.data.embedding` field read by `get_embedding` (Task 3); `cfg.model.in_mv_channels` read by `build_model`.

- [ ] **Step 1: Create the config**

```yaml
data:
  path: data/toptagging_mini_axes.npz
  rescale_data: true
  two_beams: true
  embedding: r9

model:
  in_mv_channels: 4
  hidden_mv_channels: 16
  num_blocks: 6
  num_heads: 4

training:
  epochs: 30
  batch_size: 32
  lr: 1.0e-3
  validate_every_n_steps: 200
  checkpoint_path: checkpoints/top_tagging_r9_best.pt

evaluation:
  batch_size: 64
```

Write to `config/top_tagging_r9.yaml`.

- [ ] **Step 2: Verify Hydra loads it without error**

Run: `.venv/bin/python -c "
from hydra import compose, initialize
with initialize(version_base=None, config_path='config'):
    cfg = compose(config_name='top_tagging_r9')
    assert cfg.model.in_mv_channels == 4
    assert cfg.data.embedding == 'r9'
    print('OK', cfg.data.path, cfg.model.in_mv_channels)
"`

Expected: prints `OK data/toptagging_mini_axes.npz 4` with no errors.

- [ ] **Step 3: Commit**

```bash
git add config/top_tagging_r9.yaml
git commit -m "feat: add R9 config (subjet channel-vs-scalar ablation, mini)"
```

---

### Task 5: One-shot smoke run + 5-seed runner script

**Files:**
- Create: `scripts/run_seeds_r9.sh`

**Interfaces:**
- Consumes: `config/top_tagging_r9.yaml` (Task 4), `experiments.top_tagging.experiment` module entry point (`python -m experiments.top_tagging.experiment`), `+training.seed=<int>` Hydra override (Task 3's `set_seed`).

- [ ] **Step 1: Smoke-test one short run before committing to 5 full seeds**

Run (from `SU2EQNN/` root, ~30s-2min on CPU with epochs capped):

```bash
PYTHONPATH=$PWD .venv/bin/python -m experiments.top_tagging.experiment \
  --config-name top_tagging_r9 \
  training.epochs=1 \
  +training.seed=0 \
  training.checkpoint_path=checkpoints/top_tagging_r9_smoke.pt \
  hydra.run.dir=outputs/r9_smoke
```

Expected: completes without error, logs `Test AUC=` and `accuracy=` at the end, confirming the 4-channel model trains end-to-end on the R9 embedding.

- [ ] **Step 2: Write the 5-seed runner script**

Create `scripts/run_seeds_r9.sh` (mirrors the existing `run_seeds.sh` pattern used for R4/R5d):

```bash
#!/bin/bash
# 5-seed sweep: R9 (subjet channel-vs-scalar ablation) on mini, local CPU.
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
for seed in 0 1 2 3 4; do
  tag="r9_s${seed}"
  echo "=== $tag start $(date) ==="
  PYTHONPATH=$PWD $PY -m experiments.top_tagging.experiment \
    --config-name top_tagging_r9 \
    +training.seed=$seed \
    training.checkpoint_path=checkpoints/${tag}_best.pt \
    hydra.run.dir=outputs/seeds/$tag \
    > seeds_${tag}.log 2>&1
  echo "=== $tag done: $(grep 'Test AUC' seeds_${tag}.log)"
done
echo "ALL SEEDS DONE $(date)"
```

- [ ] **Step 3: Make it executable and commit**

```bash
chmod +x scripts/run_seeds_r9.sh
git add scripts/run_seeds_r9.sh
git commit -m "feat: add 5-seed runner script for R9 ablation"
```

- [ ] **Step 4: Launch the 5-seed sweep**

Run: `scripts/run_seeds_r9.sh > run_seeds_r9.log 2>&1 &`

This runs in the background (5 sequential mini-dataset trainings, similar wall-clock to the existing R4/R5d 3-seed sweeps scaled to 5 seeds). Do not block on it — check `seeds_r9_s*.log` files for `Test AUC=` / `accuracy=` lines once complete.

- [ ] **Step 5: Collect results into the comparison table**

Once all 5 logs show `Test AUC=`/`accuracy=` lines:

```bash
grep 'Test AUC' seeds_r9_s*.log
```

Compute mean±std of `accuracy` across the 5 seeds. Compare against R4 mini (0.9788) and R8a mini (0.9804). Save this table to memory (`r9-subjet-channel-vs-scalar-results.md`, type `project`) per the project's existing convention for R4/R7/R8 result memories — do this as a follow-up conversation turn once results are in, not as a plan step (results don't exist yet).

---

## Self-Review Notes

- **Spec coverage**: layout (ch0 scalar + ch1-3 bivectors) → Task 1; equivariance → Task 2; config/wiring → Tasks 3-4; 5-seed mini run → Task 5. All spec sections covered.
- **Type consistency**: `embed_tagging_data_r9` signature matches between Task 1's implementation, Task 2's test, and Task 3's `get_embedding` call site (`jet_axis`, `subjet_axes_2`, `subjet_axis_highest` keyword names identical throughout).
- **No placeholders**: every step has literal code/commands, no "add tests for the above" stand-ins.
- **Seed prerequisite**: confirmed via direct inspection that this branch's `experiment.py` has no seed-setting code (unlike the `r5d-mass`/`r8-single-channel-ablation` worktrees, which added it independently) — Task 3 adds it here since Task 5's 5-seed run depends on it.
