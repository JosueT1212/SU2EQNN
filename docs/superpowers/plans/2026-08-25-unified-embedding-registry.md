# Unified Embedding Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse eight experiment branches (r4, r5a-d, r7, r8, r9) into one `main` where the variant is chosen by config, with byte-identical behaviour to the branch it replaces.

**Architecture:** A new `experiments/top_tagging/embedding/` package splits the current monolithic module into a shared `core` (wedge, spurion interleaving, ptr bookkeeping, dropout), four `layouts` that own channel structure, and six `injectors` that write a single slot each. A `registry` maps a config name to `(layout, [injectors])` and validates slot collisions and layout compatibility before any tensor is allocated.

**Tech Stack:** Python 3.9+, PyTorch 2.12, torch-geometric, hydra/omegaconf, pytest 9, numpy. Virtualenv at `.venv`.

**Spec:** `docs/superpowers/specs/2026-08-25-unified-embedding-registry-design.md`

## Global Constraints

- Test runner is `.venv/bin/python -m pytest`. Bare `python` and bare `pytest` are not on PATH — never write them into a command.
- Work happens on a new branch `unify-embeddings` cut from `main`. Do not commit to `main` directly; the merge is a separate decision at the end.
- `UNITS = 20.0` (GeV divisor). Multivector slot layout is `[scalar, e1, e2, e3, e12, e13, e23, e123]` — indices 4/5/6 are the bivector components in that exact order, index 7 is the pseudoscalar.
- Beam spurion tokens carry zeros in every channel except channel 0.
- Injectors write in place and return `None`. They never allocate, reshape, or change channel count.
- Physics code is copied verbatim from its source branch. Do not "improve" a formula, reorder a `clamp_min`, or vectorize a loop — bit-exactness against the captured fixtures is the acceptance gate. `r5b`'s `clamp_min(1e-3)` / `clamp_min(1e-12)` ordering is load-bearing.
- `r4-explore` and the three `r10*` branches are out of scope. `r10*` branches stay alive and untouched. `config/top_tagging_r4_explore.yaml` is not carried forward.
- Never propagate to `main`: r5c's committed `outputs/pseudoscalar_phase_mini/` and its logs, the `README.md` deletion riding in r4-explore/r5a/r5b.

---

### Task 1: Capture equivalence fixtures and AUC baselines from every branch

Fixtures are the only defence against silent numerical drift, and they can only be captured while the branches still exist. Nothing else starts until this task is committed.

**Files:**
- Create: `scripts/capture_embedding_fixtures.py`
- Create: `tests/fixtures/` (8 `.pt` files, committed)
- Create: `docs/reference/variant-auc-baselines.md`

**Interfaces:**
- Produces: `tests/fixtures/<variant>.pt`, each a dict with keys `inputs` (dict of `fourmomenta`, `ptr`, plus aux tensors) and `outputs` (dict of `mv`, `batch`, `is_beam`, and `patch_mv` for r7). Variant names: `r4`, `r5a`, `r5b`, `r5c`, `r5d`, `r7`, `r8`, `r9`.
- Produces: `docs/reference/variant-auc-baselines.md` — the recorded AUC per variant, used as the acceptance bar in Task 12.

- [ ] **Step 1: Create the branch**

```bash
cd /Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN
git checkout main
git checkout -b unify-embeddings
```

- [ ] **Step 2: Write the capture script**

Create `scripts/capture_embedding_fixtures.py`. It is deliberately branch-agnostic: it looks up whichever embedding function the branch provides and feeds it a deterministic input.

```python
"""Capture a branch's embedding output as a regression fixture.

Run from inside a worktree checked out to the variant's branch:

    .venv/bin/python scripts/capture_embedding_fixtures.py r5d /abs/path/to/tests/fixtures

Writes <out_dir>/<variant>.pt containing both the inputs the branch consumed
and the outputs it produced, so the unified implementation can be checked
against it without re-deriving aux tensors.
"""

import sys

import torch

from experiments.top_tagging import embedding as emb

N_JETS = 3
COUNTS = [7, 4, 9]          # particles per jet
K = 4                        # jet axis + 3 subjet axes
SEED = 20260825


def make_inputs():
    """Deterministic (fourmomenta, ptr, aux) shared by every variant."""
    g = torch.Generator().manual_seed(SEED)
    n = sum(COUNTS)
    p3 = torch.randn(n, 3, generator=g, dtype=torch.float32) * 30.0
    e = p3.norm(dim=-1, keepdim=True) + torch.rand(n, 1, generator=g) * 5.0
    fourmomenta = torch.cat([e, p3], dim=-1)

    ptr = torch.tensor([0] + COUNTS, dtype=torch.long).cumsum(0)

    axes = torch.randn(N_JETS, K, 3, generator=g, dtype=torch.float32)
    axes = axes / axes.norm(dim=-1, keepdim=True)

    masses = torch.rand(N_JETS, 4, generator=g, dtype=torch.float32) * 80.0
    phases = torch.rand(N_JETS, 4, generator=g, dtype=torch.float32) * 2.0 - 1.0
    # Synthetic and deliberately NOT pT-ordered: only its direction is used,
    # and both branch and unified code consume this same tensor. Do not
    # "fix" it to a real leading constituent — that invalidates the fixtures.
    p_lead = fourmomenta[[0, COUNTS[0], COUNTS[0] + COUNTS[1]]].clone()

    return fourmomenta, ptr, {
        "axes": axes, "masses": masses, "phases": phases, "p_lead": p_lead,
    }


def call_branch(variant, fourmomenta, ptr, aux):
    """Invoke the branch's embedding entry point for this variant."""
    if variant in ("r4", "r5a", "r5b"):
        return emb.embed_tagging_data(fourmomenta, ptr, axes=aux["axes"])
    if variant == "r5c":
        return emb.embed_tagging_data(
            fourmomenta, ptr, axes=aux["axes"], phases=aux["phases"]
        )
    if variant == "r5d":
        return emb.embed_tagging_data(
            fourmomenta, ptr, axes=aux["axes"], masses=aux["masses"]
        )
    if variant == "r7":
        return emb.embed_tagging_data_bivector_patch(fourmomenta, ptr, axes=aux["axes"])
    if variant == "r8":
        return emb.embed_tagging_data_single_channel_axis(
            fourmomenta, ptr, aux["axes"][:, 0, :], p_lead=aux["p_lead"]
        )
    if variant == "r9":
        return emb.embed_tagging_data_r9(
            fourmomenta, ptr,
            jet_axis=aux["axes"][:, 0, :],
            subjet_axes_2=aux["axes"][:, 2:4, :],
            subjet_axis_highest=aux["axes"][:, 1, :],
        )
    raise SystemExit(f"unknown variant {variant}")


def main(variant, out_dir):
    fourmomenta, ptr, aux = make_inputs()
    out = call_branch(variant, fourmomenta, ptr, aux)
    payload = {
        "inputs": {"fourmomenta": fourmomenta, "ptr": ptr, **aux},
        "outputs": {k: v for k, v in out.items()},
    }
    path = f"{out_dir}/{variant}.pt"
    torch.save(payload, path)
    shapes = {k: tuple(v.shape) for k, v in payload["outputs"].items()}
    print(f"wrote {path}  outputs={shapes}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: capture_embedding_fixtures.py <variant> <out_dir>")
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 3: Capture one fixture per branch via throwaway worktrees**

`.worktrees/` is already gitignored. The script is copied in rather than committed to each branch, so no branch history is modified.

```bash
mkdir -p tests/fixtures
FIX=$(pwd)/tests/fixtures
SCRIPT=$(pwd)/scripts/capture_embedding_fixtures.py
VENV=$(pwd)/.venv/bin/python

capture () {   # $1 = branch, $2 = variant
  git worktree add ".worktrees/$2" "$1"
  cp "$SCRIPT" ".worktrees/$2/scripts/capture_embedding_fixtures.py"
  (cd ".worktrees/$2" && "$VENV" scripts/capture_embedding_fixtures.py "$2" "$FIX")
  git worktree remove --force ".worktrees/$2"
}

capture r4-multiaxis-bivector      r4
capture r5a-pseudoscalar-triad     r5a
capture r5b-pseudoscalar-hand      r5b
capture r5c-pseudoscalar-phase     r5c
capture r5d-subjet-mass            r5d
capture r7-patch-injection         r7
capture r8-single-channel-ablation r8
capture r9-subjet-channel-vs-scalar r9
```

Expected: eight `wrote .../tests/fixtures/<variant>.pt` lines. `r4`, `r5a`, `r5b`, `r5c`, `r5d` report `mv=(3*n_spurions + 20, 5, 8)`; `r7` additionally reports `patch_mv`; `r8` reports `mv=(..., 1, 8)`; `r9` reports `mv=(..., 4, 8)`.

- [ ] **Step 4: Verify all eight fixtures exist and differ where they should**

```bash
.venv/bin/python - <<'EOF'
import torch, itertools
v = ["r4","r5a","r5b","r5c","r5d","r7","r8","r9"]
f = {k: torch.load(f"tests/fixtures/{k}.pt", weights_only=False) for k in v}
for k in v:
    print(k, {n: tuple(t.shape) for n, t in f[k]["outputs"].items()})
for a, b in itertools.combinations(["r4","r5a","r5b","r5c","r5d"], 2):
    same = torch.equal(f[a]["outputs"]["mv"], f[b]["outputs"]["mv"])
    print(f"{a} vs {b}: mv identical = {same}")
EOF
```

Expected: eight shape lines, then a `mv identical = ...` line per pair. These are **diagnostics, not assertions** — a `True` is only a problem if it is unexplained. Each r5 variant differs from r4 by one slot, so a `False` confirms the capture hit distinct branch code. A `True` has exactly two innocent explanations: the injector's output was identically zero for these synthetic inputs, or the capture fed the wrong entry point. Rule out the first by printing the differing slot before concluding the second:

```bash
.venv/bin/python - <<'EOF'
import torch
f = lambda k: torch.load(f"tests/fixtures/{k}.pt", weights_only=False)["outputs"]["mv"]
r4 = f("r4")
for k in ["r5a", "r5b", "r5c", "r5d"]:
    d = (f(k) - r4).abs()
    slots = [i for i in range(8) if d[..., i].max() > 0]
    print(k, "differs in slots", slots, "max diff", float(d.max()))
EOF
```

Expected: `r5a`, `r5b`, `r5c` differ in slot 7; `r5d` differs in slot 0. An empty slot list means the capture used the wrong entry point — stop and fix before continuing.

- [ ] **Step 5: Record the AUC baselines**

Read each branch's recorded numbers and write them down. Sources: `outputs/*/experiment.log` on the branch, top-level `*.log` files, and any AUC line in the branch's commit messages.

```bash
for b in r4-multiaxis-bivector r5a-pseudoscalar-triad r5b-pseudoscalar-hand \
         r5c-pseudoscalar-phase r5d-subjet-mass r7-patch-injection \
         r8-single-channel-ablation r9-subjet-channel-vs-scalar; do
  echo "===== $b"
  git log "$b" --oneline -15
  git ls-tree -r --name-only "$b" | grep -iE '\.log$|outputs/' | head -10
done
```

Create `docs/reference/variant-auc-baselines.md` with one row per variant: variant, source branch, config used, recorded test AUC, and whether the number came from a mini or full run. Where a branch has no recorded number, write `none recorded — baseline must be generated in Task 12` rather than guessing.

- [ ] **Step 6: Commit**

```bash
git add scripts/capture_embedding_fixtures.py tests/fixtures docs/reference/variant-auc-baselines.md
git commit -m "test: capture per-variant embedding fixtures and AUC baselines before unification"
```

---

### Task 2: Land the shared base on `unify-embeddings`

Every branch carries this identical code. Landing it once removes the bulk of every later conflict. `main` currently has none of it.

**Files:**
- Modify: `su2eqnn/utils/subjets.py` (union of the r4, r5d, r8 versions)
- Create: `su2eqnn/utils/catree.py` (verbatim from r5c)
- Create: `scripts/precompute_features.py`
- Modify: `experiments/top_tagging/dataset.py`
- Modify: `experiments/top_tagging/experiment.py`
- Test: `tests/test_subjets.py`, `tests/test_precompute_features.py`

**Interfaces:**
- Produces: `su2eqnn.utils.subjets.jet_axis(kinematics) -> (3,)`, `exclusive_kt_subjets(kinematics, n_subjets, R) -> ((n,3), (n,))`, `exclusive_kt_axes(kinematics, n_subjets, R) -> (n,3)`, `jet_mass(kinematics) -> float`, `leading_momentum(kinematics) -> (4,)`.
- Produces: `su2eqnn.utils.catree.ca_hard_branch_phases(kinematics, n_phases=4, R=0.8) -> (n_phases,) float32`.
- Produces: dataset `Data` objects carrying `axes (1,4,3)`, `masses (1,4)`, `phases (1,4)`, `p_lead (1,4)` when the corresponding npz keys exist.
- Produces: `experiments.top_tagging.experiment.collect_aux(batch) -> dict` — the `aux` dict consumed by every layout and injector.

- [ ] **Step 1: Bring over subjets.py, catree.py and the shared tests**

`r5d` refactors `exclusive_kt_axes` into `exclusive_kt_subjets` plus a thin wrapper; `r8` adds `leading_momentum`. The two edits touch disjoint parts of the file, so the union is r5d's file with r8's function inserted. Checking out both would clobber the first, so take r5d's whole file and hand-insert r8's function:

```bash
git checkout r5d-subjet-mass -- su2eqnn/utils/subjets.py
git show r8-single-channel-ablation:su2eqnn/utils/subjets.py \
  | sed -n '/^def leading_momentum/,/^def _eta_phi_pt/p' | sed '$d' \
  > "$TMPDIR/leading.py"
```

Insert the contents of `$TMPDIR/leading.py` into `su2eqnn/utils/subjets.py` immediately after `jet_axis` and before `_eta_phi_pt`. Then bring the tests and catree:

```bash
git checkout r5c-pseudoscalar-phase -- su2eqnn/utils/catree.py tests/test_catree.py
git checkout r8-single-channel-ablation -- tests/test_subjets.py
git checkout r5d-subjet-mass -- tests/test_masses.py
```

`r8`'s `test_subjets.py` is the longest version (+101 vs +76) and covers `leading_momentum`; `r5d`'s `test_masses.py` covers `jet_mass` and `exclusive_kt_subjets`.

- [ ] **Step 2: Run the utility tests**

```bash
.venv/bin/python -m pytest tests/test_subjets.py tests/test_masses.py tests/test_catree.py -q
```

Expected: PASS. A failure here means the r5d/r8 union was merged wrong — the two edits touch disjoint parts of the file, so re-do the insert rather than editing the physics.

- [ ] **Step 3: Write the failing test for the merged precompute script**

Create `tests/test_precompute_features.py`:

```python
"""precompute_features.py must reproduce the four scripts it replaces."""

import numpy as np
import pytest

from scripts.precompute_features import FEATURES, compute_feature

N_JETS = 4
N_PAD = 12


@pytest.fixture
def kinematics():
    rng = np.random.default_rng(20260825)
    k = np.zeros((N_JETS, N_PAD, 4), dtype=np.float32)
    for i in range(N_JETS):
        n_real = 5 + i
        p3 = rng.normal(scale=30.0, size=(n_real, 3))
        e = np.linalg.norm(p3, axis=-1) + rng.uniform(1.0, 5.0, size=n_real)
        k[i, :n_real, 0] = e
        k[i, :n_real, 1:] = p3
    return k


def test_all_features_registered():
    assert set(FEATURES) == {"axes", "masses", "phases", "leading"}


@pytest.mark.parametrize(
    "name,shape",
    [("axes", (N_JETS, 4, 3)), ("masses", (N_JETS, 4)),
     ("phases", (N_JETS, 4)), ("leading", (N_JETS, 4))],
)
def test_feature_shape_and_dtype(kinematics, name, shape):
    out = compute_feature(name, kinematics)
    assert out.shape == shape
    assert out.dtype == np.float32


def test_axes_are_unit_vectors(kinematics):
    axes = compute_feature("axes", kinematics)
    norms = np.linalg.norm(axes, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_masses_non_negative(kinematics):
    assert (compute_feature("masses", kinematics) >= 0.0).all()


def test_phases_in_unit_interval(kinematics):
    ph = compute_feature("phases", kinematics)
    assert (np.abs(ph) <= 1.0 + 1e-6).all()
```

- [ ] **Step 4: Run it to confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_precompute_features.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.precompute_features'`.

- [ ] **Step 5: Write the merged precompute script**

Create `scripts/precompute_features.py`. The four `compute_*` bodies are copied verbatim from `precompute_axes.py`, `precompute_masses.py`, `precompute_phases.py`, `precompute_leading.py` so the byte-identity check in Step 7 passes.

```python
"""Precompute per-jet auxiliary features for a top-tagging npz dataset.

Replaces precompute_axes.py, precompute_masses.py, precompute_phases.py and
precompute_leading.py. Each feature is an independent writer; keys already in
the source npz are carried through unchanged, so re-running with a subset is
safe.

Usage:
    .venv/bin/python scripts/precompute_features.py \
        --features axes,masses data/toptagging_mini.npz data/toptagging_mini_aux.npz
"""

import argparse

import numpy as np

from su2eqnn.utils.catree import ca_hard_branch_phases
from su2eqnn.utils.subjets import (
    exclusive_kt_subjets, jet_axis, jet_mass, leading_momentum,
)

N_SUBJETS = 3
R = 0.8


def _compute_axes(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 1 + N_SUBJETS, 3) float32."""
    n = kinematics.shape[0]
    out = np.zeros((n, 1 + N_SUBJETS, 3), dtype=np.float32)
    for i in range(n):
        out[i, 0] = jet_axis(kinematics[i])
        out[i, 1:] = exclusive_kt_subjets(kinematics[i], n_subjets=N_SUBJETS, R=R)[0]
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def _compute_masses(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32, [m_jet, m_s1, m_s2, m_s3] in GeV."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i, 0] = jet_mass(kinematics[i])
        out[i, 1:] = exclusive_kt_subjets(kinematics[i], n_subjets=N_SUBJETS, R=R)[1]
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def _compute_phases(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32, sin(dpsi) on the C-A hard branch."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i] = ca_hard_branch_phases(kinematics[i])
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


def _compute_leading(kinematics: np.ndarray) -> np.ndarray:
    """(N, 200, 4) -> (N, 4) float32, highest-pT constituent 4-momentum."""
    n = kinematics.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        out[i] = leading_momentum(kinematics[i])
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    return out


# feature name -> (compute fn, npz key prefix)
FEATURES = {
    "axes":    (_compute_axes,    "axes"),
    "masses":  (_compute_masses,  "masses"),
    "phases":  (_compute_phases,  "phases"),
    "leading": (_compute_leading, "p_lead"),
}


def compute_feature(name: str, kinematics: np.ndarray) -> np.ndarray:
    """Dispatch to one feature's compute function."""
    if name not in FEATURES:
        raise KeyError(f"unknown feature {name!r}; known: {sorted(FEATURES)}")
    return FEATURES[name][0](kinematics)


def main(src: str, dst: str, features) -> None:
    raw = dict(np.load(src))
    modes = sorted(k.split("_", 1)[1] for k in raw if k.startswith("kinematics_"))
    for mode in modes:
        for name in features:
            print(f"mode={mode} feature={name}")
            key = f"{FEATURES[name][1]}_{mode}"
            raw[key] = compute_feature(name, raw[f"kinematics_{mode}"])
    np.savez(dst, **raw)
    print("wrote", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument(
        "--features", default="axes",
        help="comma-separated subset of " + ",".join(sorted(FEATURES)),
    )
    args = ap.parse_args()
    main(args.src, args.dst, [f.strip() for f in args.features.split(",") if f.strip()])
```

Add `scripts/__init__.py` (empty) so the test can import it.

- [ ] **Step 6: Run the test**

```bash
.venv/bin/python -m pytest tests/test_precompute_features.py -q
```

Expected: PASS.

- [ ] **Step 7: Verify byte-identity against the scripts being replaced**

This is the check the spec calls out: the equivalence fixtures pin the embedding, but nothing pins the aux producer unless this runs.

```bash
.venv/bin/python - <<'EOF'
import numpy as np, subprocess, sys, tempfile, os
sys.path.insert(0, ".")
from scripts.precompute_features import compute_feature

raw = np.load("data/toptagging_mini.npz")
kin = raw["kinematics_test"][:200]

# reference implementations, taken from the branches being replaced
import importlib.util
def load(branch, path, modname):
    src = subprocess.check_output(["git", "show", f"{branch}:{path}"]).decode()
    p = os.path.join(tempfile.mkdtemp(), modname + ".py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location(modname, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

checks = [
    ("axes",    load("r4-multiaxis-bivector",       "scripts/precompute_axes.py",    "pa").compute_axes),
    ("masses",  load("r5d-subjet-mass",             "scripts/precompute_masses.py",  "pm").compute_masses),
    ("phases",  load("r5c-pseudoscalar-phase",      "scripts/precompute_phases.py",  "pp").compute_phases),
    ("leading", load("r8-single-channel-ablation",  "scripts/precompute_leading.py", "pl").compute_leading),
]
for name, ref in checks:
    a, b = compute_feature(name, kin), ref(kin)
    print(name, "identical:", np.array_equal(a, b), "| max abs diff:", np.abs(a - b).max())
EOF
```

Expected: four lines, all `identical: True`, `max abs diff: 0.0`. Any `False` means a compute body was paraphrased instead of copied — go back and copy it verbatim.

- [ ] **Step 8: Make dataset.py load any aux key generically**

Replace the hardcoded `axes_{mode}` lookup in `experiments/top_tagging/dataset.py`. The `AUX_KEYS` mapping is npz key prefix -> `Data` attribute name.

```python
# module level, after EPS
AUX_KEYS = {"axes": "axes", "masses": "masses", "phases": "phases", "p_lead": "p_lead"}
```

and inside `load_data`, replacing the `axes_key` block and the `if axes is not None` line:

```python
        aux = {
            attr: torch.tensor(raw[f"{key}_{mode}"], dtype=dtype)
            for key, attr in AUX_KEYS.items()
            if f"{key}_{mode}" in raw.files
        }

        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded particles: keep rows where any |component| > EPS
            mask = (kinematics[i].abs() > EPS).any(dim=-1)
            fourmomenta = kinematics[i][mask]   # (n_real, 4)
            label = labels[i]                    # scalar bool
            data = Data(x=fourmomenta, label=label)
            for attr, tensor in aux.items():
                setattr(data, attr, tensor[i].unsqueeze(0))
            self.data_list.append(data)
```

Every aux tensor gets a leading batch dimension of 1, so torch-geometric's collation stacks them to `(B, ...)` exactly as the branches' `axes` handling did.

- [ ] **Step 9: Write the dataset test**

Append to `tests/test_top_tagging.py`:

```python
def test_dataset_loads_all_present_aux_keys(tmp_path):
    """Any known aux key in the npz reaches the Data object; absent ones don't."""
    import numpy as np
    from experiments.top_tagging.dataset import TopTaggingDataset

    n, n_pad = 3, 6
    rng = np.random.default_rng(7)
    kin = np.zeros((n, n_pad, 4), dtype=np.float32)
    kin[:, :4, :] = rng.normal(size=(n, 4, 4)).astype(np.float32) + 10.0
    path = tmp_path / "toy.npz"
    np.savez(
        path,
        kinematics_test=kin,
        labels_test=np.array([0, 1, 0], dtype=bool),
        axes_test=rng.normal(size=(n, 4, 3)).astype(np.float32),
        masses_test=rng.random((n, 4)).astype(np.float32),
    )

    ds = TopTaggingDataset(rescale_data=True)
    ds.load_data(str(path), "test")

    assert ds[0].axes.shape == (1, 4, 3)
    assert ds[0].masses.shape == (1, 4)
    assert not hasattr(ds[0], "phases")
    assert not hasattr(ds[0], "p_lead")
```

- [ ] **Step 10: Run it**

```bash
.venv/bin/python -m pytest tests/test_top_tagging.py::test_dataset_loads_all_present_aux_keys -q
```

Expected: PASS.

- [ ] **Step 11: Bring experiment.py up to the shared baseline**

Take `r9`'s `experiment.py` as the base — it is the only branch with `set_seed`, which the AUC reproduction in Task 12 depends on.

```bash
git checkout r9-subjet-channel-vs-scalar -- experiments/top_tagging/experiment.py
```

Then edit it: delete its `get_embedding` function and the `embed_tagging_data_r9` import (Task 3 replaces both), **and replace the two `get_embedding(cfg, batch)` call sites that function left behind** — one in `evaluate`, one in the training loop — with the explicit call below. Skipping this leaves `NameError: get_embedding` and Step 12 fails:

```python
            embedding = embed_tagging_data(
                batch.x, batch.ptr,
                two_beams=cfg.data.two_beams,
                rescale=cfg.data.rescale_data,
                axes=getattr(batch, "axes", None),
            )
```

Then add the aux collector next to `set_seed`:

```python
AUX_ATTRS = ("axes", "masses", "phases", "p_lead")


def collect_aux(batch) -> dict:
    """Gather the per-jet auxiliary tensors present on a collated batch."""
    return {
        name: getattr(batch, name)
        for name in AUX_ATTRS
        if getattr(batch, name, None) is not None
    }
```

Leave both call sites (`evaluate` and the training loop) calling `embed_tagging_data(...)` directly for now; Task 3 switches them to `build_embedding`. Keep the Lion branch, the cosine-warm-restarts scheduler, `set_seed`, and `in_mv_channels=getattr(cfg.model, "in_mv_channels", 1)` exactly as r9 has them.

- [ ] **Step 12: Run the full suite**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: PASS, with the count at or above the 66 tests the r4 branch reports.

- [ ] **Step 13: Commit**

```bash
git add su2eqnn/utils/subjets.py su2eqnn/utils/catree.py scripts/precompute_features.py \
        scripts/__init__.py experiments/top_tagging/dataset.py \
        experiments/top_tagging/experiment.py tests/
git commit -m "refactor: land shared subjet/aux infrastructure from experiment branches"
```

---

### Task 3: Embedding package — core, `multi_channel` layout, registry

This replaces `experiments/top_tagging/embedding.py` with a package. At the end of this task, R4 runs through the registry and matches its fixture exactly.

**Files:**
- Delete: `experiments/top_tagging/embedding.py`
- Create: `experiments/top_tagging/embedding/__init__.py`, `core.py`, `layouts.py`, `injectors.py`, `registry.py`
- Modify: `experiments/top_tagging/experiment.py`
- Test: `tests/test_embedding_equivalence.py`, `tests/test_registry.py`

**Interfaces:**
- Produces: `build_embedding(cfg, fourmomenta, ptr, aux) -> dict` with keys `mv`, `batch`, `is_beam` (plus `patch_mv` for the `patch` layout).
- Produces: `core.get_batch_from_ptr(ptr)`, `core.wedge_into(mv, chan_slice, p3, ax)`, `core.attach_spurions(mv, ptr, two_beams)`, `core.apply_axis_channel_dropout(embedding, p)`, `core.UNITS`.
- Produces: layout signature `(fourmomenta, ptr, aux, two_beams, rescale) -> dict`; injector signature `(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None`.
- Produces: `registry.LAYOUTS`, `registry.INJECTORS`, `registry.resolve(name, inject) -> (layout_fn, [Injector])`.

- [ ] **Step 1: Write the failing equivalence test for R4**

Create `tests/test_embedding_equivalence.py`:

```python
"""Unified embedding must reproduce each branch's output bit-for-bit.

Fixtures were captured from the branches in Task 1 and are the acceptance
gate for the refactor. A mismatch here means behaviour changed.
"""

from types import SimpleNamespace

import pytest
import torch

from experiments.top_tagging.embedding import build_embedding

FIXTURE_DIR = "tests/fixtures"

# variant -> (layout name, injector list)
VARIANTS = {
    "r4": ("multi_channel", []),
}


def _cfg(layout, inject):
    return SimpleNamespace(
        data=SimpleNamespace(
            embedding=layout, inject=inject, two_beams=True, rescale_data=True,
        )
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_matches_branch_fixture(variant):
    layout, inject = VARIANTS[variant]
    fx = torch.load(f"{FIXTURE_DIR}/{variant}.pt", weights_only=False)
    inputs, expected = fx["inputs"], fx["outputs"]

    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}
    got = build_embedding(
        _cfg(layout, inject), inputs["fourmomenta"], inputs["ptr"], aux
    )

    assert set(got) >= set(expected)
    for key, want in expected.items():
        assert torch.equal(got[key], want), f"{variant}: {key} differs"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ImportError: cannot import name 'build_embedding'`.

- [ ] **Step 3: Create `core.py`**

The bodies are lifted verbatim from the current `embedding.py`.

```python
"""Shared mechanics for every embedding layout: wedges, spurions, dropout.

Multivector slot layout: [scalar, e1, e2, e3, e12, e13, e23, e123].
"""

import torch
from torch import Tensor

from su2eqnn.interface.spurions import embed_spurions

UNITS = 20.0  # GeV — all momenta divided by this before embedding


def get_batch_from_ptr(ptr: Tensor) -> Tensor:
    """ptr (batchsize+1,) -> per-token jet index (n_total,)."""
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1]
    )


def wedge_into(mv: Tensor, chan: slice, p3: Tensor, ax: Tensor) -> None:
    """Write the pure grade-2 wedge p3 ∧ ax into slots 4/5/6 of `chan`.

    Parameters
    ----------
    mv   : (n_particles, C, 8) — written in place
    chan : channel slice receiving the wedge
    p3   : (n_particles, 1, 3) 3-momenta, broadcast over axes
    ax   : (n_particles, K, 3) unit axes, K matching the width of `chan`
    """
    mv[:, chan, 4] = p3[..., 0] * ax[..., 1] - p3[..., 1] * ax[..., 0]  # e12
    mv[:, chan, 5] = p3[..., 0] * ax[..., 2] - p3[..., 2] * ax[..., 0]  # e13
    mv[:, chan, 6] = p3[..., 1] * ax[..., 2] - p3[..., 2] * ax[..., 1]  # e23


def attach_spurions(mv: Tensor, ptr: Tensor, two_beams: bool) -> dict:
    """Interleave beam spurion tokens into `mv` and rebuild the batch index.

    Spurions carry the beam multivector in channel 0 and zeros elsewhere.

    Returns
    -------
    dict with "mv" (n_total, C, 8), "batch" (n_total,), "is_beam" (n_total,)
    """
    batchsize = len(ptr) - 1
    n_channels = mv.shape[1]

    spurions = embed_spurions(
        two_beams=two_beams, device=mv.device, dtype=mv.dtype,
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

    return {"mv": mv_out, "batch": get_batch_from_ptr(ptr_out), "is_beam": is_beam}


def apply_axis_channel_dropout(embedding: dict, p: float) -> dict:
    """Zero each axis channel (1..C-1) independently per jet with prob p.

    Training-time only (caller's responsibility). Channel 0 is never touched.
    Beam tokens already carry zeros in channels >= 1, so masking every token
    by its jet index is safe. No 1/(1-p) rescaling — ModDrop convention: a
    modality is present or absent, and bivector magnitudes are physical.
    Equivariance is preserved: zeroing a channel commutes with the rotor
    action (w · 0 · w^{-1} = 0).
    """
    mv = embedding["mv"]
    n_axis_channels = mv.shape[1] - 1
    if p <= 0.0 or n_axis_channels == 0:
        return embedding
    batch = embedding["batch"]
    n_jets = int(batch.max().item()) + 1
    keep = (
        torch.rand(n_jets, n_axis_channels, device=mv.device) >= p
    ).to(mv.dtype)                                     # (B, C-1)
    mv[:, 1:, :] = mv[:, 1:, :] * keep[batch][:, :, None]
    return embedding
```

Note: `attach_spurions` zero-pads spurions across all channels, which for `C == 1` is identical to the branches' `spurions.repeat(batchsize, 1, 1)` — the r8 fixture verifies this in Task 8.

- [ ] **Step 4: Create `layouts.py` with `multi_channel` only**

```python
"""Channel layouts. Each owns channel count and any extra output streams.

Signature: (fourmomenta, ptr, aux, two_beams, rescale) -> (mv, extras)

`mv` is pre-spurion (n_particles, C, 8); build_embedding attaches spurions
after injectors have run.
"""

import torch

from experiments.top_tagging.embedding.core import (
    UNITS, attach_spurions, get_batch_from_ptr, wedge_into,
)


def multi_channel(fourmomenta, ptr, aux, two_beams=True, rescale=True) -> dict:
    """R4/R5 layout: channel 0 kinematics, channels 1..K the wedge p⃗ ∧ n̂_k.

    Requires aux["axes"] of shape (batchsize, K, 3). Without it the layout
    degenerates to a single kinematic channel (the R1 baseline).
    """
    batchsize = len(ptr) - 1
    axes = aux.get("axes")
    n_channels = 1 if axes is None else 1 + axes.shape[1]

    if rescale:
        fourmomenta = fourmomenta / UNITS

    mv = torch.zeros(
        fourmomenta.shape[0], n_channels, 8,
        dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    mv[:, 0, 0] = fourmomenta[:, 0]  # E  → grade-0
    mv[:, 0, 1] = fourmomenta[:, 1]  # px → e1
    mv[:, 0, 2] = fourmomenta[:, 2]  # py → e2
    mv[:, 0, 3] = fourmomenta[:, 3]  # pz → e3

    if axes is not None:
        if axes.shape[0] != batchsize:
            raise ValueError(
                f"axes batch size {axes.shape[0]} != number of jets {batchsize}"
            )
        particle_batch = get_batch_from_ptr(ptr)
        ax = axes.to(fourmomenta.dtype)[particle_batch]     # (n_particles, K, 3)
        p = fourmomenta[:, 1:4].unsqueeze(1)                # (n_particles, 1, 3)
        wedge_into(mv, slice(1, None), p, ax)

    return mv, {}
```

The contract: a layout returns the **pre-spurion** `mv` plus a dict of extra pre-spurion tensors, and `build_embedding` calls `attach_spurions` only after every injector has run. Injectors therefore index particle rows directly, with no spurion offset to reason about. `multi_channel` has no extras, so it returns `{}`; only the `patch` layout in Task 10 uses the extras slot. Fix the module docstring accordingly:

```python
Signature: (fourmomenta, ptr, aux, two_beams, rescale) -> (mv, extras)
```

- [ ] **Step 5: Create `injectors.py` (empty registry for now)**

```python
"""Slot injectors. Each writes one slot of an existing mv, in place.

Signature: (mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None

Injectors never allocate, reshape, or change the channel count. Physics
bodies are copied verbatim from the branch that introduced them.
"""

import torch

from experiments.top_tagging.embedding.core import UNITS  # noqa: F401  (used by later injectors)
```

- [ ] **Step 6: Create `registry.py`**

```python
"""Config name -> (layout, injectors), with build-time validation.

Collision detection keys on (channel range, slot), not on slot alone: r5a's
triad injector legitimately writes slot 7 on channel 1 and on channels 2..K,
and must not be reported as colliding with itself.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from experiments.top_tagging.embedding import injectors, layouts

# Channel ranges an injector may claim, and which pairs overlap.
CH0 = "ch0"
AXIS_ALL = "axis_all"        # channels 1..K
AXIS_FIRST4 = "axis_first4"  # channels 1..4

_OVERLAPS = {
    (CH0, CH0),
    (AXIS_ALL, AXIS_ALL), (AXIS_ALL, AXIS_FIRST4),
    (AXIS_FIRST4, AXIS_ALL), (AXIS_FIRST4, AXIS_FIRST4),
}


@dataclass(frozen=True)
class Injector:
    name: str
    fn: Callable
    slot: int
    channels: str
    requires_aux: Tuple[str, ...] = ()
    requires_layout: Optional[str] = None
    requires_k: Optional[int] = None


LAYOUTS = {
    "multi_channel": layouts.multi_channel,
}

INJECTORS = {}


def resolve(layout_name: str, inject_names):
    """Look up a layout and its injectors, validating compatibility.

    Raises ValueError on an unknown name, a layout mismatch, or two injectors
    claiming overlapping (channel range, slot) regions.
    """
    if layout_name not in LAYOUTS:
        raise ValueError(
            f"unknown embedding layout {layout_name!r}; known: {sorted(LAYOUTS)}"
        )
    chosen = []
    for name in inject_names:
        if name not in INJECTORS:
            raise ValueError(
                f"unknown injector {name!r}; known: {sorted(INJECTORS)}"
            )
        inj = INJECTORS[name]
        if inj.requires_layout is not None and inj.requires_layout != layout_name:
            raise ValueError(
                f"injector {name!r} requires layout {inj.requires_layout!r}, "
                f"got {layout_name!r}"
            )
        chosen.append(inj)

    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            if a.slot == b.slot and (a.channels, b.channels) in _OVERLAPS:
                raise ValueError(
                    f"injectors {a.name!r} and {b.name!r} both write slot "
                    f"{a.slot} on overlapping channels "
                    f"({a.channels}, {b.channels})"
                )
    return LAYOUTS[layout_name], chosen


def check_aux(injectors_chosen, aux, layout_name):
    """Raise if a chosen injector's aux key or K requirement is unmet."""
    for inj in injectors_chosen:
        for key in inj.requires_aux:
            if aux.get(key) is None:
                raise ValueError(
                    f"injector {inj.name!r} needs aux key {key!r}; regenerate "
                    f"the dataset with: scripts/precompute_features.py "
                    f"--features {key if key != 'p_lead' else 'leading'}"
                )
        if inj.requires_k is not None:
            k = aux["axes"].shape[1] if aux.get("axes") is not None else 0
            if k != inj.requires_k:
                raise ValueError(
                    f"injector {inj.name!r} requires K={inj.requires_k} axis "
                    f"channels, dataset has K={k}"
                )
```

- [ ] **Step 7: Create `__init__.py` with `build_embedding`**

```python
"""Public embedding API: one entry point, variant chosen by config."""

from experiments.top_tagging.embedding.core import (  # noqa: F401
    UNITS, apply_axis_channel_dropout, get_batch_from_ptr,
)
from experiments.top_tagging.embedding.core import attach_spurions
from experiments.top_tagging.embedding.registry import check_aux, resolve

__all__ = [
    "build_embedding", "apply_axis_channel_dropout", "get_batch_from_ptr", "UNITS",
]


def build_embedding(cfg, fourmomenta, ptr, aux) -> dict:
    """Embed 4-momenta as Cl(3,0) multivectors per the configured variant.

    Parameters
    ----------
    cfg  : config exposing cfg.data.embedding (layout name), cfg.data.inject
           (list of injector names), cfg.data.two_beams, cfg.data.rescale_data
    fourmomenta : (n_particles, 4) sparse, zero-padded rows already removed
    ptr         : (batchsize+1,) cumulative particle counts per jet
    aux         : dict of per-jet tensors — "axes", "masses", "phases", "p_lead"

    Returns
    -------
    dict with "mv" (n_total, C, 8), "batch" (n_total,), "is_beam" (n_total,),
    plus "patch_mv" for the patch layout.
    """
    layout_name = getattr(cfg.data, "embedding", "multi_channel")
    inject_names = list(getattr(cfg.data, "inject", []) or [])
    two_beams = cfg.data.two_beams
    rescale = cfg.data.rescale_data

    layout_fn, chosen = resolve(layout_name, inject_names)
    check_aux(chosen, aux, layout_name)

    mv, extras = layout_fn(fourmomenta, ptr, aux, two_beams, rescale)

    if chosen:
        particle_batch = get_batch_from_ptr(ptr)
        for inj in chosen:
            inj.fn(mv, fourmomenta, ptr, particle_batch, aux, rescale)

    out = attach_spurions(mv, ptr, two_beams)
    out.update(extras)
    return out
```

Note the injector contract detail: `fourmomenta` passed to injectors is the **unrescaled** tensor, and each injector applies `/ UNITS` itself when it needs to — matching r5d, which rescales masses, and r5b/r5a, which do not rescale their pseudoscalar. Layouts rescale their own copy internally.

- [ ] **Step 8: Delete the old module and point experiment.py at the package**

```bash
git rm experiments/top_tagging/embedding.py
```

In `experiments/top_tagging/experiment.py`, change the import to:

```python
from experiments.top_tagging.embedding import build_embedding, apply_axis_channel_dropout
```

and replace both `embed_tagging_data(...)` call sites (in `evaluate` and in the training loop) with:

```python
            embedding = build_embedding(cfg, batch.x, batch.ptr, collect_aux(batch))
```

- [ ] **Step 9: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS — `r4` matches its fixture exactly.

- [ ] **Step 10: Write the registry validation test**

Create `tests/test_registry.py`:

```python
"""Registry validation: unknown names, layout mismatch, slot collisions."""

import pytest

from experiments.top_tagging.embedding import registry
from experiments.top_tagging.embedding.registry import Injector, resolve


def _noop(mv, fourmomenta, ptr, particle_batch, aux, rescale):
    return None


@pytest.fixture
def fake_injectors(monkeypatch):
    table = {
        "a_all": Injector("a_all", _noop, slot=7, channels=registry.AXIS_ALL),
        "b_first4": Injector("b_first4", _noop, slot=7, channels=registry.AXIS_FIRST4),
        "c_ch0": Injector("c_ch0", _noop, slot=7, channels=registry.CH0),
        "d_slot0": Injector("d_slot0", _noop, slot=0, channels=registry.AXIS_ALL),
    }
    monkeypatch.setattr(registry, "INJECTORS", table)
    return table


def test_unknown_layout_raises():
    with pytest.raises(ValueError, match="unknown embedding layout"):
        resolve("nope", [])


def test_unknown_injector_raises(fake_injectors):
    with pytest.raises(ValueError, match="unknown injector"):
        resolve("multi_channel", ["nope"])


def test_overlapping_channel_ranges_same_slot_collide(fake_injectors):
    with pytest.raises(ValueError, match="both write slot 7"):
        resolve("multi_channel", ["a_all", "b_first4"])


def test_disjoint_channel_ranges_same_slot_are_fine(fake_injectors):
    _, chosen = resolve("multi_channel", ["a_all", "c_ch0"])
    assert [i.name for i in chosen] == ["a_all", "c_ch0"]


def test_different_slots_never_collide(fake_injectors):
    _, chosen = resolve("multi_channel", ["a_all", "d_slot0"])
    assert len(chosen) == 2


def test_layout_mismatch_raises(fake_injectors, monkeypatch):
    monkeypatch.setitem(
        registry.INJECTORS, "pinned",
        Injector("pinned", _noop, slot=7, channels=registry.CH0,
                 requires_layout="single_channel"),
    )
    with pytest.raises(ValueError, match="requires layout 'single_channel'"):
        resolve("multi_channel", ["pinned"])
```

- [ ] **Step 11: Run it**

```bash
.venv/bin/python -m pytest tests/test_registry.py -q
```

Expected: PASS (6 tests).

- [ ] **Step 12: Run the full suite and commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add -A experiments/top_tagging tests/
git commit -m "refactor: split embedding into core/layouts/injectors with a validating registry"
```

---

### Task 4: `subjet_mass` injector (replaces r5d)

**Files:**
- Modify: `experiments/top_tagging/embedding/injectors.py`, `registry.py`
- Modify: `tests/test_embedding_equivalence.py`
- Create: `config/top_tagging_subjet_mass.yaml`

**Interfaces:**
- Consumes: `Injector` from Task 3's `registry.py`; `core.UNITS`.
- Produces: injector name `subjet_mass` — slot 0, channels `axis_first4`, requires aux `masses`, requires layout `multi_channel`, requires K=4.

- [ ] **Step 1: Add the variant to the equivalence test**

In `tests/test_embedding_equivalence.py`, extend `VARIANTS`:

```python
VARIANTS = {
    "r4": ("multi_channel", []),
    "r5d": ("multi_channel", ["subjet_mass"]),
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ValueError: unknown injector 'subjet_mass'`.

- [ ] **Step 3: Write the injector**

Append to `injectors.py` — body copied verbatim from `r5d-subjet-mass`:

```python
def subjet_mass(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Broadcast [m_jet, m_s1..s3] into slot 0 of channels 1-4 (r5d).

    Masses arrive in GeV and are divided by UNITS when rescale is set, the
    same convention as the momenta in channel 0.
    """
    masses = aux["masses"].to(mv.dtype)
    if rescale:
        masses = masses / UNITS
    mv[:, 1:5, 0] = masses[particle_batch]
```

- [ ] **Step 4: Register it**

In `registry.py`, populate `INJECTORS`:

```python
INJECTORS = {
    "subjet_mass": Injector(
        name="subjet_mass",
        fn=injectors.subjet_mass,
        slot=0,
        channels=AXIS_FIRST4,
        requires_aux=("masses",),
        requires_layout="multi_channel",
        requires_k=4,
    ),
}
```

- [ ] **Step 5: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS for both `r4` and `r5d`.

- [ ] **Step 6: Add a missing-aux test**

Append to `tests/test_registry.py`:

```python
def test_missing_aux_key_names_the_precompute_flag():
    from types import SimpleNamespace
    import torch
    from experiments.top_tagging.embedding import build_embedding

    cfg = SimpleNamespace(data=SimpleNamespace(
        embedding="multi_channel", inject=["subjet_mass"],
        two_beams=True, rescale_data=True,
    ))
    fourmomenta = torch.randn(5, 4)
    ptr = torch.tensor([0, 5])
    axes = torch.eye(3).repeat(2, 1)[:4].unsqueeze(0)   # (1, 4, 3)

    with pytest.raises(ValueError, match="--features masses"):
        build_embedding(cfg, fourmomenta, ptr, {"axes": axes})
```

- [ ] **Step 7: Run it**

```bash
.venv/bin/python -m pytest tests/test_registry.py -q
```

Expected: PASS.

- [ ] **Step 8: Add the config**

Create `config/top_tagging_subjet_mass.yaml` by taking r5d's config and adding the two registry lines:

```bash
git show r5d-subjet-mass:config/top_tagging_scalar_masses.yaml > config/top_tagging_subjet_mass.yaml
```

Then edit the `data:` block so it reads:

```yaml
data:
  path: data/toptagging_mini_aux.npz
  rescale_data: true
  two_beams: true
  embedding: multi_channel
  inject: [subjet_mass]
```

leaving the `model:`, `training:` and `evaluation:` blocks exactly as r5d had them — the AUC comparison in Task 12 depends on identical hyperparameters.

- [ ] **Step 9: Commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add experiments/top_tagging/embedding tests/ config/top_tagging_subjet_mass.yaml
git commit -m "feat: port r5d subjet-mass injection to the embedding registry"
```

---

### Task 5: `triad_handedness` injector (replaces r5a)

**Files:**
- Modify: `experiments/top_tagging/embedding/injectors.py`, `registry.py`
- Modify: `tests/test_embedding_equivalence.py`
- Create: `config/top_tagging_pseudoscalar_triad.yaml`

**Interfaces:**
- Produces: injector name `triad_handedness` — slot 7, channels `axis_all`, requires aux `axes`, requires layout `multi_channel`.

This is the injector that motivated `(channel range, slot)` collision keying: it writes slot 7 on channel 1 *and* on channels 2..K, as one injector.

- [ ] **Step 1: Add the variant to the equivalence test**

```python
    "r5a": ("multi_channel", ["triad_handedness"]),
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ValueError: unknown injector 'triad_handedness'`.

- [ ] **Step 3: Write the injector**

Body copied verbatim from `r5a-pseudoscalar-triad`. Note it uses the **unrescaled** `fourmomenta` for the signed volumes and does not rescale `chi_triad` (pure unit-vector product, already O(1)).

```python
def triad_handedness(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Signed subjet triads into slot 7 (r5a).

    Channel 1 carries the jet-level triad handedness broadcast to every
    particle; channels 2..K carry per-particle signed volumes built from the
    jet axis and each subjet axis. chi_triad is a product of unit vectors
    (O(1)) and is deliberately not rescaled.
    """
    axes_f = aux["axes"].to(mv.dtype)                        # (B, K, 3)
    n_jet = axes_f[:, 0]                                     # (B, 3)
    subs = axes_f[:, 1:]                                     # (B, K-1, 3)

    chi_triad = (torch.cross(subs[:, 0], subs[:, 1], dim=-1) * subs[:, 2]).sum(-1)
    mv[:, 1, 7] = chi_triad[particle_batch]

    cross_jk = torch.cross(n_jet.unsqueeze(1).expand_as(subs), subs, dim=-1)
    mv[:, 2:, 7] = torch.einsum(
        "nd,nkd->nk", fourmomenta[:, 1:4], cross_jk[particle_batch]
    )
```

- [ ] **Step 4: Register it**

```python
    "triad_handedness": Injector(
        name="triad_handedness",
        fn=injectors.triad_handedness,
        slot=7,
        channels=AXIS_ALL,
        requires_aux=("axes",),
        requires_layout="multi_channel",
    ),
```

- [ ] **Step 5: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS for `r4`, `r5a`, `r5d`.

If `r5a` fails on channel 2..K only, the cause is `fourmomenta` rescaling: the branch computes the signed volumes from the raw momenta, so the injector must receive the unrescaled tensor. Confirm `build_embedding` passes the original `fourmomenta`, not the layout's rescaled copy.

- [ ] **Step 6: Add the config**

```bash
git show r5a-pseudoscalar-triad:config/top_tagging_pseudoscalar_triad.yaml > config/top_tagging_pseudoscalar_triad.yaml
```

Edit its `data:` block to add `embedding: multi_channel` and `inject: [triad_handedness]`, and point `path:` at the aux npz. Leave all other blocks untouched.

- [ ] **Step 7: Commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add experiments/top_tagging/embedding tests/ config/top_tagging_pseudoscalar_triad.yaml
git commit -m "feat: port r5a triad-handedness injection to the embedding registry"
```

---

### Task 6: `local_handedness` injector (replaces r5b)

**Files:**
- Modify: `experiments/top_tagging/embedding/injectors.py`, `registry.py`
- Modify: `tests/test_embedding_equivalence.py`
- Create: `config/top_tagging_pseudoscalar_handedness.yaml`

**Interfaces:**
- Produces: injector name `local_handedness` — slot 7, channels `axis_all`, requires aux `axes`, requires layout `multi_channel`.

- [ ] **Step 1: Add the variant to the equivalence test**

```python
    "r5b": ("multi_channel", ["local_handedness"]),
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ValueError: unknown injector 'local_handedness'`.

- [ ] **Step 3: Write the injector**

Copied verbatim from `r5b-pseudoscalar-hand`, comments included — they record why the beam-frame `eta/phi/DeltaR` weighting was rejected, which is exactly the kind of reasoning that gets lost in a refactor. The `clamp_min` ordering is load-bearing for bit-exactness; do not reorder or vectorize the loop.

```python
def local_handedness(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Efremov-style local handedness into slot 7 of every axis channel (r5b).

    v_i = normalized sum_{j!=i} (|p_j|/max(|p_i-p_j|,1e-3)) p_j within the
    jet; chi_i^(k) = n_k . (p_i x v_i). Neighborhood average (not
    nearest-neighbor) for IRC smoothness. O(n^2) per jet.

    NOTE: deviates from the brief's eta/phi-and-pT ("DeltaR") weighting.
    eta, phi and DeltaR are only invariant under boosts/rotations about the
    beam (z) axis, not under a general Spin(3) rotation -- using them here
    made v (and hence chi) fail the full-rotation equivariance test.
    Replacing pT with |p| and DeltaR with the Euclidean momentum separation
    |p_i-p_j| (the direct SO(3)-invariant analog of DeltaR) keeps the same
    "nearby, high-momentum neighbors dominate" idea, makes v a genuinely
    SO(3)-equivariant function of the momenta (so chi = n . (p x v) is
    invariant as required), and avoids the unbounded gradient that an
    arccos-based opening-angle metric would have for collimated jets.

    NOTE: the 1e-3 floor is applied to |p_i-p_j| in *rescaled* momentum space
    when rescale=True (GeV/UNITS=20), not to an eta-phi DeltaR. A separation
    of 1e-3 there is ~0.02 GeV of 3-momentum difference -- below realistic
    constituent spacing, so it only guards the coincident-momentum edge case.
    """
    ax = aux["axes"].to(mv.dtype)[particle_batch]      # (n_particles, K, 3)
    p3 = fourmomenta[:, 1:4]
    if rescale:
        p3 = p3 / UNITS
    pnorm = p3.norm(dim=-1).clamp_min(1e-12)
    v = torch.zeros_like(p3)
    for j0, j1 in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        if j1 - j0 < 2:
            continue
        pj = p3[j0:j1]
        nj = pnorm[j0:j1]
        dist = torch.cdist(pj, pj)
        w = nj[None, :] / dist.clamp_min(1e-3)
        w = w - torch.diag_embed(torch.diagonal(w))          # zero self-weight
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        v[j0:j1] = w @ pj
    cross_pv = torch.linalg.cross(p3, v)                      # (n, 3)
    mv[:, 1:, 7] = torch.einsum("nkd,nd->nk", ax, cross_pv)
```

The branch computed `p3` from the already-rescaled `fourmomenta` inside `embed_tagging_data`. Since injectors receive the unrescaled tensor, the `if rescale: p3 = p3 / UNITS` line above restores that. If the fixture mismatches, this line is the first place to check.

- [ ] **Step 4: Register it**

```python
    "local_handedness": Injector(
        name="local_handedness",
        fn=injectors.local_handedness,
        slot=7,
        channels=AXIS_ALL,
        requires_aux=("axes",),
        requires_layout="multi_channel",
    ),
```

- [ ] **Step 5: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS for `r4`, `r5a`, `r5b`, `r5d`.

- [ ] **Step 6: Assert the collision rule catches the real case**

`triad_handedness` and `local_handedness` both claim slot 7 on `axis_all`. Append to `tests/test_registry.py`:

```python
def test_two_slot7_injectors_collide():
    from experiments.top_tagging.embedding.registry import resolve
    with pytest.raises(ValueError, match="both write slot 7"):
        resolve("multi_channel", ["triad_handedness", "local_handedness"])
```

- [ ] **Step 7: Run it, add the config, commit**

```bash
.venv/bin/python -m pytest tests/ -q
git show r5b-pseudoscalar-hand:config/top_tagging_pseudoscalar_handedness.yaml > config/top_tagging_pseudoscalar_handedness.yaml
```

Edit its `data:` block to add `embedding: multi_channel` and `inject: [local_handedness]`, then:

```bash
git add experiments/top_tagging/embedding tests/ config/top_tagging_pseudoscalar_handedness.yaml
git commit -m "feat: port r5b local-handedness injection to the embedding registry"
```

---

### Task 7: `ca_phase` injector (replaces r5c)

`su2eqnn/utils/catree.py` already landed in Task 2.

**Files:**
- Modify: `experiments/top_tagging/embedding/injectors.py`, `registry.py`
- Modify: `tests/test_embedding_equivalence.py`
- Create: `config/top_tagging_pseudoscalar_phase.yaml`

**Interfaces:**
- Produces: injector name `ca_phase` — slot 7, channels `axis_first4`, requires aux `phases`, requires layout `multi_channel`, requires K=4.

- [ ] **Step 1: Add the variant to the equivalence test**

```python
    "r5c": ("multi_channel", ["ca_phase"]),
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ValueError: unknown injector 'ca_phase'`.

- [ ] **Step 3: Write the injector**

```python
def ca_phase(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """C-A splitting-plane phases into slot 7 of channels 1-4 (r5c).

    sin(dpsi) between successive splitting planes on the hard branch, broadcast
    per jet. O(1) by construction, so never rescaled.
    """
    phases = aux["phases"].to(mv.dtype)
    mv[:, 1:5, 7] = phases[particle_batch]
```

- [ ] **Step 4: Register it**

```python
    "ca_phase": Injector(
        name="ca_phase",
        fn=injectors.ca_phase,
        slot=7,
        channels=AXIS_FIRST4,
        requires_aux=("phases",),
        requires_layout="multi_channel",
        requires_k=4,
    ),
```

- [ ] **Step 5: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS for `r4`, `r5a`, `r5b`, `r5c`, `r5d`.

- [ ] **Step 6: Test the K-mismatch guard**

Append to `tests/test_registry.py`:

```python
def test_ca_phase_rejects_wrong_k():
    from types import SimpleNamespace
    import torch
    from experiments.top_tagging.embedding import build_embedding

    cfg = SimpleNamespace(data=SimpleNamespace(
        embedding="multi_channel", inject=["ca_phase"],
        two_beams=True, rescale_data=True,
    ))
    aux = {
        "axes": torch.eye(3).unsqueeze(0),            # K = 3, not 4
        "phases": torch.zeros(1, 4),
    }
    with pytest.raises(ValueError, match="requires K=4"):
        build_embedding(cfg, torch.randn(5, 4), torch.tensor([0, 5]), aux)
```

- [ ] **Step 7: Run it, add the config, commit**

```bash
.venv/bin/python -m pytest tests/ -q
git show r5c-pseudoscalar-phase:config/top_tagging_pseudoscalar_phase.yaml > config/top_tagging_pseudoscalar_phase.yaml 2>/dev/null \
  || cp config/top_tagging_pseudoscalar_triad.yaml config/top_tagging_pseudoscalar_phase.yaml
```

Edit its `data:` block to `embedding: multi_channel` / `inject: [ca_phase]`, and set `training.checkpoint_path` to a phase-specific filename so it does not overwrite another variant's checkpoint. Do **not** copy r5c's `outputs/pseudoscalar_phase_mini/` directory or its logs.

```bash
git add experiments/top_tagging/embedding tests/ config/top_tagging_pseudoscalar_phase.yaml
git commit -m "feat: port r5c C-A phase injection to the embedding registry"
```

---

### Task 8: `single_channel` layout + `lead_pseudoscalar` injector (replaces r8)

**Files:**
- Modify: `experiments/top_tagging/embedding/layouts.py`, `injectors.py`, `registry.py`
- Modify: `tests/test_embedding_equivalence.py`
- Create: `config/top_tagging_single_channel_axis.yaml`

**Interfaces:**
- Produces: layout name `single_channel` — one channel: kinematics in slots 0-3, jet-axis wedge in 4-6, slot 7 free.
- Produces: injector name `lead_pseudoscalar` — slot 7, channels `ch0`, requires aux `p_lead`, requires layout `single_channel`.

- [ ] **Step 1: Add the variant to the equivalence test**

```python
    "r8": ("single_channel", ["lead_pseudoscalar"]),
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ValueError: unknown embedding layout 'single_channel'`.

- [ ] **Step 3: Write the layout**

Append to `layouts.py`, body copied verbatim from `r8-single-channel-ablation`:

```python
def single_channel(fourmomenta, ptr, aux, two_beams=True, rescale=True):
    """r8 layout: everything in channel 0.

    Slots 0-3 hold (E, px, py, pz); slots 4-6 hold p⃗ ∧ n̂_jet, the K=1
    restriction of the multi_channel wedge; slot 7 is left for an injector.
    """
    if aux.get("axes") is None:
        raise ValueError("single_channel layout requires aux['axes']")
    jet_axis = aux["axes"][:, 0, :]

    if rescale:
        fourmomenta = fourmomenta / UNITS

    mv = torch.zeros(
        fourmomenta.shape[0], 1, 8,
        dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    mv[:, 0, 0] = fourmomenta[:, 0]
    mv[:, 0, 1:4] = fourmomenta[:, 1:4]

    particle_batch = get_batch_from_ptr(ptr)
    ax = jet_axis.to(fourmomenta.dtype)[particle_batch]   # (n_particles, 3)
    p = fourmomenta[:, 1:4]                                # (n_particles, 3)
    mv[:, 0, 4] = p[:, 0] * ax[:, 1] - p[:, 1] * ax[:, 0]   # e12
    mv[:, 0, 5] = p[:, 0] * ax[:, 2] - p[:, 2] * ax[:, 0]   # e13
    mv[:, 0, 6] = p[:, 1] * ax[:, 2] - p[:, 2] * ax[:, 1]   # e23

    return mv, {}
```

- [ ] **Step 4: Write the injector**

Append to `injectors.py`. The branch computes this from the rescaled momenta inside the layout, so the injector rescales its own copy.

```python
def lead_pseudoscalar(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Pseudoscalar volume I = p⃗ · (n̂_jet × p̂_lead) into slot 7 of channel 0 (r8).

    p_lead is the highest-pT constituent's 4-momentum; only its direction is
    used, so the value is scale-free apart from p⃗ itself.
    """
    ax = aux["axes"].to(mv.dtype)[:, 0, :][particle_batch]   # (n_particles, 3)
    p = fourmomenta[:, 1:4].to(mv.dtype)
    if rescale:
        p = p / UNITS
    pl = aux["p_lead"][:, 1:4].to(mv.dtype)
    pl_norm = pl.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pl_hat = (pl / pl_norm)[particle_batch]                  # (n_particles, 3)
    cross = torch.cross(ax, pl_hat, dim=-1)
    mv[:, 0, 7] = (p * cross).sum(dim=-1)
```

- [ ] **Step 5: Register both**

```python
LAYOUTS = {
    "multi_channel": layouts.multi_channel,
    "single_channel": layouts.single_channel,
}
```

and in `INJECTORS`:

```python
    "lead_pseudoscalar": Injector(
        name="lead_pseudoscalar",
        fn=injectors.lead_pseudoscalar,
        slot=7,
        channels=CH0,
        requires_aux=("axes", "p_lead"),
        requires_layout="single_channel",
    ),
```

- [ ] **Step 6: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS for all six variants so far. `r8` also exercises `attach_spurions` at `C == 1`, confirming the zero-padded spurion path matches the branch's unpadded `spurions.repeat(...)`.

- [ ] **Step 7: Add the ablation-(a) test**

r8's point is the ablation: variant (a) is the same layout with no pseudoscalar. Append to `tests/test_embedding_equivalence.py`:

```python
def test_single_channel_without_injector_leaves_slot7_zero():
    fx = torch.load(f"{FIXTURE_DIR}/r8.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}

    got = build_embedding(
        _cfg("single_channel", []), inputs["fourmomenta"], inputs["ptr"], aux
    )
    assert got["mv"].shape[1] == 1
    assert torch.equal(got["mv"][:, 0, 7], torch.zeros_like(got["mv"][:, 0, 7]))
    # slots 0-6 are unaffected by the injector's presence
    assert torch.equal(got["mv"][..., :7], fx["outputs"]["mv"][..., :7])
```

- [ ] **Step 8: Run it, add the configs, commit**

```bash
.venv/bin/python -m pytest tests/ -q
git show r8-single-channel-ablation:config/top_tagging_single_channel_axis_a_full.yaml > config/top_tagging_single_channel_axis.yaml
```

Edit its `data:` block to `embedding: single_channel` and `inject: []` (variant a). Create a sibling `config/top_tagging_single_channel_axis_b.yaml` from r8's `_b` config with `inject: [lead_pseudoscalar]`.

```bash
git add experiments/top_tagging/embedding tests/ config/top_tagging_single_channel_axis*.yaml
git commit -m "feat: port r8 single-channel layout and lead-pseudoscalar injector"
```

---

### Task 9: `hybrid_r9` layout + `cos_theta` injector (replaces r9)

r9's experimental question — subjet information as an extra *channel* versus as a *scalar* — becomes a config toggle: the layout provides the channels, the injector provides the scalar.

**Files:**
- Modify: `experiments/top_tagging/embedding/layouts.py`, `injectors.py`, `registry.py`
- Modify: `tests/test_embedding_equivalence.py`
- Create: `config/top_tagging_r9_channel.yaml`, `config/top_tagging_r9_scalar.yaml`

**Interfaces:**
- Produces: layout name `hybrid_r9` — 4 channels: ch0 kinematics, ch1-3 wedges against `axes[:,0]`, `axes[:,2]`, `axes[:,3]`.
- Produces: injector name `cos_theta` — slot 4, channels `ch0`, requires aux `axes`, requires layout `hybrid_r9`, requires K=4.

Axis indexing matters and is easy to get wrong: `axes[:,1]` is the **highest-pT subjet**, consumed by the injector, and is deliberately *excluded* from the layout's wedge channels.

- [ ] **Step 1: Add the variant to the equivalence test**

```python
    "r9": ("hybrid_r9", ["cos_theta"]),
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ValueError: unknown embedding layout 'hybrid_r9'`.

- [ ] **Step 3: Write the layout**

```python
def hybrid_r9(fourmomenta, ptr, aux, two_beams=True, rescale=True):
    """r9 layout: 4 channels, subjet-highest axis reserved for the scalar.

    Channel 0 holds kinematics (slot 4 is left for the cos_theta injector);
    channels 1-3 hold p⃗ ∧ n̂ for the jet axis, and the two remaining subjet
    axes axes[:, 2] and axes[:, 3]. axes[:, 1] — the highest-pT subjet — is
    deliberately absent here; it feeds the cos_theta injector instead.
    """
    axes = aux.get("axes")
    if axes is None:
        raise ValueError("hybrid_r9 layout requires aux['axes']")
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

    wedge_axes = torch.cat(
        [axes[:, 0].unsqueeze(1), axes[:, 2:4]], dim=1
    ).to(fourmomenta.dtype)                                 # (batchsize, 3, 3)
    ax = wedge_axes[particle_batch]                         # (n_particles, 3, 3)
    wedge_into(mv, slice(1, None), p.unsqueeze(1), ax)

    return mv, {}
```

- [ ] **Step 4: Write the injector**

```python
def cos_theta(mv, fourmomenta, ptr, particle_batch, aux, rescale) -> None:
    """Invariant cos(theta) against the highest-pT subjet, into ch0 slot 4 (r9).

    cos_theta = p̂ · n̂_sub_highest, where n̂_sub_highest is axes[:, 1]. Scale
    free by construction: the momentum is normalized before the dot product,
    so rescale does not change the value.
    """
    p = fourmomenta[:, 1:4].to(mv.dtype)
    if rescale:
        p = p / UNITS
    p_norm = p.norm(dim=-1).clamp_min(1e-12)
    ax_highest = aux["axes"][:, 1, :].to(mv.dtype)[particle_batch]
    mv[:, 0, 4] = (p * ax_highest).sum(dim=-1) / p_norm
```

- [ ] **Step 5: Register both**

```python
    "hybrid_r9": layouts.hybrid_r9,
```

and:

```python
    "cos_theta": Injector(
        name="cos_theta",
        fn=injectors.cos_theta,
        slot=4,
        channels=CH0,
        requires_aux=("axes",),
        requires_layout="hybrid_r9",
        requires_k=4,
    ),
```

- [ ] **Step 6: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS for all seven variants so far.

If `r9` mismatches only in `mv[:, 0, 4]`, the likely cause is the rescale interaction: the branch normalizes the rescaled momentum, and `p / |p|` is invariant to the `/UNITS` factor, so both orderings agree in exact arithmetic but not necessarily in float32. Keep the `if rescale` line so the arithmetic is performed in the same order as the branch.

- [ ] **Step 7: Add the channel-vs-scalar ablation test**

```python
def test_hybrid_r9_without_cos_theta_leaves_ch0_slot4_zero():
    fx = torch.load(f"{FIXTURE_DIR}/r9.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}

    got = build_embedding(
        _cfg("hybrid_r9", []), inputs["fourmomenta"], inputs["ptr"], aux
    )
    assert torch.equal(got["mv"][:, 0, 4], torch.zeros_like(got["mv"][:, 0, 4]))
    assert torch.equal(got["mv"][:, 1:], fx["outputs"]["mv"][:, 1:])
```

- [ ] **Step 8: Run it, add the configs, commit**

```bash
.venv/bin/python -m pytest tests/ -q
```

Create `config/top_tagging_r9_scalar.yaml` (`embedding: hybrid_r9`, `inject: [cos_theta]`) and `config/top_tagging_r9_channel.yaml` (`embedding: multi_channel`, `inject: []`, `model.in_mv_channels: 5`) — the two arms of r9's comparison — copying hyperparameters from r9's committed config.

```bash
git add experiments/top_tagging/embedding tests/ config/top_tagging_r9_*.yaml
git commit -m "feat: port r9 hybrid layout and cos-theta injector"
```

---

### Task 10: `patch` layout + GATr patch pathway (replaces r7)

The only ported variant that changes the model. The GATr changes are inert when `patch_mv_channels` is None, so `main` behaviour is unaffected.

**Files:**
- Modify: `su2eqnn/nets/gatr.py`
- Modify: `experiments/top_tagging/wrapper.py`
- Modify: `experiments/top_tagging/embedding/layouts.py`, `registry.py`, `core.py`
- Modify: `experiments/top_tagging/experiment.py`
- Modify: `tests/test_embedding_equivalence.py`
- Create: `config/top_tagging_bivector_patch.yaml`
- Test: `tests/test_bivector_patch_e2e.py`

**Interfaces:**
- Produces: layout name `patch` — `mv` is channel-0-only kinematics; the wedges are returned as a separate `patch_mv` tensor `(n_total, K, 8)`, zero on spurion rows.
- Produces: `GATr(..., patch_mv_channels: Optional[int] = None)` and `GATr.forward(..., patch_mv=None)`.
- Produces: `TopTaggingWrapperBivectorPatch(net)`.
- Produces: `core.apply_patch_channel_dropout(embedding, p)`.

- [ ] **Step 1: Bring over the model-side changes**

```bash
git checkout r7-patch-injection -- su2eqnn/nets/gatr.py experiments/top_tagging/wrapper.py tests/test_bivector_patch_e2e.py
```

- [ ] **Step 2: Confirm the pathway is inert by default**

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: PASS, including every pre-existing test — `patch_mv_channels=None` must leave `GATr` numerically unchanged. A failure here means the checkout pulled in an unrelated r7 change; inspect with `git diff --cached su2eqnn/nets/gatr.py`.

- [ ] **Step 3: Add the variant to the equivalence test**

```python
    "r7": ("patch", []),
```

- [ ] **Step 4: Run it to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: FAIL with `ValueError: unknown embedding layout 'patch'`.

- [ ] **Step 5: Write the layout**

The `patch` layout is the one case where a layout produces an extra tensor. `attach_spurions` interleaves `mv`; `patch_mv` needs the same interleaving with zeros on beam rows, so the layout returns it as an extra and `build_embedding`'s `out.update(extras)` carries it through — but the extra must already be spurion-interleaved. Handle this by having the layout call a small helper:

Add to `core.py`:

```python
def scatter_particles(values: Tensor, is_beam: Tensor) -> Tensor:
    """Place per-particle rows into a spurion-interleaved tensor, zeros on beams.

    Parameters
    ----------
    values  : (n_particles, C, 8)
    is_beam : (n_total,) bool from attach_spurions
    """
    out = torch.zeros(
        is_beam.shape[0], values.shape[1], values.shape[2],
        dtype=values.dtype, device=values.device,
    )
    out[~is_beam] = values
    return out


def apply_patch_channel_dropout(embedding: dict, p: float) -> dict:
    """Zero each patch channel independently per jet with prob p.

    Same ModDrop convention as apply_axis_channel_dropout, applied to the
    separate patch_mv stream instead of to input channels.
    """
    patch = embedding.get("patch_mv")
    if patch is None or p <= 0.0:
        return embedding
    batch = embedding["batch"]
    n_jets = int(batch.max().item()) + 1
    keep = (
        torch.rand(n_jets, patch.shape[1], device=patch.device) >= p
    ).to(patch.dtype)
    embedding["patch_mv"] = patch * keep[batch][:, :, None]
    return embedding
```

Then in `layouts.py`:

```python
def patch(fourmomenta, ptr, aux, two_beams=True, rescale=True):
    """r7 layout: plain kinematic mv plus a separate patch_mv wedge stream.

    Unlike multi_channel, the wedges are not baked into input channels; they
    are returned separately for per-block residual re-injection into GATr's
    hidden stream.
    """
    axes = aux.get("axes")
    if axes is None:
        raise ValueError("patch layout requires aux['axes']")

    if rescale:
        fourmomenta = fourmomenta / UNITS

    n = fourmomenta.shape[0]
    mv = torch.zeros(n, 1, 8, dtype=fourmomenta.dtype, device=fourmomenta.device)
    mv[:, 0, 0] = fourmomenta[:, 0]
    mv[:, 0, 1:4] = fourmomenta[:, 1:4]

    particle_batch = get_batch_from_ptr(ptr)
    ax = axes.to(fourmomenta.dtype)[particle_batch]      # (n_particles, K, 3)
    p = fourmomenta[:, 1:4].unsqueeze(1)                 # (n_particles, 1, 3)

    patch_mv = torch.zeros(
        n, axes.shape[1], 8, dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    wedge_into(patch_mv, slice(None), p, ax)

    return mv, {"_patch_particles": patch_mv}
```

The leading underscore marks a pre-spurion extra. In `__init__.py`, after `out = attach_spurions(...)`, translate it:

```python
    for key, value in extras.items():
        if key == "_patch_particles":
            out["patch_mv"] = scatter_particles(value, out["is_beam"])
        else:
            out[key] = value
```

Import `scatter_particles` from `core` alongside `attach_spurions`.

- [ ] **Step 6: Register the layout**

```python
    "patch": layouts.patch,
```

- [ ] **Step 7: Run the equivalence test**

```bash
.venv/bin/python -m pytest tests/test_embedding_equivalence.py -q
```

Expected: PASS for all eight variants. The `assert set(got) >= set(expected)` line covers `patch_mv`, so a missing or mis-scattered patch stream fails here.

- [ ] **Step 8: Wire the wrapper selection into experiment.py**

In `build_model`, restore r7's branch:

```python
    patch_mv_channels = getattr(cfg.model, "patch_mv_channels", None)
    net = GATr(
        in_mv_channels=getattr(cfg.model, "in_mv_channels", 1),
        out_mv_channels=1,
        hidden_mv_channels=cfg.model.hidden_mv_channels,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=cfg.model.num_heads),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=cfg.model.num_blocks,
        patch_mv_channels=patch_mv_channels,
    )
    if patch_mv_channels is not None:
        return TopTaggingWrapperBivectorPatch(net)
    return TopTaggingWrapper(net)
```

and import both wrappers. In the training loop, apply patch dropout alongside axis dropout:

```python
            embedding = build_embedding(cfg, batch.x, batch.ptr, collect_aux(batch))
            if axis_dropout_p > 0.0:
                apply_axis_channel_dropout(embedding, axis_dropout_p)
                apply_patch_channel_dropout(embedding, axis_dropout_p)
```

`apply_axis_channel_dropout` is a no-op when `C == 1` (the patch layout's `mv`), and `apply_patch_channel_dropout` is a no-op when `patch_mv` is absent, so both can be called unconditionally.

- [ ] **Step 9: Run the end-to-end patch test**

```bash
.venv/bin/python -m pytest tests/test_bivector_patch_e2e.py -q
```

Expected: PASS. This test came from r7 and may call the old `embed_tagging_data_bivector_patch` name — update its call to `build_embedding(cfg, ...)` with `embedding: patch`, keeping every assertion unchanged.

- [ ] **Step 10: Add the config and commit**

```bash
.venv/bin/python -m pytest tests/ -q
```

Create `config/top_tagging_bivector_patch.yaml` with `data.embedding: patch`, `data.inject: []`, `model.in_mv_channels: 1`, `model.patch_mv_channels: 4`, hyperparameters copied from r7's config.

```bash
git add su2eqnn/nets/gatr.py experiments/ tests/ config/top_tagging_bivector_patch.yaml
git commit -m "feat: port r7 bivector patch injection (layout + GATr patch pathway)"
```

---

### Task 11: Collapse the duplicated equivariance tests into one parametrized sweep

Eight branches each grew their own `test_equivariance.py`, ranging from +116 to +471 lines, largely re-testing the same property.

**Files:**
- Modify: `tests/test_equivariance.py`

**Interfaces:**
- Consumes: `VARIANTS` from `tests/test_embedding_equivalence.py`; `build_embedding`.

- [ ] **Step 1: Write the parametrized equivariance test**

Append to `tests/test_equivariance.py`:

```python
"""Every registered layout/injector combination must be Spin(3)-equivariant.

A rotation R acting on the input momenta and on the reference axes must act on
the output multivectors by the corresponding rotor: grade-1 slots rotate as
vectors, grade-2 slots as bivectors, and grade-0/grade-3 slots are invariant.
"""

import pytest
import torch

from experiments.top_tagging.embedding import build_embedding
from tests.test_embedding_equivalence import VARIANTS, _cfg

FIXTURE_DIR = "tests/fixtures"


def _random_rotation(seed):
    """Uniform-ish SO(3) element via QR of a Gaussian matrix, det forced to +1."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(torch.float32)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("seed", [1, 2])
def test_scalar_slots_are_rotation_invariant(variant, seed):
    """Grade-0 (slot 0) and grade-3 (slot 7) outputs must not change under R."""
    layout, inject = VARIANTS[variant]
    fx = torch.load(f"{FIXTURE_DIR}/{variant}.pt", weights_only=False)
    inputs = fx["inputs"]
    aux = {k: v for k, v in inputs.items() if k not in ("fourmomenta", "ptr")}
    cfg = _cfg(layout, inject)

    plain = build_embedding(cfg, inputs["fourmomenta"], inputs["ptr"], aux)

    R = _random_rotation(seed)
    rot_fm = inputs["fourmomenta"].clone()
    rot_fm[:, 1:4] = rot_fm[:, 1:4] @ R.T
    rot_aux = dict(aux)
    rot_aux["axes"] = aux["axes"] @ R.T
    if "p_lead" in aux:
        pl = aux["p_lead"].clone()
        pl[:, 1:4] = pl[:, 1:4] @ R.T
        rot_aux["p_lead"] = pl

    rotated = build_embedding(cfg, rot_fm, inputs["ptr"], rot_aux)

    for slot in (0, 7):
        assert torch.allclose(
            plain["mv"][..., slot], rotated["mv"][..., slot], atol=1e-4
        ), f"{variant}: slot {slot} is not rotation invariant"
```

The `axes` used here come from the fixture and are already unit vectors, so `axes @ R.T` stays normalized.

Caveat to record in the test file as a comment: the kT subjet axes and the C-A tree are computed in the beam frame, so *precomputed* aux is only equivariant under azimuthal rotations in production. This test rotates the aux tensors explicitly, which isolates the embedding's own equivariance — the property being asserted — from the aux producer's frame dependence.

- [ ] **Step 2: Run it**

```bash
.venv/bin/python -m pytest tests/test_equivariance.py -q
```

Expected: PASS, 16 parametrized cases plus the pre-existing tests.

A failure for `r5b` or `r5a` is meaningful, not a test bug: those injectors' invariance claims are the physics content of their branches. Investigate before weakening any tolerance.

- [ ] **Step 3: Remove now-redundant duplicates**

Compare the pre-existing tests in `tests/test_equivariance.py` against the sweep. Delete only tests whose assertion is strictly covered by the parametrized version. Keep every test that asserts something the sweep does not — bivector transformation law, spurion invariance, per-injector physics identities.

- [ ] **Step 4: Run the full suite and commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add tests/test_equivariance.py
git commit -m "test: sweep equivariance across every registered embedding variant"
```

---

### Task 12: Reproduce AUCs, archive the branches, document the registry

The acceptance bar from the spec: tests passing is not sufficient.

**Files:**
- Modify: `docs/reference/variant-auc-baselines.md`
- Create: `docs/reference/embedding-variants.md`
- Modify: `README.md`

- [ ] **Step 1: Regenerate the aux dataset with every feature**

```bash
.venv/bin/python scripts/precompute_features.py \
  --features axes,masses,phases,leading \
  data/toptagging_mini.npz data/toptagging_mini_aux.npz
```

Expected: one `mode=... feature=...` line per (mode, feature) pair, then `wrote data/toptagging_mini_aux.npz`.

- [ ] **Step 2: Run each variant's mini config**

For each of the eight configs, run the experiment and record the final test AUC:

```bash
for cfg in top_tagging_grade2_bivectors top_tagging_subjet_mass \
           top_tagging_pseudoscalar_triad top_tagging_pseudoscalar_handedness \
           top_tagging_pseudoscalar_phase top_tagging_single_channel_axis \
           top_tagging_r9_scalar top_tagging_bivector_patch; do
  echo "===== $cfg"
  .venv/bin/python -m experiments.top_tagging.experiment --config-name "$cfg" \
    2>&1 | tee "outputs/unified_${cfg}.log" | grep -E "Test AUC"
done
```

Every config must set `training.seed` so the run is reproducible. If a config lacks it, add `seed: 0` before running.

- [ ] **Step 3: Compare against the baselines**

Fill in the "unified AUC" column of `docs/reference/variant-auc-baselines.md`. For each variant:

- Baseline and unified agree to the recorded precision → mark the variant **reproduced**.
- They disagree → the embedding fixture already passed, so the difference is in the training path, not the embedding. Diff the branch's config against the unified one (`git show <branch>:config/<file>`) before touching any code.
- Baseline recorded as "none recorded" → run that branch's own mini config in a worktree to generate the baseline first, then compare.

- [ ] **Step 4: Write the variant reference**

Create `docs/reference/embedding-variants.md`: a table mapping each old branch name to its `data.embedding` / `data.inject` / `model.*` config, the aux features it needs, and its config file. This is what makes the branches safe to delete — it is the index from "the R5b experiment" to "how do I run that now".

- [ ] **Step 5: Update the README**

Add a short section: how to precompute features, how to select a variant by config, and a pointer to `docs/reference/embedding-variants.md`. Do not restate the physics — link to the spec.

- [ ] **Step 6: Archive the branches**

Only for variants marked reproduced in Step 3.

```bash
for b in r4-explore r4-multiaxis-bivector r5a-pseudoscalar-triad \
         r5b-pseudoscalar-hand r5c-pseudoscalar-phase r5d-subjet-mass \
         r7-patch-injection r8-single-channel-ablation \
         r9-subjet-channel-vs-scalar; do
  git tag "archive/$b" "$b"
done
git push origin --tags
```

Verify each tag resolves before deleting anything:

```bash
for b in r4-explore r4-multiaxis-bivector r5a-pseudoscalar-triad \
         r5b-pseudoscalar-hand r5c-pseudoscalar-phase r5d-subjet-mass \
         r7-patch-injection r8-single-channel-ablation \
         r9-subjet-channel-vs-scalar; do
  echo "$b -> $(git rev-parse --short "archive/$b")"
done
```

Expected: nine lines, each with a commit hash matching `git rev-parse --short <branch>`.

**Deleting the branches is the user's call, not the implementer's.** Report the tag list and stop. `r10a-wedge-basis`, `r10b-frozen-basis` and `r10-rotor-primitive` are neither tagged nor deleted — they stay live.

- [ ] **Step 7: Commit**

```bash
git add docs/reference README.md
git commit -m "docs: record variant AUC reproduction and the embedding variant index"
```

---

## Notes for the executor

- If an equivalence fixture mismatches, the fault is in the port, not the fixture. Resist adjusting tolerances — `torch.equal` is intentional.
- The three rescale interactions that break bit-exactness, in order of likelihood: an injector rescaling when its branch did not (r5a's `chi_triad`, r5c's phases), an injector *not* rescaling when its branch did (r5b's `p3`, r5d's masses), and arithmetic reordering around a `clamp_min`.
- `git worktree remove --force` after each capture in Task 1; a stale worktree blocks the branch from being tagged later.
