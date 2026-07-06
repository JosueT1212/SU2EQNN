# Top Tagging Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a binary jet classification (top quark vs QCD) experiment using GATr on Cl(3,0), with torch_geometric sparse jets, Hydra config, and block-diagonal attention masking.

**Architecture:** 4-momenta (E,px,py,pz) embedded as Cl(3,0) multivectors [E,px,py,pz,0,0,0,0]; ±z beam spurion tokens prepended to each jet; GATr processes the sparse token sequence with block-diagonal attention; grade-0 of output MV is mean-pooled per jet to produce a classification logit.

**Tech Stack:** PyTorch, torch-geometric, hydra-core, omegaconf, scikit-learn, su2eqnn (GATr, EquiLinear, SelfAttentionConfig, MLPConfig)

## Global Constraints

- MV layout: index 0 = grade-0 (scalar), indices 1–3 = grade-1 (e1,e2,e3), indices 4–6 = grade-2 (e12,e13,e23), index 7 = grade-3 (e123)
- `UNITS = 20.0` — divide all 4-momenta components by 20 GeV before embedding
- Beam spurions: `+z = [0,0,0,1,0,0,0,0]`, `−z = [0,0,0,−1,0,0,0,0]` (index 3 is pz / e3)
- Attention mask: float tensor, `0.0` where tokens can attend, `float("-inf")` where blocked
- GATr configured with `in_mv_channels=1, out_mv_channels=1, in_s_channels=None, out_s_channels=None, hidden_s_channels=None`
- No scalar feature stream (first experiment; `add_scalar_features: false`)
- Dataset EPS = `1e-5` — drop zero-padded particles where ALL 4 components have absolute value ≤ EPS
- All new Python files go under `su2eqnn/` or `experiments/` or `config/` — no other locations
- Tests go in `tests/test_top_tagging.py`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add torch-geometric, hydra-core, omegaconf, scikit-learn |
| `data/` | Create symlink | `data/toptagging_mini.npz → ../lorentz-gatr/data/toptagging_mini.npz` |
| `config/top_tagging.yaml` | Create | Hydra config: data, model, training, evaluation sections |
| `experiments/__init__.py` | Create | Empty package marker |
| `experiments/top_tagging/__init__.py` | Create | Empty package marker |
| `su2eqnn/interface/spurions.py` | Create | `embed_spurions`, `get_num_spurions` for Cl(3,0) z-axis beams |
| `su2eqnn/interface/__init__.py` | Modify | Export `embed_spurions`, `get_num_spurions` |
| `experiments/top_tagging/dataset.py` | Create | `TopTaggingDataset` loading .npz → torch_geometric Data objects |
| `experiments/top_tagging/embedding.py` | Create | `embed_tagging_data`, `get_batch_from_ptr` |
| `experiments/top_tagging/wrapper.py` | Create | `build_attention_mask`, `TopTaggingWrapper` |
| `experiments/top_tagging/experiment.py` | Create | Hydra `main`, training loop, eval |
| `tests/test_top_tagging.py` | Create | All tests for new code |

---

### Task 1: Setup — dependencies, directories, config

**Files:**
- Modify: `pyproject.toml`
- Create: `data/` (symlink), `config/top_tagging.yaml`
- Create: `experiments/__init__.py`, `experiments/top_tagging/__init__.py`

**Interfaces:**
- Produces: `data/toptagging_mini.npz` accessible as a path; `config/top_tagging.yaml` loadable by Hydra

- [ ] **Step 1: Install dependencies**

```bash
cd /Users/josuetapiahernandez/Documents/Investigacion_Andres/SU2EQNN
.venv/bin/pip install torch-geometric hydra-core omegaconf scikit-learn
```

Expected: packages install without error.

- [ ] **Step 2: Add dependencies to `pyproject.toml`**

Current `pyproject.toml` has a `dependencies` list. Add the four new packages:

```toml
[project]
name = "su2eqnn"
version = "0.1.0"
description = "Spin(3) ≅ SU(2) equivariant neural network on Cl(3,0) — stepping stone toward L-GATr + Furey construction"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "clifford>=1.4",
    "numpy",
    "opt_einsum",
    "einops",
    "torch-geometric",
    "hydra-core",
    "omegaconf",
    "scikit-learn",
]
```

- [ ] **Step 3: Create data symlink**

```bash
mkdir -p data
ln -s ../../lorentz-gatr/data/toptagging_mini.npz data/toptagging_mini.npz
```

Verify: `ls -la data/toptagging_mini.npz` shows symlink.

- [ ] **Step 4: Create package markers**

Create `experiments/__init__.py`:
```python
```
(empty file)

Create `experiments/top_tagging/__init__.py`:
```python
```
(empty file)

- [ ] **Step 5: Create `config/top_tagging.yaml`**

```bash
mkdir -p config
```

Write `config/top_tagging.yaml`:

```yaml
data:
  path: data/toptagging_mini.npz
  rescale_data: true
  two_beams: true

model:
  hidden_mv_channels: 16
  num_blocks: 6
  num_heads: 4

training:
  epochs: 30
  batch_size: 32
  lr: 1.0e-3
  validate_every_n_steps: 200
  checkpoint_path: checkpoints/top_tagging_best.pt

evaluation:
  batch_size: 64
```

- [ ] **Step 6: Verify data path works**

```bash
.venv/bin/python -c "import numpy as np; d=np.load('data/toptagging_mini.npz'); print(list(d.keys()), {k: d[k].shape for k in d.keys()})"
```

Expected output:
```
['kinematics_train', 'kinematics_test', 'kinematics_val', 'labels_train', 'labels_test', 'labels_val']
{'kinematics_train': (20000, 200, 4), 'kinematics_test': (2000, 200, 4), 'kinematics_val': (2000, 200, 4), 'labels_train': (20000,), 'labels_test': (2000,), 'labels_val': (2000,)}
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml data config experiments
git commit -m "feat: setup top tagging experiment infrastructure"
```

---

### Task 2: Beam spurions (`su2eqnn/interface/spurions.py`)

**Files:**
- Create: `su2eqnn/interface/spurions.py`
- Modify: `su2eqnn/interface/__init__.py`
- Test: `tests/test_top_tagging.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `get_num_spurions(two_beams: bool) -> int`
  - `embed_spurions(two_beams: bool, device, dtype) -> Tensor shape (n_spurions, 1, 8)`

- [ ] **Step 1: Write failing tests**

Create `tests/test_top_tagging.py`:

```python
"""Tests for top tagging experiment components."""

import pytest
import torch
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "toptagging_mini.npz"

# ---------------------------------------------------------------------------
# Task 2: Spurions
# ---------------------------------------------------------------------------

from su2eqnn.interface.spurions import embed_spurions, get_num_spurions


def test_get_num_spurions_two_beams():
    assert get_num_spurions(two_beams=True) == 2


def test_get_num_spurions_one_beam():
    assert get_num_spurions(two_beams=False) == 1


def test_embed_spurions_shape_two_beams():
    s = embed_spurions(two_beams=True)
    assert s.shape == (2, 1, 8)


def test_embed_spurions_shape_one_beam():
    s = embed_spurions(two_beams=False)
    assert s.shape == (1, 1, 8)


def test_embed_spurions_pz_values():
    s = embed_spurions(two_beams=True)
    # index 3 = e3 = pz direction (grade-1)
    assert s[0, 0, 3].item() == pytest.approx(1.0)   # +z beam
    assert s[1, 0, 3].item() == pytest.approx(-1.0)  # -z beam


def test_embed_spurions_only_pz_nonzero():
    s = embed_spurions(two_beams=True)
    # all components except index 3 must be zero
    mask = torch.ones(8, dtype=torch.bool)
    mask[3] = False
    assert (s[:, :, mask] == 0).all()


def test_embed_spurions_dtype():
    s = embed_spurions(dtype=torch.float64)
    assert s.dtype == torch.float64


def test_embed_spurions_device():
    s = embed_spurions(device=torch.device("cpu"))
    assert s.device.type == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_top_tagging.py::test_get_num_spurions_two_beams -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'su2eqnn.interface.spurions'`

- [ ] **Step 3: Implement `su2eqnn/interface/spurions.py`**

```python
"""Cl(3,0) beam spurion embeddings for Spin(3) symmetry breaking.

Injects z-axis reference vectors to break SO(3) → SO(2).
"""

import torch


def get_num_spurions(two_beams: bool = True) -> int:
    """Return number of beam spurion tokens."""
    return 2 if two_beams else 1


def embed_spurions(
    two_beams: bool = True,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Return z-axis beam spurion MV tokens for Cl(3,0).

    Cl(3,0) layout: index 3 = e3 (grade-1 z-direction).

    Returns
    -------
    spurions : Tensor shape (n_spurions, 1, 8)
        spurions[0, 0] = [0, 0, 0, +1, 0, 0, 0, 0]  # +z beam
        spurions[1, 0] = [0, 0, 0, -1, 0, 0, 0, 0]  # -z beam (if two_beams)
    """
    kwargs = {"device": device, "dtype": dtype}
    beam = torch.zeros(1, 1, 8, **kwargs)
    beam[0, 0, 3] = 1.0
    if two_beams:
        beam_neg = beam.clone()
        beam_neg[0, 0, 3] = -1.0
        return torch.cat([beam, beam_neg], dim=0)  # (2, 1, 8)
    return beam  # (1, 1, 8)
```

- [ ] **Step 4: Update `su2eqnn/interface/__init__.py`**

```python
from su2eqnn.interface.vector import embed_vector, extract_vector
from su2eqnn.interface.scalar import embed_scalar, extract_scalar
from su2eqnn.interface.spurions import embed_spurions, get_num_spurions
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_top_tagging.py -k "spurion or num_spurion" -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add su2eqnn/interface/spurions.py su2eqnn/interface/__init__.py tests/test_top_tagging.py
git commit -m "feat: add Cl(3,0) beam spurion embeddings for top tagging"
```

---

### Task 3: Dataset (`experiments/top_tagging/dataset.py`)

**Files:**
- Create: `experiments/top_tagging/dataset.py`
- Test: `tests/test_top_tagging.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `EPS = 1e-5`
  - `TopTaggingDataset(rescale_data: bool)` — `torch.utils.data.Dataset`
    - `load_data(filename: str, mode: str, dtype=torch.float32) -> None`
    - `__len__() -> int`
    - `__getitem__(idx: int) -> torch_geometric.data.Data`
    - Each `Data` object has: `x` shape `(n_real, 4)`, `label` shape `()` dtype bool

- [ ] **Step 1: Write failing tests (append to `tests/test_top_tagging.py`)**

```python
# ---------------------------------------------------------------------------
# Task 3: Dataset
# ---------------------------------------------------------------------------

from experiments.top_tagging.dataset import TopTaggingDataset, EPS


def test_dataset_train_length():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    assert len(ds) == 20000


def test_dataset_test_length():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "test")
    assert len(ds) == 2000


def test_dataset_item_shape():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    item = ds[0]
    assert item.x.ndim == 2
    assert item.x.shape[1] == 4
    assert 1 <= item.x.shape[0] <= 200


def test_dataset_label_dtype():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    assert ds[0].label.dtype == torch.bool


def test_dataset_drops_padding():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "test")
    for item in ds.data_list[:20]:
        # no all-zero rows — any component abs > EPS
        assert (item.x.abs() > EPS).any(dim=-1).all(), \
            "Zero-padded row not filtered"


def test_dataset_fourmomenta_dtype():
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")
    assert ds[0].x.dtype == torch.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_top_tagging.py -k "dataset" -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`

- [ ] **Step 3: Implement `experiments/top_tagging/dataset.py`**

```python
"""TopTagging dataset for binary jet classification.

Loads .npz files with format:
  kinematics_{mode}: shape (N, 200, 4) — E, px, py, pz per particle
  labels_{mode}:     shape (N,)        — bool, True = top quark jet
"""

import numpy as np
import torch
from torch_geometric.data import Data

EPS = 1e-5


class TopTaggingDataset(torch.utils.data.Dataset):
    """Binary jet classification dataset using torch_geometric Data objects.

    Parameters
    ----------
    rescale_data : bool
        Stored but unused here — rescaling is applied in embed_tagging_data.
    """

    def __init__(self, rescale_data: bool):
        self.rescale_data = rescale_data
        self.data_list = []

    def load_data(self, filename: str, mode: str, dtype=torch.float32):
        """Load jets from .npz file.

        Parameters
        ----------
        filename : str  path to .npz file
        mode     : {"train", "val", "test"}
        """
        raw = np.load(filename)
        kinematics = torch.tensor(raw[f"kinematics_{mode}"], dtype=dtype)
        labels = torch.tensor(raw[f"labels_{mode}"], dtype=torch.bool)

        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded particles: keep rows where any |component| > EPS
            mask = (kinematics[i].abs() > EPS).any(dim=-1)
            fourmomenta = kinematics[i][mask]   # (n_real, 4)
            label = labels[i]                    # scalar bool
            self.data_list.append(Data(x=fourmomenta, label=label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_top_tagging.py -k "dataset" -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/top_tagging/__init__.py experiments/top_tagging/dataset.py tests/test_top_tagging.py
git commit -m "feat: add TopTaggingDataset for top tagging experiment"
```

---

### Task 4: Embedding (`experiments/top_tagging/embedding.py`)

**Files:**
- Create: `experiments/top_tagging/embedding.py`
- Test: `tests/test_top_tagging.py` (append)

**Interfaces:**
- Consumes: `embed_spurions(two_beams, device, dtype)` from Task 2
- Produces:
  - `UNITS = 20.0`
  - `get_batch_from_ptr(ptr: Tensor) -> Tensor` — shape `(n_total,)`
  - `embed_tagging_data(fourmomenta, ptr, two_beams, rescale) -> dict`
    - keys: `"mv"` shape `(n_total, 1, 8)`, `"batch"` shape `(n_total,)`, `"is_beam"` shape `(n_total,)` bool

- [ ] **Step 1: Write failing tests (append to `tests/test_top_tagging.py`)**

```python
# ---------------------------------------------------------------------------
# Task 4: Embedding
# ---------------------------------------------------------------------------

from experiments.top_tagging.embedding import embed_tagging_data, get_batch_from_ptr, UNITS


def test_get_batch_from_ptr():
    ptr = torch.tensor([0, 3, 5])
    batch = get_batch_from_ptr(ptr)
    expected = torch.tensor([0, 0, 0, 1, 1])
    assert (batch == expected).all()


def test_embed_tagging_data_total_tokens():
    # 2 jets: jet0=3 particles, jet1=2 particles; two_beams=True → 2 beams per jet
    fourmomenta = torch.randn(5, 4)
    ptr = torch.tensor([0, 3, 5])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    # 5 particles + 2*2 beam tokens = 9 total
    assert result["mv"].shape == (9, 1, 8)
    assert result["batch"].shape == (9,)
    assert result["is_beam"].shape == (9,)


def test_embed_tagging_data_beam_count():
    fourmomenta = torch.randn(5, 4)
    ptr = torch.tensor([0, 3, 5])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    # 2 jets * 2 beam tokens = 4 beam tokens
    assert result["is_beam"].sum().item() == 4


def test_embed_tagging_data_beam_values():
    # 1 jet, 3 particles
    fourmomenta = torch.zeros(3, 4)
    ptr = torch.tensor([0, 3])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    beam_mvs = result["mv"][result["is_beam"]]  # (2, 1, 8)
    assert beam_mvs[0, 0, 3].item() == pytest.approx(1.0)   # +z
    assert beam_mvs[1, 0, 3].item() == pytest.approx(-1.0)  # -z


def test_embed_tagging_data_4momentum_placement():
    # E=10, px=1, py=2, pz=3
    fourmomenta = torch.tensor([[10.0, 1.0, 2.0, 3.0]])
    ptr = torch.tensor([0, 1])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=False, rescale=False)
    particle_mv = result["mv"][~result["is_beam"]][0, 0]  # (8,)
    assert particle_mv[0].item() == pytest.approx(10.0)  # E → grade-0
    assert particle_mv[1].item() == pytest.approx(1.0)   # px → e1
    assert particle_mv[2].item() == pytest.approx(2.0)   # py → e2
    assert particle_mv[3].item() == pytest.approx(3.0)   # pz → e3
    assert (particle_mv[4:] == 0).all()


def test_embed_tagging_data_rescale():
    fourmomenta = torch.tensor([[20.0, 20.0, 20.0, 20.0]])
    ptr = torch.tensor([0, 1])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=False, rescale=True)
    particle_mv = result["mv"][~result["is_beam"]][0, 0]  # (8,)
    assert particle_mv[0].item() == pytest.approx(1.0)  # E/20 = 1.0
    assert particle_mv[1].item() == pytest.approx(1.0)  # px/20 = 1.0


def test_embed_tagging_data_beam_prepended():
    # Verify beam tokens come first in each jet's subsequence
    fourmomenta = torch.zeros(3, 4)  # 1 jet, 3 particles
    ptr = torch.tensor([0, 3])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    # Output order: [beam+z, beam-z, p0, p1, p2]
    assert result["is_beam"][0].item() is True  # first token is beam
    assert result["is_beam"][1].item() is True  # second token is beam
    assert not result["is_beam"][2].item()       # third token is particle


def test_embed_tagging_data_batch_indices():
    # 2 jets: 3 particles + 2 beams, 2 particles + 2 beams → jet0 has 5 tokens, jet1 has 4 tokens
    fourmomenta = torch.zeros(5, 4)
    ptr = torch.tensor([0, 3, 5])
    result = embed_tagging_data(fourmomenta, ptr, two_beams=True, rescale=False)
    batch = result["batch"]
    assert (batch[:5] == 0).all()  # first 5 tokens belong to jet 0
    assert (batch[5:] == 1).all()  # last 4 tokens belong to jet 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_top_tagging.py -k "embed_tagging or batch_from_ptr" -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.top_tagging.embedding'`

- [ ] **Step 3: Implement `experiments/top_tagging/embedding.py`**

```python
"""Embedding of 4-momenta into Cl(3,0) multivectors for jet tagging.

Particle MV layout: [E, px, py, pz, 0, 0, 0, 0]
  - E  → index 0 (grade-0, Spin(3)-invariant)
  - px → index 1 (grade-1 e1)
  - py → index 2 (grade-1 e2)
  - pz → index 3 (grade-1 e3)
"""

import torch
from torch import Tensor

from su2eqnn.interface.spurions import embed_spurions

UNITS = 20.0  # GeV — all momenta divided by this before embedding


def get_batch_from_ptr(ptr: Tensor) -> Tensor:
    """Convert ptr to batch index tensor.

    Parameters
    ----------
    ptr : Tensor shape (batchsize+1,) — cumulative token counts per jet

    Returns
    -------
    batch : Tensor shape (n_total,) — jet index for each token
    """
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1]
    )


def embed_tagging_data(
    fourmomenta: Tensor,
    ptr: Tensor,
    two_beams: bool = True,
    rescale: bool = True,
) -> dict:
    """Embed sparse 4-momenta as Cl(3,0) MVs and prepend beam spurion tokens.

    Parameters
    ----------
    fourmomenta : Tensor shape (n_particles, 4) — sparse, zero-padded rows already removed
    ptr         : Tensor shape (batchsize+1,) — cumulative particle counts per jet
    two_beams   : bool — whether to prepend both +z and -z beam tokens
    rescale     : bool — divide fourmomenta by UNITS before embedding

    Returns
    -------
    dict with keys:
        "mv"      : Tensor (n_total, 1, 8)
        "batch"   : Tensor (n_total,) — jet index per token
        "is_beam" : Tensor (n_total,) bool — True for beam tokens
    """
    batchsize = len(ptr) - 1

    if rescale:
        fourmomenta = fourmomenta / UNITS

    # Build particle MVs: (n_particles, 1, 8)
    mv = torch.zeros(
        fourmomenta.shape[0], 1, 8,
        dtype=fourmomenta.dtype, device=fourmomenta.device,
    )
    mv[:, 0, 0] = fourmomenta[:, 0]  # E  → grade-0
    mv[:, 0, 1] = fourmomenta[:, 1]  # px → e1
    mv[:, 0, 2] = fourmomenta[:, 2]  # py → e2
    mv[:, 0, 3] = fourmomenta[:, 3]  # pz → e3

    # Get beam spurions: (n_spurions, 1, 8)
    spurions = embed_spurions(
        two_beams=two_beams,
        device=fourmomenta.device,
        dtype=fourmomenta.dtype,
    )
    n_spurions = spurions.shape[0]

    # Compute output positions for beam tokens after interleaving.
    # Each jet's slice gets n_spurions beam tokens prepended.
    # Jet j's first token in output = ptr[j] + j * n_spurions
    # Beam token i of jet j lands at: ptr[j] + j*n_spurions + i
    arange = torch.arange(batchsize, device=ptr.device)
    spurion_idxs = torch.stack(
        [ptr[:-1] + i for i in range(n_spurions)], dim=0
    ) + n_spurions * arange          # (n_spurions, batchsize)
    spurion_idxs = spurion_idxs.permute(1, 0).flatten()  # (n_spurions*batchsize,)

    n_total = mv.shape[0] + n_spurions * batchsize
    is_beam = torch.zeros(n_total, dtype=torch.bool, device=mv.device)
    is_beam[spurion_idxs] = True

    mv_out = torch.empty(n_total, 1, 8, dtype=mv.dtype, device=mv.device)
    mv_out[~is_beam] = mv
    mv_out[is_beam] = spurions.repeat(batchsize, 1, 1)  # (n_spurions*batchsize, 1, 8)

    # Update ptr: each jet j gains j*n_spurions extra tokens before it
    ptr_out = ptr.clone()
    ptr_out[1:] = ptr[1:] + (arange + 1) * n_spurions

    batch = get_batch_from_ptr(ptr_out)

    return {"mv": mv_out, "batch": batch, "is_beam": is_beam}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_top_tagging.py -k "embed_tagging or batch_from_ptr" -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/top_tagging/embedding.py tests/test_top_tagging.py
git commit -m "feat: add 4-momentum embedding with beam spurion injection for top tagging"
```

---

### Task 5: Wrapper (`experiments/top_tagging/wrapper.py`)

**Files:**
- Create: `experiments/top_tagging/wrapper.py`
- Test: `tests/test_top_tagging.py` (append)

**Interfaces:**
- Consumes:
  - `GATr(in_mv_channels, out_mv_channels, hidden_mv_channels, in_s_channels, out_s_channels, hidden_s_channels, attention, mlp, num_blocks)` from `su2eqnn.nets.gatr`
  - embedding dict from Task 4: `{"mv": (n_total, 1, 8), "batch": (n_total,), "is_beam": (n_total,) bool}`
- Produces:
  - `build_attention_mask(batch: Tensor, dtype: torch.dtype) -> Tensor shape (n_total, n_total)`
  - `TopTaggingWrapper(net: nn.Module)` — `forward(embedding: dict) -> Tensor shape (batchsize, 1)`

- [ ] **Step 1: Write failing tests (append to `tests/test_top_tagging.py`)**

```python
# ---------------------------------------------------------------------------
# Task 5: Wrapper
# ---------------------------------------------------------------------------

from experiments.top_tagging.wrapper import build_attention_mask, TopTaggingWrapper
from su2eqnn.nets.gatr import GATr
from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.mlp.config import MLPConfig


def _make_tiny_net():
    return GATr(
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=4,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=2),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=2,
    )


def test_build_attention_mask_shape():
    batch = torch.tensor([0, 0, 0, 1, 1])
    mask = build_attention_mask(batch, torch.float32)
    assert mask.shape == (5, 5)


def test_build_attention_mask_within_jet_zero():
    batch = torch.tensor([0, 0, 0, 1, 1])
    mask = build_attention_mask(batch, torch.float32)
    # within jet 0: rows/cols 0-2
    assert (mask[:3, :3] == 0.0).all()
    # within jet 1: rows/cols 3-4
    assert (mask[3:, 3:] == 0.0).all()


def test_build_attention_mask_cross_jet_neginf():
    batch = torch.tensor([0, 0, 0, 1, 1])
    mask = build_attention_mask(batch, torch.float32)
    assert (mask[:3, 3:] == float("-inf")).all()
    assert (mask[3:, :3] == float("-inf")).all()


def test_wrapper_output_shape():
    net = _make_tiny_net()
    wrapper = TopTaggingWrapper(net)
    # 2 jets: jet0 has 3 tokens, jet1 has 2 tokens
    mv = torch.randn(5, 1, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])
    is_beam = torch.zeros(5, dtype=torch.bool)
    logits = wrapper({"mv": mv, "batch": batch, "is_beam": is_beam})
    assert logits.shape == (2, 1)


def test_wrapper_output_dtype():
    net = _make_tiny_net()
    wrapper = TopTaggingWrapper(net)
    mv = torch.randn(5, 1, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])
    is_beam = torch.zeros(5, dtype=torch.bool)
    logits = wrapper({"mv": mv, "batch": batch, "is_beam": is_beam})
    assert logits.dtype == torch.float32


def test_wrapper_gradient_flows():
    net = _make_tiny_net()
    wrapper = TopTaggingWrapper(net)
    mv = torch.randn(5, 1, 8)
    batch = torch.tensor([0, 0, 0, 1, 1])
    is_beam = torch.zeros(5, dtype=torch.bool)
    logits = wrapper({"mv": mv, "batch": batch, "is_beam": is_beam})
    loss = logits.sum()
    loss.backward()
    # at least one parameter should have gradient
    has_grad = any(p.grad is not None for p in net.parameters())
    assert has_grad
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_top_tagging.py -k "attention_mask or wrapper" -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.top_tagging.wrapper'`

- [ ] **Step 3: Implement `experiments/top_tagging/wrapper.py`**

```python
"""GATr wrapper for binary jet classification (top tagging).

Architecture:
  embed → GATr → extract grade-0 from MV output → mean pool per jet → logit
"""

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


def build_attention_mask(batch: Tensor, dtype: torch.dtype) -> Tensor:
    """Block-diagonal additive float attention mask.

    Tokens in the same jet can attend to each other.
    Tokens in different jets cannot attend to each other.

    Parameters
    ----------
    batch : Tensor shape (n_total,) — jet index per token
    dtype : float dtype for the mask

    Returns
    -------
    mask : Tensor shape (n_total, n_total)
        0.0 where tokens can attend; float("-inf") where blocked.
    """
    same_jet = batch.unsqueeze(0) == batch.unsqueeze(1)  # (n_total, n_total)
    mask = torch.full(
        (len(batch), len(batch)), float("-inf"),
        dtype=dtype, device=batch.device,
    )
    mask[same_jet] = 0.0
    return mask


class TopTaggingWrapper(nn.Module):
    """GATr + mean-pool head for binary jet classification.

    Parameters
    ----------
    net : GATr configured with in_mv_channels=1, out_mv_channels=1,
          in_s_channels=None, out_s_channels=None
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, embedding: dict) -> Tensor:
        """Classify jets.

        Parameters
        ----------
        embedding : dict from embed_tagging_data
            "mv"      : Tensor (n_total, 1, 8)
            "batch"   : Tensor (n_total,) jet index per token
            "is_beam" : Tensor (n_total,) bool  (not used for masking here)

        Returns
        -------
        logits : Tensor (batchsize, 1)
        """
        mv = embedding["mv"].unsqueeze(0)   # (1, n_total, 1, 8)
        batch = embedding["batch"]          # (n_total,)

        mask = build_attention_mask(batch, mv.dtype)   # (n_total, n_total)

        mv_out, _ = self.net(mv, scalars=None, attention_mask=mask)
        # mv_out: (1, n_total, out_mv_channels=1, 8)

        # Extract grade-0 (Spin(3)-invariant) as classification signal
        scalars_out = mv_out[0, :, :, 0]  # (n_total, 1)

        batchsize = int(batch.max().item()) + 1
        logits = scatter(scalars_out, batch, dim=0, dim_size=batchsize, reduce="mean")
        return logits  # (batchsize, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_top_tagging.py -k "attention_mask or wrapper" -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/top_tagging/wrapper.py tests/test_top_tagging.py
git commit -m "feat: add TopTaggingWrapper with block-diagonal attention mask"
```

---

### Task 6: Experiment (`experiments/top_tagging/experiment.py`)

**Files:**
- Create: `experiments/top_tagging/experiment.py`
- Test: `tests/test_top_tagging.py` (append)

**Interfaces:**
- Consumes (all from earlier tasks):
  - `TopTaggingDataset` (Task 3)
  - `embed_tagging_data` (Task 4)
  - `TopTaggingWrapper` (Task 5)
  - `GATr`, `SelfAttentionConfig`, `MLPConfig`
- Produces:
  - `build_model(cfg) -> TopTaggingWrapper` — builds GATr + wrapper from Hydra config
  - `run_experiment(cfg) -> None` — training + eval entry point (called by `main`)
  - `main` — Hydra entry point, callable as `python -m experiments.top_tagging.experiment`

- [ ] **Step 1: Write failing test (append to `tests/test_top_tagging.py`)**

```python
# ---------------------------------------------------------------------------
# Task 6: Experiment — smoke test
# ---------------------------------------------------------------------------

from experiments.top_tagging.experiment import build_model
from torch_geometric.loader import DataLoader as GeometricDataLoader
from experiments.top_tagging.embedding import embed_tagging_data


class _FakeCfg:
    """Minimal config object for testing."""
    class model:
        hidden_mv_channels = 4
        num_blocks = 2
        num_heads = 2
    class data:
        rescale_data = True
        two_beams = True
    class training:
        lr = 1e-3


def test_build_model_returns_wrapper():
    model = build_model(_FakeCfg())
    assert isinstance(model, TopTaggingWrapper)


def test_smoke_one_training_step():
    """One forward + backward through the full pipeline on real data."""
    ds = TopTaggingDataset(rescale_data=False)
    ds.load_data(str(DATA_PATH), "train")

    loader = GeometricDataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = build_model(_FakeCfg())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    embedding = embed_tagging_data(
        batch.x, batch.ptr,
        two_beams=True, rescale=True,
    )
    logits = model(embedding)                              # (4, 1)
    targets = batch.label.float().unsqueeze(1)            # (4, 1)
    loss = criterion(logits, targets)

    assert loss.ndim == 0             # scalar
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # parameters changed after optimizer step
    p0 = next(model.parameters())
    assert p0.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_top_tagging.py::test_build_model_returns_wrapper -v
```

Expected: FAIL with `ImportError: cannot import name 'build_model'`

- [ ] **Step 3: Implement `experiments/top_tagging/experiment.py`**

```python
"""Top quark jet tagging experiment using GATr on Cl(3,0).

Entry point:
    python -m experiments.top_tagging.experiment

Hydra config: config/top_tagging.yaml
"""

import os
import time
import logging

import torch
import hydra
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from su2eqnn.nets.gatr import GATr
from su2eqnn.layers.attention.config import SelfAttentionConfig
from su2eqnn.layers.mlp.config import MLPConfig
from experiments.top_tagging.dataset import TopTaggingDataset
from experiments.top_tagging.embedding import embed_tagging_data
from experiments.top_tagging.wrapper import TopTaggingWrapper

log = logging.getLogger(__name__)


def build_model(cfg) -> TopTaggingWrapper:
    """Instantiate GATr + TopTaggingWrapper from config.

    Parameters
    ----------
    cfg : DictConfig or object with .model attributes:
        hidden_mv_channels : int
        num_blocks         : int
        num_heads          : int
    """
    net = GATr(
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=cfg.model.hidden_mv_channels,
        in_s_channels=None,
        out_s_channels=None,
        hidden_s_channels=None,
        attention=SelfAttentionConfig(num_heads=cfg.model.num_heads),
        mlp=MLPConfig(activation="gelu"),
        num_blocks=cfg.model.num_blocks,
    )
    return TopTaggingWrapper(net)


def evaluate(model, loader, cfg, device):
    """Run evaluation on a DataLoader, return AUC and accuracy."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embedding = embed_tagging_data(
                batch.x, batch.ptr,
                two_beams=cfg.data.two_beams,
                rescale=cfg.data.rescale_data,
            )
            logits = model(embedding)  # (B, 1)
            all_logits.append(logits.cpu())
            all_labels.append(batch.label.cpu())

    logits = torch.cat(all_logits, dim=0).squeeze(1).numpy()
    labels = torch.cat(all_labels, dim=0).numpy().astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    auc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    return {"auc": auc, "accuracy": acc}


def run_experiment(cfg: DictConfig) -> None:
    """Training + evaluation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Data
    log.info("Loading datasets...")
    ds_train = TopTaggingDataset(rescale_data=cfg.data.rescale_data)
    ds_val = TopTaggingDataset(rescale_data=cfg.data.rescale_data)
    ds_test = TopTaggingDataset(rescale_data=cfg.data.rescale_data)
    ds_train.load_data(cfg.data.path, "train")
    ds_val.load_data(cfg.data.path, "val")
    ds_test.load_data(cfg.data.path, "test")

    loader_train = GeometricDataLoader(ds_train, batch_size=cfg.training.batch_size, shuffle=True)
    loader_val = GeometricDataLoader(ds_val, batch_size=cfg.evaluation.batch_size, shuffle=False)
    loader_test = GeometricDataLoader(ds_test, batch_size=cfg.evaluation.batch_size, shuffle=False)

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = 0.0
    ckpt_path = cfg.training.checkpoint_path
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader_train:
            batch = batch.to(device)
            embedding = embed_tagging_data(
                batch.x, batch.ptr,
                two_beams=cfg.data.two_beams,
                rescale=cfg.data.rescale_data,
            )
            logits = model(embedding)              # (B, 1)
            targets = batch.label.float().unsqueeze(1)  # (B, 1)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % cfg.training.validate_every_n_steps == 0:
                metrics = evaluate(model, loader_val, cfg, device)
                log.info(
                    f"Step {global_step} | val AUC={metrics['auc']:.4f} "
                    f"acc={metrics['accuracy']:.4f}"
                )
                if metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    torch.save(model.state_dict(), ckpt_path)
                    log.info(f"Saved best model (AUC={best_auc:.4f}) to {ckpt_path}")
                model.train()

        dt = time.time() - t0
        avg_loss = epoch_loss / len(loader_train)
        log.info(f"Epoch {epoch}/{cfg.training.epochs} | loss={avg_loss:.4f} | {dt:.1f}s")

    # Final test evaluation
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, loader_test, cfg, device)
    log.info(f"Test AUC={test_metrics['auc']:.4f} | accuracy={test_metrics['accuracy']:.4f}")


@hydra.main(config_path="../../config", config_name="top_tagging", version_base=None)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_top_tagging.py::test_build_model_returns_wrapper tests/test_top_tagging.py::test_smoke_one_training_step -v
```

Expected: both tests PASS. `test_smoke_one_training_step` may take 5–15 seconds.

- [ ] **Step 5: Run all top tagging tests**

```bash
.venv/bin/pytest tests/test_top_tagging.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Smoke-run one epoch**

```bash
.venv/bin/python -m experiments.top_tagging.experiment training.epochs=1 training.validate_every_n_steps=9999 training.batch_size=32
```

Expected: runs without error, prints epoch loss at the end.

- [ ] **Step 7: Commit**

```bash
git add experiments/top_tagging/experiment.py tests/test_top_tagging.py
git commit -m "feat: add top tagging training loop with Hydra config and BCE+AUC evaluation"
```

---

## Self-Review Checklist (completed)

1. **Spec coverage:**
   - `su2eqnn/interface/spurions.py` — Task 2 ✓
   - `experiments/top_tagging/dataset.py` — Task 3 ✓
   - `experiments/top_tagging/embedding.py` — Task 4 ✓
   - `experiments/top_tagging/wrapper.py` — Task 5 ✓
   - `experiments/top_tagging/experiment.py` — Task 6 ✓
   - `config/top_tagging.yaml` — Task 1 ✓
   - Beam spurion ±z tokens — Task 2 + Task 4 ✓
   - Block-diagonal float attention mask — Task 5 ✓
   - MV = [E, px, py, pz, 0,0,0,0] — Task 4 ✓
   - Test plan items (`test_embed_spurions_shape`, `test_embed_tagging_data_shape`, `test_dataset_load`, `test_wrapper_output_shape`, `test_build_attention_mask_block_diagonal`) — all covered ✓

2. **No placeholders** — all steps have complete code.

3. **Type consistency:**
   - `embed_tagging_data` returns `dict` with keys `"mv"`, `"batch"`, `"is_beam"` — used correctly in `TopTaggingWrapper.forward` ✓
   - `build_model(cfg)` returns `TopTaggingWrapper` — used in test and experiment ✓
   - `get_batch_from_ptr(ptr)` returns `Tensor` — used in `embed_tagging_data` ✓
