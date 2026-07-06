# Top Tagging Experiment Design

**Date:** 2026-07-06
**Status:** Approved

## Goal

Binary jet classification (top quark vs QCD) using `GATr` on Cl(3,0). 4-momenta embedded as MV=[E,px,py,pz,0,0,0,0]. Infrastructure matches L-GATr: Hydra configs, torch_geometric sparse jets, xformers-compatible block-diagonal attention mask.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding | MV=[E,px,py,pz,0,0,0,0], rescale ÷20 | E→grade-0 (Spin(3)-invariant), (px,py,pz)→grade-1 (equivariant), matches L-GATr scale |
| Beam spurions | ±z tokens [0,0,0,±1,0,0,0,0] prepended to sequence | Breaks SO(3)→SO(2); matches L-GATr beam_token pattern |
| Scalar features | None by default (`add_scalar_features: false`) | L-GATr default; can add log_pt etc. later |
| Aggregation | Mean over all tokens (including beam) | Simpler; no global token needed |
| Attention masking | Block-diagonal float mask via `attention_mask` | SelfAttention already accepts `(..., items, items)` mask; avoids batch_indices plumbing |
| MLflow | Optional (`use_mlflow: false` default) | L-GATr uses it, but not critical for first run |

## Files

```
su2eqnn/interface/spurions.py                    # NEW — Cl(3,0) beam spurions
experiments/__init__.py                          # NEW (empty)
experiments/top_tagging/__init__.py              # NEW (empty)
experiments/top_tagging/embedding.py             # NEW — 4-momentum → MV, inject beam tokens
experiments/top_tagging/dataset.py               # NEW — TopTaggingDataset (torch_geometric)
experiments/top_tagging/wrapper.py               # NEW — GATr + classification head
experiments/top_tagging/experiment.py            # NEW — Hydra training + eval loop
config/top_tagging.yaml                          # NEW — Hydra config
```

---

## Component Specs

### `su2eqnn/interface/spurions.py`

```python
def get_num_spurions(two_beams: bool = True) -> int:
    """Returns 2 if two_beams else 1."""

def embed_spurions(
    two_beams: bool = True,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns z-axis beam spurion MV tokens.

    Returns
    -------
    spurions : Tensor of shape (n_spurions, 1, 8)
        spurions[0] = [0, 0, 0, 1, 0, 0, 0, 0]   # +z beam
        spurions[1] = [0, 0, 0,-1, 0, 0, 0, 0]   # -z beam (if two_beams)
    Shape: (1, 1, 8) or (2, 1, 8)
    """
```

---

### `experiments/top_tagging/embedding.py`

```python
UNITS = 20.0  # GeV — same as L-GATr

def get_batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
    """ptr (batchsize+1,) → batch (n_total,) index tensor."""

def embed_tagging_data(
    fourmomenta: torch.Tensor,   # (n_particles, 4)  sparse, already masked
    ptr: torch.Tensor,           # (batchsize+1,)
    cfg_data,                    # Hydra DictConfig
) -> dict:
    """
    Embed sparse 4-momenta into Cl(3,0) MVs and prepend beam spurion tokens.

    Returns
    -------
    dict with keys:
        "mv"    : Tensor (n_total, 1, 8)   — particles + beam tokens interleaved
        "s"     : None  (no scalar features by default)
        "batch" : Tensor (n_total,)         — jet index per token
        "is_beam": Tensor (n_total,) bool  — True for beam tokens, False for particles
    """

    # 1. Rescale: fourmomenta /= UNITS
    # 2. Build particle MVs: (n_particles, 1, 8)
    #    mv[..., 0] = E, mv[..., 1] = px, mv[..., 2] = py, mv[..., 3] = pz
    # 3. Get beam spurions: (n_spurions, 1, 8) via embed_spurions(cfg_data.two_beams, ...)
    # 4. Prepend beam tokens into sequence (same interleave logic as L-GATr embedding.py:93-122)
    #    Insert n_spurions tokens at start of each jet's slice in the sparse tensor
    #    Update ptr accordingly
    # 5. Build batch from updated ptr
    # 6. Return dict
```

---

### `experiments/top_tagging/dataset.py`

```python
EPS = 1e-5

class TopTaggingDataset(torch.utils.data.Dataset):
    """
    Loads minitoptagging .npz file.

    npz keys: kinematics_{mode} shape (N, 200, 4), labels_{mode} shape (N,)
    Produces torch_geometric Data objects with fields:
        x       : Tensor (n_real_particles, 4)  — zero-padded rows dropped
        label   : Tensor (1,) bool
    """

    def __init__(self, rescale_data: bool):
        ...

    def load_data(self, filename: str, mode: str, dtype=torch.float32):
        """mode in {"train", "val", "test"}"""
        # Load npz, iterate jets, drop zero-padded (|p|>EPS on all 4 components),
        # build Data(x=fourmomenta, label=label)

    def __len__(self): ...
    def __getitem__(self, idx): ...
```

Note: rescaling done in `embed_tagging_data`, not here (matches L-GATr pattern).

---

### `experiments/top_tagging/wrapper.py`

```python
def build_attention_mask(batch: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Block-diagonal additive attention mask.

    Parameters
    ----------
    batch : Tensor (n_total,) — jet index per token

    Returns
    -------
    mask : Tensor (n_total, n_total)  — 0.0 within-jet, -inf cross-jet
    """

class TopTaggingWrapper(nn.Module):
    """
    GATr wrapper for binary jet classification.

    Architecture:
        embed → GATr → extract_scalar → mean pool → linear(out_s → 1) → logit

    Parameters
    ----------
    net : GATr — configured with in_mv=1, out_mv=1, in_s=None, out_s=1
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, embedding: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        embedding : dict from embed_tagging_data

        Returns
        -------
        logits : Tensor (batchsize, 1)
        """
        mv = embedding["mv"].unsqueeze(0)    # (1, n_total, 1, 8)
        mask = build_attention_mask(embedding["batch"], mv.dtype)  # (n_total, n_total)

        mv_out, s_out = self.net(mv, scalars=None, attention_mask=mask)
        # mv_out: (1, n_total, 1, 8)
        # Extract grade-0 from mv output: (n_total, 1)
        scalars_out = mv_out[0, :, :, 0]           # (n_total, out_mv_channels=1)

        # Mean pool per jet (all tokens including beam)
        logits = scatter(scalars_out, index=embedding["batch"], dim=0, reduce="mean")
        return logits   # (batchsize, 1)
```

Note: `out_s_channels=None` in GATr (no scalar stream). Classification signal read from grade-0 of output MV.

---

### `experiments/top_tagging/experiment.py`

```python
@hydra.main(config_path="../../config", config_name="top_tagging", version_base=None)
def main(cfg: DictConfig) -> None:
    """Training + evaluation entry point."""
    # 1. Build datasets (TopTaggingDataset)
    # 2. Build GATr + TopTaggingWrapper
    # 3. Training loop:
    #    - DataLoader (torch_geometric)
    #    - embed_tagging_data per batch
    #    - BCE loss (BCEWithLogitsLoss)
    #    - Adam optimizer
    #    - log train loss every N steps
    #    - validate every cfg.training.validate_every_n_steps
    # 4. Eval: AUC (sklearn.metrics.roc_auc_score) + accuracy
    # 5. Save best model checkpoint
```

Callable from: `python -m experiments.top_tagging.experiment`

---

### `config/top_tagging.yaml`

```yaml
data:
  path: data/toptagging_mini.npz
  rescale_data: true
  two_beams: true
  add_scalar_features: false

model:
  hidden_mv_channels: 4
  hidden_s_channels: null     # no scalar stream
  num_blocks: 4
  num_heads: 4
  num_hidden_mv: 16           # attention hidden_mv_channels per head

training:
  epochs: 10
  batch_size: 32
  lr: 1e-3
  validate_every_n_steps: 200

evaluation:
  batch_size: 64
```

---

## Data Flow

```
.npz
  └─ TopTaggingDataset                 Data(x=(n_real,4), label=bool)
       └─ torch_geometric DataLoader   batches of variable-length jets
            └─ embed_tagging_data      mv=(n_total,1,8), batch=(n_total,), is_beam=(n_total,)
                 └─ TopTaggingWrapper  build_attention_mask → GATr → scatter mean → logit
                      └─ BCEWithLogitsLoss
```

## Equivariance

- `embed_tagging_data` — not required to be equivariant (it's the input embedding)
- `TopTaggingWrapper.forward` — equivariant in mv stream; scalar output (grade-0) is Spin(3)-invariant
- Beam spurion tokens break Spin(3)→SO(2) by design — this is intentional

## Test Plan

Add to `tests/test_top_tagging.py`:
- `test_embed_spurions_shape` — shape (2,1,8), correct ±z values
- `test_embed_tagging_data_shape` — beam tokens prepended, ptr updated
- `test_dataset_load` — drops zero-padded rows, correct labels
- `test_wrapper_output_shape` — output (batchsize, 1)
- `test_attention_mask_block_diagonal` — within-jet=0, cross-jet=-inf

No equivariance test for the full wrapper — embedding breaks equivariance by design (beam spurion).
