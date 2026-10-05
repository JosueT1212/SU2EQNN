"""A/B guard for the full-dataset sweep: every *_full_b12 config must match the
reference (rest_boosted, aligned to the L-GATr paper config) except for the
keys that define the variant under test."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG = Path(__file__).parent.parent / "config"
REFERENCE = "top_tagging_rest_frame_boosted_beams_full_b12"
# Keys allowed to differ between variants (the tested factor + bookkeeping).
VARIANT_KEYS = {
    "data.path", "data.embedding", "data.inject", "data.frame", "data.beams", "data.beam_reference",
    "model.in_mv_channels", "model.patch_mv_channels", "training.checkpoint_path",
}
# L-GATr paper values (lorentz-gatr/config_paper: default, default_tagging,
# toptagging, model/gatr_tagging) that the sweep reproduces.
LGATR = {
    "training.optimizer": "lion", "training.lr": 3e-4, "training.weight_decay": 0.2,
    "training.betas": [0.9, 0.999], "training.batch_size": 128,
    "training.scheduler": "cosine", "training.eta_min": 0.0, "training.clip_grad_norm": 5,
    "training.select_best_by": "loss", "training.validate_every_n_steps": 5000,
    "training.axis_dropout_p": 0.0,
    "model.num_blocks": 12, "model.num_heads": 8, "model.hidden_mv_channels": 16,
    "model.head_scale": True, "model.multi_query": False,
}


def flat(name):
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG / f"{name}.yaml"))
    return {f"{sec}.{k}": v for sec, d in cfg.items() for k, v in d.items()}


FULL = sorted(p.stem for p in CONFIG.glob("top_tagging_*_full_b12.yaml"))


def test_sweep_has_all_eighteen_versions():
    assert len(FULL) == 18


@pytest.mark.parametrize("name", FULL)
def test_config_differs_from_reference_only_in_variant_keys(name):
    ref, cfg = flat(REFERENCE), flat(name)
    differing = {k for k in ref.keys() | cfg.keys() if ref.get(k) != cfg.get(k)}
    assert differing <= VARIANT_KEYS, sorted(differing - VARIANT_KEYS)


@pytest.mark.parametrize("key,value", sorted(LGATR.items()))
def test_reference_matches_lgatr(key, value):
    assert flat(REFERENCE)[key] == value


def test_multi_head_model_size():
    from types import SimpleNamespace
    from experiments.top_tagging.experiment import build_model
    m = SimpleNamespace(model=SimpleNamespace(**{k.split(".")[1]: v for k, v in LGATR.items()
                                                if k.startswith("model.")}))
    assert sum(p.numel() for p in build_model(m).parameters()) == 445_425
