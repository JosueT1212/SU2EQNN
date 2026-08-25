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
