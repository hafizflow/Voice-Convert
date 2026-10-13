import numpy as np
import pytest

from engine.rvc import RVCConverter

SR = 48000


class FakeChanger:
    def __init__(self, **kw):
        self.kw = kw
        self.calls = []

    def on_request(self, block, f0_up_key=0, index_rate=0.5, protect=0.5, **rest):
        self.calls.append({"n": len(block), "f0_up_key": f0_up_key,
                           "index_rate": index_rate, "protect": protect})
        return block * 0.5, 0.0, [0, 1.0, 0]


def make(**kw):
    holder = {}

    def factory(**fkw):
        holder["c"] = FakeChanger(**fkw)
        return holder["c"]

    conv = RVCConverter("m.pth", "m.index", changer_factory=factory, **kw)
    return conv, holder["c"]


def test_block_size_is_derived_from_block_ms():
    conv, changer = make(block_ms=100)
    assert changer.kw["block_frame"] == 4800
    assert changer.kw["model_path"].endswith("/m.pth") and changer.kw["index_path"].endswith("/m.index")


def test_latency_is_block_plus_crossfade():
    conv, _ = make(block_ms=100, crossfade_s=0.05)
    assert conv.latency_samples == 4800 + int(0.05 * SR)


def test_output_length_matches_input_for_small_frames():
    conv, changer = make(block_ms=100)
    total_in = total_out = 0
    for _ in range(50):
        x = np.ones(480, np.float32)
        y = conv.process_block(x)
        assert y.shape == x.shape and y.dtype == np.float32
        total_in += len(x)
        total_out += len(y)
    assert total_in == total_out
    assert all(c["n"] == 4800 for c in changer.calls) and len(changer.calls) == 5


def test_parameters_are_forwarded():
    conv, changer = make(block_ms=20, index_rate=0.7, protect=0.2, pitch_shift=-3)
    conv.process_block(np.zeros(960, np.float32))
    c = changer.calls[0]
    assert (c["f0_up_key"], c["index_rate"], c["protect"]) == (-3, 0.7, 0.2)


def test_pitch_shift_can_change_between_blocks():
    conv, changer = make(block_ms=20)
    conv.process_block(np.zeros(960, np.float32))
    conv.pitch_shift = 5
    conv.process_block(np.zeros(960, np.float32))
    assert [c["f0_up_key"] for c in changer.calls] == [0, 5]


def test_rejects_non_float32_1d_blocks():
    conv, _ = make()
    with pytest.raises(ValueError):
        conv.process_block(np.zeros((2, 10), np.float32))
    with pytest.raises(ValueError):
        conv.process_block(np.zeros(10, np.float64))


def test_converted_audio_comes_out_after_one_block_of_delay():
    conv, _ = make(block_ms=20)  # 960
    x = np.ones(960 * 3, np.float32)
    y = conv.process_block(x)
    assert not y[:960].any()
    assert np.allclose(y[960:], 0.5)


def test_model_and_index_paths_are_made_absolute_before_the_factory_runs():
    seen = {}

    def factory(**kw):
        seen.update(kw)
        return FakeChanger(**kw)

    RVCConverter("relative/model.pth", "relative/m.index", changer_factory=factory)
    from pathlib import Path
    assert Path(seen["model_path"]).is_absolute() and Path(seen["index_path"]).is_absolute()


def test_silence_gate_defaults_to_minus_60_db_not_zero():
    # Applio's 0 dB threshold gates every block of normal audio into silence.
    _, changer = make()
    assert changer.kw["silent_threshold"] == -60


def test_changer_is_warmed_up_once_at_construction_when_it_supports_it():
    calls = []

    class Warm(FakeChanger):
        def warmup(self, **kw):
            calls.append(kw)

    RVCConverter("m", None, changer_factory=lambda **kw: Warm(**kw), index_rate=0.6)
    assert len(calls) == 1 and calls[0]["index_rate"] == 0.6


def test_relative_paths_resolve_against_startup_dir_even_after_a_chdir(tmp_path, monkeypatch):
    import engine.rvc as rvc_mod

    monkeypatch.setattr(rvc_mod, "_START_CWD", tmp_path)
    seen = {}

    def factory(**kw):
        seen.update(kw)
        return FakeChanger(**kw)

    monkeypatch.chdir("/")  # simulates Applio having chdir'd away
    RVCConverter("voices/q/model.pth", "voices/q/added.index", changer_factory=factory)
    assert seen["model_path"] == str(tmp_path / "voices/q/model.pth")
    assert seen["index_path"] == str(tmp_path / "voices/q/added.index")
