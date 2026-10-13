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
    assert changer.kw["model_path"] == "m.pth" and changer.kw["index_path"] == "m.index"


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
