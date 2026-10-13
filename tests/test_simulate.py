import numpy as np

from engine.converter import PassthroughConverter
from engine.rvc import RVCConverter
from simulate import boundary_step_ratio, simulate


def tone(seconds=2.0, hz=220.0, sr=48000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_passthrough_output_is_input_aligned_and_same_length():
    x = tone()
    res = simulate(x, PassthroughConverter(), frame_ms=20)
    assert res.output.shape == x.shape
    assert np.allclose(res.output, x, atol=1e-6)
    assert res.algorithmic_latency_ms == 0.0


def test_reports_realtime_factor_and_block_times():
    class FakeChanger:
        def __init__(self, **kw):
            pass

        def on_request(self, block, **kw):
            return block, 0.0, [0, 2.0, 0]

    conv = RVCConverter("m", None, block_ms=100, changer_factory=lambda **kw: FakeChanger(**kw))
    res = simulate(tone(3.0), conv, frame_ms=20)
    assert res.blocks == (3 * 48000 + conv.latency_samples) // 4800  # input + flush of the latency
    assert res.algorithmic_latency_ms == conv.latency_samples / 48.0
    assert res.rtf >= 0
    assert res.output.shape == tone(3.0).shape


def test_boundary_ratio_near_one_for_smooth_audio():
    assert 0.5 < boundary_step_ratio(tone(), block=4800) < 2.0


def test_boundary_ratio_detects_a_click_at_block_edges():
    y = tone()
    y[4800::4800] += 0.8  # click at every block boundary
    assert boundary_step_ratio(y, block=4800) > 5.0
