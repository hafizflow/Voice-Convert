import numpy as np
import pytest
from prometheus_client import REGISTRY

from engine.converter import PassthroughConverter
from server.session import FrameProcessor


def pcm(x) -> bytes:
    return (np.asarray(x, dtype=np.float32) * 32767).astype(np.int16).tobytes()


class Fixed:
    latency_samples = 0

    def __init__(self, fn):
        self.fn = fn

    def process_block(self, block):
        return self.fn(block)

    def reset(self):
        pass


def test_passthrough_preserves_audio():
    x = np.sin(np.linspace(0, 20, 960)).astype(np.float32) * 0.5
    out = FrameProcessor(PassthroughConverter()).process(pcm(x), 1)
    back = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767
    assert np.allclose(back, x, atol=1e-3)


def test_empty_frame_returns_empty_bytes():
    assert FrameProcessor(PassthroughConverter()).process(b"", 1) == b""


def test_stereo_frame_is_downmixed_to_mono_length():
    stereo = np.zeros(960 * 2, dtype=np.int16).tobytes()
    out = FrameProcessor(PassthroughConverter()).process(stereo, 2)
    assert len(out) == 960 * 2  # 960 mono int16 samples


def test_wrong_length_from_converter_raises():
    fp = FrameProcessor(Fixed(lambda b: b[:-1]))
    with pytest.raises(ValueError):
        fp.process(pcm(np.zeros(100)), 1)


def test_nan_output_becomes_silence_not_nan():
    fp = FrameProcessor(Fixed(lambda b: np.full_like(b, np.nan)))
    out = np.frombuffer(fp.process(pcm(np.ones(100) * 0.3), 1), dtype=np.int16)
    assert not out.any()


def test_records_inference_time():
    before = REGISTRY.get_sample_value("vc_inference_ms_count") or 0
    fp = FrameProcessor(PassthroughConverter())
    fp.process(pcm(np.zeros(480)), 1)
    assert fp.last_inference_ms >= 0
    assert REGISTRY.get_sample_value("vc_inference_ms_count") == before + 1
