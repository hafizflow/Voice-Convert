import numpy as np
import pytest

from server.audio import f32_to_pcm16_bytes, pcm16_to_mono_f32


def pcm(values):
    return np.array(values, dtype=np.int16).tobytes()


def test_mono_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    out = pcm16_to_mono_f32(f32_to_pcm16_bytes(x), 1)
    assert np.allclose(out, x, atol=1e-3)


def test_stereo_is_downmixed_by_average():
    out = pcm16_to_mono_f32(pcm([16384, -16384, 16384, 16384]), 2)
    assert np.allclose(out, [0.0, 0.5], atol=1e-4)


def test_empty_frame_gives_empty_array():
    assert pcm16_to_mono_f32(b"", 1).size == 0


def test_odd_byte_length_raises_value_error():
    with pytest.raises(ValueError):
        pcm16_to_mono_f32(b"\x00\x01\x02", 1)


def test_sample_count_not_multiple_of_channels_raises():
    with pytest.raises(ValueError):
        pcm16_to_mono_f32(pcm([1, 2, 3]), 2)


def test_memoryview_is_accepted():
    assert pcm16_to_mono_f32(memoryview(pcm([100, 200])), 1).size == 2


def test_output_clips_and_removes_nan():
    x = np.array([2.0, -2.0, np.nan, np.inf], dtype=np.float32)
    out = np.frombuffer(f32_to_pcm16_bytes(x), dtype=np.int16)
    assert out[0] == 32767 and out[1] == -32767
    assert out[2] == 0
    assert out[3] == 32767
