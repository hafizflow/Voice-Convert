import numpy as np
import pytest

from engine.converter import PassthroughConverter


def test_passthrough_returns_equal_copy():
    c = PassthroughConverter()
    x = np.linspace(-1, 1, 480, dtype=np.float32)
    y = c.process_block(x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.array_equal(x, y)
    y[0] = 9.0
    assert x[0] != 9.0  # no aliasing


def test_passthrough_declares_zero_latency():
    assert PassthroughConverter().latency_samples == 0


@pytest.mark.parametrize("bad", [np.zeros((2, 10), np.float32), np.zeros(10, np.float64)])
def test_passthrough_rejects_wrong_shape_or_dtype(bad):
    with pytest.raises(ValueError):
        PassthroughConverter().process_block(bad)
