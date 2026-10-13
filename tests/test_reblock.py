import numpy as np
import pytest

from engine.reblock import Reblocker


def test_output_length_always_equals_input_length():
    rng = np.random.default_rng(0)
    rb = Reblocker(480, lambda b: b)
    for n in [0, 1, 10, 479, 480, 481, 960, 1500, 3, 0, 7]:
        assert rb.push(rng.standard_normal(n).astype(np.float32)).shape == (n,)


def test_output_is_input_delayed_by_exactly_one_block():
    B = 480
    x = np.arange(5000, dtype=np.float32) + 1
    rb = Reblocker(B, lambda b: b)
    chunks = [x[i:i + 160] for i in range(0, len(x), 160)]
    out = np.concatenate([rb.push(c) for c in chunks])
    assert np.array_equal(out[:B], np.zeros(B, np.float32))
    assert np.array_equal(out[B:], x[:len(x) - B])


def test_fn_always_receives_exactly_block_size():
    seen = []

    def fn(b):
        seen.append(len(b))
        return b

    rb = Reblocker(256, fn)
    rng = np.random.default_rng(1)
    for n in [100, 700, 1, 255, 1000]:
        rb.push(rng.standard_normal(n).astype(np.float32))
    assert seen and set(seen) == {256}


def test_irregular_chunking_matches_single_push():
    x = np.random.default_rng(2).standard_normal(4000).astype(np.float32)
    a = Reblocker(300, lambda b: b * 2)
    whole = a.push(x)
    b = Reblocker(300, lambda b: b * 2)
    parts = np.concatenate([b.push(x[i:i + 37]) for i in range(0, len(x), 37)])
    assert np.array_equal(whole, parts)


def test_fn_returning_wrong_length_raises():
    rb = Reblocker(100, lambda b: b[:-1])
    with pytest.raises(ValueError):
        rb.push(np.zeros(100, np.float32))


def test_reset_clears_state():
    rb = Reblocker(100, lambda b: b)
    rb.push(np.ones(150, np.float32))
    rb.reset()
    out = rb.push(np.ones(100, np.float32))
    assert not out.any()  # back to the zero prefill
