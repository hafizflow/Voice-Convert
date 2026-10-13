from typing import Callable

import numpy as np


class Reblocker:
    """Turn arbitrary-size input chunks into fixed-size blocks for `fn`.

    `push(x)` always returns exactly `len(x)` samples. Output lags input by
    exactly `block_size` samples because the output queue starts with one
    block of silence; that is the price of never under-running.
    """

    def __init__(self, block_size: int, fn: Callable[[np.ndarray], np.ndarray]):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self._block = block_size
        self._fn = fn
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._ready = np.zeros(self._block, dtype=np.float32)

    def push(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n:
            self._pending = np.concatenate([self._pending, x.astype(np.float32, copy=False)])
        while len(self._pending) >= self._block:
            block, self._pending = self._pending[:self._block], self._pending[self._block:]
            out = np.asarray(self._fn(block), dtype=np.float32)
            if out.shape != (self._block,):
                raise ValueError(f"fn returned shape {out.shape}, expected ({self._block},)")
            self._ready = np.concatenate([self._ready, out])
        out, self._ready = self._ready[:n], self._ready[n:]
        return out
