from typing import Protocol

import numpy as np


class StreamingConverter(Protocol):
    """One instance per session. Blocks are float32 mono, shape (n,), 48 kHz.

    process_block returns an array of the same shape, aligned in time with the
    input shifted by `latency_samples` of algorithmic delay.
    """

    latency_samples: int

    def process_block(self, block: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class PassthroughConverter:
    latency_samples = 0

    def process_block(self, block: np.ndarray) -> np.ndarray:
        if block.ndim != 1 or block.dtype != np.float32:
            raise ValueError("block must be float32 with shape (n,)")
        return block.copy()

    def reset(self) -> None:
        pass
