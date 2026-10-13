import time

import numpy as np

from engine.converter import StreamingConverter
from server import metrics
from server.audio import f32_to_pcm16_bytes, pcm16_to_mono_f32


class FrameProcessor:
    """Pure per-frame logic: PCM16 in, converted mono PCM16 out. No LiveKit imports."""

    def __init__(self, converter: StreamingConverter):
        self._converter = converter
        self.last_inference_ms = 0.0

    def process(self, data, num_channels: int) -> bytes:
        x = pcm16_to_mono_f32(data, num_channels)
        if x.size == 0:
            return b""
        t0 = time.perf_counter()
        y = self._converter.process_block(x)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        metrics.INFERENCE_MS.observe(self.last_inference_ms)
        y = np.asarray(y)
        if y.shape != x.shape:
            raise ValueError(f"converter returned shape {y.shape}, expected {x.shape}")
        return f32_to_pcm16_bytes(y.astype(np.float32, copy=False))
