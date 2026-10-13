"""Offline streaming simulator: processes a file exactly like the live worker does.

The input is cut into fixed frames (default 20 ms at 48 kHz), fed through a
StreamingConverter one frame at a time, and the result is written aligned to
the input timeline. Prints per-block inference time, real-time factor and
algorithmic latency.

    python simulate.py IN.mp3 OUT.wav --model voices/qari012/model.pth --index voices/qari012/added.index
    python simulate.py IN.wav OUT.wav --passthrough
"""
import argparse
import time
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 48000


@dataclass
class SimResult:
    output: np.ndarray
    algorithmic_latency_ms: float
    blocks: int
    block_ms: list
    total_ms: float
    audio_s: float

    @property
    def rtf(self) -> float:
        return (self.total_ms / 1000.0) / self.audio_s if self.audio_s else 0.0


def simulate(x: np.ndarray, converter, frame_ms: int = 20) -> SimResult:
    frame = SAMPLE_RATE * frame_ms // 1000
    lat = int(getattr(converter, "latency_samples", 0))
    padded = np.concatenate([x, np.zeros(lat, dtype=np.float32)])
    outs, block_ms = [], []
    seen_blocks = getattr(converter, "blocks", 0)
    t_all = time.perf_counter()
    for i in range(0, len(padded), frame):
        outs.append(converter.process_block(padded[i:i + frame]))
        blocks = getattr(converter, "blocks", None)
        if blocks is not None and blocks != seen_blocks:
            block_ms.append(float(getattr(converter, "last_block_ms", 0.0)))
            seen_blocks = blocks
    total_ms = (time.perf_counter() - t_all) * 1000.0
    y = np.concatenate(outs)[lat:lat + len(x)]
    return SimResult(y.astype(np.float32), lat / (SAMPLE_RATE / 1000.0), len(block_ms),
                     block_ms, total_ms, len(x) / SAMPLE_RATE)


def boundary_step_ratio(y: np.ndarray, block: int) -> float:
    """Mean |sample step| at block boundaries over mean |step| elsewhere. ~1 means no clicks."""
    d = np.abs(np.diff(y))
    idx = np.arange(block - 1, len(d), block)
    mask = np.zeros(len(d), dtype=bool)
    mask[idx] = True
    other = d[~mask].mean()
    return float(d[mask].mean() / other) if other > 0 and mask.any() else 0.0


def load_48k_mono(path: str) -> np.ndarray:
    import soundfile as sf
    import soxr

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE)
    return mono.astype(np.float32)


def main() -> None:
    import soundfile as sf

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model")
    p.add_argument("--index")
    p.add_argument("--passthrough", action="store_true")
    p.add_argument("--block-ms", type=int, default=100)
    p.add_argument("--extra-s", type=float, default=0.5)
    p.add_argument("--crossfade-s", type=float, default=0.05)
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--seconds", type=float, default=0, help="only process the first N seconds")
    a = p.parse_args()

    x = load_48k_mono(a.input)
    if a.seconds:
        x = x[: int(a.seconds * SAMPLE_RATE)]
    if a.passthrough:
        from engine.converter import PassthroughConverter

        conv = PassthroughConverter()
    else:
        from engine.rvc import RVCConverter

        conv = RVCConverter(a.model, a.index, block_ms=a.block_ms, extra_s=a.extra_s,
                            crossfade_s=a.crossfade_s, index_rate=a.index_rate,
                            protect=a.protect, pitch_shift=a.pitch)
    res = simulate(x, conv)
    sf.write(a.output, res.output, SAMPLE_RATE)
    bm = np.array(res.block_ms) if res.block_ms else np.zeros(1)
    print(f"audio {res.audio_s:.1f}s | blocks {res.blocks} | per-block ms: "
          f"mean {bm.mean():.1f} p95 {np.percentile(bm, 95):.1f} max {bm.max():.1f}")
    print(f"real-time factor {res.rtf:.2f} (must be < 1.0) | algorithmic latency "
          f"{res.algorithmic_latency_ms:.0f} ms")
    print(f"boundary step ratio {boundary_step_ratio(res.output, a.block_ms * 48):.2f} (~1 = no clicks)")


if __name__ == "__main__":
    main()
