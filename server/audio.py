import numpy as np


def pcm16_to_mono_f32(data, num_channels: int) -> np.ndarray:
    if num_channels < 1:
        raise ValueError("num_channels must be >= 1")
    x = np.frombuffer(data, dtype=np.int16)  # raises ValueError on odd byte length
    if x.size % num_channels:
        raise ValueError("sample count is not a multiple of channel count")
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    x = x.reshape(-1, num_channels).astype(np.float32) / 32768.0
    return x.mean(axis=1, dtype=np.float32)


def f32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
