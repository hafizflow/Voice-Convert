import os
import sys
from pathlib import Path

import numpy as np

from engine.reblock import Reblocker

SAMPLE_RATE = 48000
APPLIO_DIR = Path(__file__).resolve().parent.parent / "third_party" / "applio"


def _applio_voice_changer(**kwargs):
    """Build Applio's VoiceChanger. Applio resolves rvc/models against the cwd."""
    if str(APPLIO_DIR) not in sys.path:
        sys.path.insert(0, str(APPLIO_DIR))
    os.chdir(APPLIO_DIR)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from rvc.realtime.core import VoiceChanger

    return VoiceChanger(**kwargs)


class RVCConverter:
    """StreamingConverter backed by Applio's realtime VoiceChanger (RVC v2).

    Accepts frames of any size at 48 kHz; Applio sees fixed `block_ms` blocks.
    Output is delayed by `latency_samples`.
    """

    def __init__(self, model_path: str, index_path: str | None = None, *,
                 block_ms: int = 100, extra_s: float = 0.5, crossfade_s: float = 0.05,
                 index_rate: float = 0.5, protect: float = 0.33, f0_method: str = "rmvpe",
                 embedder_model: str = "contentvec", pitch_shift: int = 0,
                 device: str | None = None, changer_factory=None):
        self._block = SAMPLE_RATE * block_ms // 1000
        self.index_rate = index_rate
        self.protect = protect
        self.pitch_shift = pitch_shift
        self.latency_samples = self._block + int(crossfade_s * SAMPLE_RATE)
        factory = changer_factory or _applio_voice_changer
        kwargs = dict(
            block_frame=self._block,
            cross_fade_overlap_size=crossfade_s,
            extra_convert_size=extra_s,
            model_path=str(model_path),
            index_path=str(index_path) if index_path else "",
            f0_method=f0_method,
            embedder_model=embedder_model,
        )
        if device:
            kwargs["device"] = device
        self._changer = factory(**kwargs)
        self._reblock = Reblocker(self._block, self._convert)
        self.last_block_ms = 0.0
        self.blocks = 0

    def _convert(self, block: np.ndarray) -> np.ndarray:
        out, _vol, perf = self._changer.on_request(
            block,
            f0_up_key=self.pitch_shift,
            index_rate=self.index_rate,
            protect=self.protect,
        )
        self.last_block_ms = float(perf[1])
        self.blocks += 1
        return np.asarray(out, dtype=np.float32)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        if block.ndim != 1 or block.dtype != np.float32:
            raise ValueError("block must be float32 with shape (n,)")
        return self._reblock.push(block)

    def reset(self) -> None:
        self._reblock.reset()
