import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_WEAK = {"devkey", "secret", "change-me", "change-me-too", "dev-app-key", "dev-internal"}


@dataclass(frozen=True)
class Config:
    livekit_url: str            # URL browsers connect to
    livekit_internal_url: str   # URL the worker connects to
    livekit_api_key: str
    livekit_api_secret: str
    app_api_key: str            # shared key the test page sends to /token
    internal_secret: str        # token server -> worker
    worker_url: str
    voices_dir: Path
    max_sessions: int
    join_timeout_s: float
    # RVC streaming. Defaults are the best config measured on an Apple M4 (see tools/bench_blocks.py).
    rvc_block_ms: int = 200
    rvc_extra_s: float = 0.15
    rvc_f0: str = "fcpe"
    rvc_device: str | None = "cpu"
    rvc_index_rate: float = 0.4
    rvc_pitch_shift: int = 0     # semitones applied to the speaker before conversion


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env

    def req(name: str) -> str:
        v = env.get(name, "")
        if not v:
            raise RuntimeError(f"missing required env var {name}")
        return v

    if env.get("VC_ENV") == "production":
        for name in ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "APP_API_KEY", "INTERNAL_SECRET"):
            if env.get(name, "") in _WEAK:
                raise RuntimeError(f"{name} is a known development default; refusing to start in production")
        if len(env.get("LIVEKIT_API_SECRET", "")) < 32:
            raise RuntimeError("LIVEKIT_API_SECRET must be at least 32 characters in production")
    max_sessions = int(env.get("MAX_SESSIONS", "4"))
    if max_sessions < 1:
        raise RuntimeError("MAX_SESSIONS must be >= 1")
    return Config(
        livekit_url=req("LIVEKIT_URL"),
        livekit_internal_url=env.get("LIVEKIT_INTERNAL_URL") or req("LIVEKIT_URL"),
        livekit_api_key=req("LIVEKIT_API_KEY"),
        livekit_api_secret=req("LIVEKIT_API_SECRET"),
        app_api_key=req("APP_API_KEY"),
        internal_secret=req("INTERNAL_SECRET"),
        worker_url=req("WORKER_URL"),
        voices_dir=Path(env.get("VOICES_DIR", "voices")).resolve(),
        max_sessions=max_sessions,
        join_timeout_s=float(env.get("JOIN_TIMEOUT_S", "30")),
        rvc_block_ms=int(env.get("RVC_BLOCK_MS", "200")),
        rvc_extra_s=float(env.get("RVC_EXTRA_S", "0.15")),
        rvc_f0=env.get("RVC_F0", "fcpe"),
        rvc_device=env.get("RVC_DEVICE", "cpu") or None,
        rvc_index_rate=float(env.get("RVC_INDEX_RATE", "0.4")),
        rvc_pitch_shift=int(env.get("RVC_PITCH_SHIFT", "0")),
    )



