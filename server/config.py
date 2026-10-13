import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


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


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env

    def req(name: str) -> str:
        v = env.get(name, "")
        if not v:
            raise RuntimeError(f"missing required env var {name}")
        return v

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
        voices_dir=Path(env.get("VOICES_DIR", "voices")),
        max_sessions=max_sessions,
        join_timeout_s=float(env.get("JOIN_TIMEOUT_S", "30")),
    )
