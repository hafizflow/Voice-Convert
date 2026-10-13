from pathlib import Path

import pytest

from server.config import load_config

BASE = {
    "LIVEKIT_URL": "ws://localhost:7880",
    "LIVEKIT_API_KEY": "k",
    "LIVEKIT_API_SECRET": "s",
    "APP_API_KEY": "app",
    "INTERNAL_SECRET": "int",
    "WORKER_URL": "http://127.0.0.1:8100",
}


def test_defaults():
    c = load_config(BASE)
    assert c.livekit_internal_url == "ws://localhost:7880"
    assert c.voices_dir == Path("voices")
    assert c.max_sessions == 4
    assert c.join_timeout_s == 30.0


def test_overrides():
    c = load_config({**BASE, "LIVEKIT_INTERNAL_URL": "ws://lk:7880", "MAX_SESSIONS": "9",
                     "JOIN_TIMEOUT_S": "2.5", "VOICES_DIR": "/v"})
    assert (c.livekit_internal_url, c.max_sessions, c.join_timeout_s, c.voices_dir) == (
        "ws://lk:7880", 9, 2.5, Path("/v"))


@pytest.mark.parametrize("missing", list(BASE))
def test_missing_required_var_raises(missing):
    env = {k: v for k, v in BASE.items() if k != missing}
    with pytest.raises(RuntimeError, match=missing):
        load_config(env)


def test_max_sessions_must_be_positive():
    with pytest.raises(RuntimeError, match="MAX_SESSIONS"):
        load_config({**BASE, "MAX_SESSIONS": "0"})
