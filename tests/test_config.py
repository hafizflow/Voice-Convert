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
    assert c.voices_dir == Path("voices").resolve()  # absolute: Applio chdirs
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


PROD = {**BASE, "VC_ENV": "production",
        "LIVEKIT_API_KEY": "APIk3yUnique", "LIVEKIT_API_SECRET": "x" * 40,
        "APP_API_KEY": "a-long-random-app-key", "INTERNAL_SECRET": "a-long-random-internal"}


def test_production_accepts_strong_secrets():
    assert load_config(PROD).livekit_api_key == "APIk3yUnique"


@pytest.mark.parametrize("name,weak", [
    ("LIVEKIT_API_KEY", "devkey"), ("LIVEKIT_API_SECRET", "secret"),
    ("LIVEKIT_API_SECRET", "short-but-not-default"),
    ("APP_API_KEY", "change-me"), ("INTERNAL_SECRET", "change-me-too"),
])
def test_production_refuses_weak_or_default_secrets(name, weak):
    with pytest.raises(RuntimeError, match=name):
        load_config({**PROD, name: weak})


def test_dev_mode_allows_dev_defaults():
    assert load_config({**BASE, "LIVEKIT_API_KEY": "devkey", "LIVEKIT_API_SECRET": "secret"})


def test_rvc_defaults_and_overrides():
    c = load_config(BASE)
    assert (c.rvc_block_ms, c.rvc_f0, c.rvc_device) == (200, "fcpe", "cpu")
    c = load_config({**BASE, "RVC_BLOCK_MS": "100", "RVC_EXTRA_S": "0.3", "RVC_F0": "rmvpe",
                     "RVC_DEVICE": "mps", "RVC_INDEX_RATE": "0.0"})
    assert (c.rvc_block_ms, c.rvc_extra_s, c.rvc_f0, c.rvc_device, c.rvc_index_rate) == (
        100, 0.3, "rmvpe", "mps", 0.0)
