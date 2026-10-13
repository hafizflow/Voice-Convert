import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.registry import Voice
from server.config import Config
from server.worker_app import create_app

CFG = Config("ws://x", "ws://x", "k", "s", "app", "int-secret", "http://w", Path("v"), 1, 5.0)
VOICES = {"demo": Voice("demo", "Demo", 120.0, "test", None, None)}
H = {"X-Internal-Secret": "int-secret"}


class FakeSession:
    def __init__(self, cfg, room, voice, converter, **kw):
        self.room = room

    async def run(self):
        await asyncio.Event().wait()


@pytest.fixture
def client():
    with TestClient(create_app(CFG, VOICES, session_factory=FakeSession)) as c:
        yield c


def test_wrong_secret_is_401(client):
    r = client.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"},
                    headers={"X-Internal-Secret": "nope"})
    assert r.status_code == 401


def test_unknown_voice_is_404(client):
    r = client.post("/internal/sessions", json={"room": "r1", "voice_id": "ghost"}, headers=H)
    assert r.status_code == 404


def test_create_then_health_shows_active(client):
    r = client.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
    assert r.status_code == 201
    assert client.get("/health").json() == {"status": "ok", "active": 1, "max": 1}


def test_capacity_is_503(client):
    client.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
    r = client.post("/internal/sessions", json={"room": "r2", "voice_id": "demo"}, headers=H)
    assert r.status_code == 503


def test_duplicate_room_is_409():
    cfg = Config("ws://x", "ws://x", "k", "s", "app", "int-secret", "http://w", Path("v"), 5, 5.0)
    with TestClient(create_app(cfg, VOICES, session_factory=FakeSession)) as c:
        c.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
        r = c.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
        assert r.status_code == 409


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200 and "vc_active_sessions" in r.text
