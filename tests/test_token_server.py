from pathlib import Path

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from engine.registry import Voice
from server.config import Config
from server.token_server import WorkerFull, create_app

CFG = Config("ws://lk:7880", "ws://lk:7880", "devkey", "devsecret-devsecret-devsecret-1234",
             "app-key", "int", "http://w", Path("v"), 4, 30.0)
VOICES = {"demo": Voice("demo", "Demo", 120.0, "secret-consent-text", None, None)}
KEY = {"X-API-Key": "app-key"}


class FakeWorker:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def start_session(self, room, voice_id):
        if self.error:
            raise self.error
        self.calls.append((room, voice_id))


def client(worker=None):
    return TestClient(create_app(CFG, VOICES, worker or FakeWorker()))


def test_voices_lists_id_and_name_only():
    r = client().get("/voices")
    assert r.json() == [{"id": "demo", "name": "Demo"}]


def test_token_requires_api_key():
    r = client().post("/token", json={"voice_id": "demo"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_unknown_voice_is_404_and_no_session_started():
    w = FakeWorker()
    r = client(w).post("/token", json={"voice_id": "ghost"}, headers=KEY)
    assert r.status_code == 404 and w.calls == []


def test_token_is_scoped_to_one_private_room():
    w = FakeWorker()
    body = client(w).post("/token", json={"voice_id": "demo"}, headers=KEY).json()
    claims = jwt.decode(body["token"], CFG.livekit_api_secret, algorithms=["HS256"],
                        options={"verify_aud": False})
    assert claims["iss"] == "devkey"
    assert claims["sub"] == body["identity"]
    assert claims["video"]["room"] == body["room"]
    assert claims["video"]["roomJoin"] is True
    assert claims["exp"] - claims["nbf"] <= 900
    assert body["url"] == "ws://lk:7880"
    assert w.calls == [(body["room"], "demo")]


def test_each_request_gets_a_new_room():
    c = client()
    a = c.post("/token", json={"voice_id": "demo"}, headers=KEY).json()
    b = c.post("/token", json={"voice_id": "demo"}, headers=KEY).json()
    assert a["room"] != b["room"] and a["identity"] != b["identity"]


def test_worker_full_is_503():
    r = client(FakeWorker(WorkerFull())).post("/token", json={"voice_id": "demo"}, headers=KEY)
    assert r.status_code == 503


def test_worker_unreachable_is_502():
    w = FakeWorker(httpx.ConnectError("down"))
    r = client(w).post("/token", json={"voice_id": "demo"}, headers=KEY)
    assert r.status_code == 502


def test_index_is_served_with_label():
    r = client().get("/")
    assert r.status_code == 200 and "AI practice voice" in r.text
