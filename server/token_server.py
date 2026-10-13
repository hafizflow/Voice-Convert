import hmac
import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from livekit import api
from pydantic import BaseModel

from engine.registry import load_registry
from server.config import Config, load_config

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class WorkerFull(Exception):
    pass


class HttpWorkerClient:
    def __init__(self, base_url: str, secret: str, transport: httpx.AsyncBaseTransport | None = None):
        self._base = base_url.rstrip("/")
        self._secret = secret
        self._transport = transport

    async def start_session(self, room: str, voice_id: str) -> None:
        async with httpx.AsyncClient(timeout=5.0, transport=self._transport) as http:
            r = await http.post(
                f"{self._base}/internal/sessions",
                json={"room": room, "voice_id": voice_id},
                headers={"X-Internal-Secret": self._secret},
            )
        if r.status_code == 503:
            raise WorkerFull()
        r.raise_for_status()


class TokenRequest(BaseModel):
    voice_id: str


def create_app(cfg: Config, registry, worker_client) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/voices")
    async def voices():
        return [{"id": v.id, "name": v.name} for v in registry.values()]

    @app.post("/token")
    async def token(body: TokenRequest, x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key, cfg.app_api_key):
            raise HTTPException(401, "bad api key")
        if body.voice_id not in registry:
            raise HTTPException(404, "unknown voice")
        room = f"vc-{uuid.uuid4().hex[:12]}"
        identity = f"user-{uuid.uuid4().hex[:8]}"
        try:
            await worker_client.start_session(room, body.voice_id)
        except WorkerFull:
            raise HTTPException(503, "all conversion slots are busy, try again shortly")
        except httpx.HTTPError:
            log.exception("worker unavailable")
            raise HTTPException(502, "conversion worker unavailable")
        jwt_token = (
            api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_metadata(json.dumps({"voice_id": body.voice_id}))
            .with_grants(api.VideoGrants(
                room_join=True, room=room,
                can_publish=True, can_subscribe=True, can_publish_data=False,
                can_publish_sources=["microphone"],
            ))
            .with_ttl(timedelta(minutes=15))
            .to_jwt()
        )
        return {"url": cfg.livekit_url, "token": jwt_token, "room": room,
                "identity": identity, "voice_id": body.voice_id}

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    return app


def build_app() -> FastAPI:
    cfg = load_config()
    registry = load_registry(cfg.voices_dir)
    return create_app(cfg, registry, HttpWorkerClient(cfg.worker_url, cfg.internal_secret))
