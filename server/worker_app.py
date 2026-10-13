import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from engine.factory import default_converter_factory
from engine.registry import load_registry
from server.config import Config, load_config
from server.manager import AtCapacity, DuplicateRoom, SessionManager
from server.session import ConversionSession

log = logging.getLogger(__name__)


class SessionRequest(BaseModel):
    room: str
    voice_id: str


def create_app(cfg: Config, registry, session_factory=ConversionSession,
               converter_factory=None) -> FastAPI:
    manager = SessionManager(cfg.max_sessions)
    converter_factory = converter_factory or default_converter_factory(cfg)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await manager.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.post("/internal/sessions", status_code=201)
    async def create_session(body: SessionRequest, x_internal_secret: str = Header(default="")):
        if not hmac.compare_digest(x_internal_secret, cfg.internal_secret):
            raise HTTPException(401, "bad internal secret")
        voice = registry.get(body.voice_id)
        if voice is None:
            raise HTTPException(404, "unknown voice")
        # the converter is built inside the session, on its own thread, after the capacity check
        session = session_factory(cfg, body.room, voice, None, converter_factory=converter_factory)
        try:
            manager.start(body.room, session.run)
        except AtCapacity:
            raise HTTPException(503, "worker at capacity")
        except DuplicateRoom:
            raise HTTPException(409, "room already has a session")
        return {"room": body.room}

    @app.get("/health")
    async def health():
        return {"status": "ok", "active": manager.active, "max": manager.capacity}

    app.mount("/metrics", make_asgi_app())
    return app


def build_app() -> FastAPI:
    cfg = load_config()
    registry = load_registry(cfg.voices_dir)
    log.info("loaded voices: %s", sorted(registry))
    return create_app(cfg, registry)
