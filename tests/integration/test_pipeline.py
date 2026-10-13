import asyncio
import os

import httpx
import pytest

from tools.smoke_client import run_smoke

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("VC_INTEGRATION") != "1", reason="set VC_INTEGRATION=1"),
]

TOKEN_URL = os.environ.get("TOKEN_URL", "http://localhost:8000")
WORKER_URL = os.environ.get("WORKER_URL", "http://127.0.0.1:8100")
API_KEY = os.environ.get("APP_API_KEY", "dev-app-key")


async def test_audio_round_trips_and_slot_is_released_after_leaving():
    async with httpx.AsyncClient() as http:
        before = (await http.get(f"{WORKER_URL}/health")).json()["active"]
        res = await run_smoke(TOKEN_URL, API_KEY, "demo", seconds=10.0)
        assert len(res) >= 3, "converted audio did not come back"
        res.sort()
        assert res[len(res) // 2] < 800, f"median round trip too high: {res}"
        for _ in range(20):  # the client has left mid-stream; the worker must free the slot
            if (await http.get(f"{WORKER_URL}/health")).json()["active"] == before:
                break
            await asyncio.sleep(0.25)
        assert (await http.get(f"{WORKER_URL}/health")).json()["active"] == before


async def test_abandoned_session_releases_its_slot():
    # requires the worker to run with JOIN_TIMEOUT_S=3
    async with httpx.AsyncClient() as http:
        before = (await http.get(f"{WORKER_URL}/health")).json()["active"]
        r = await http.post(f"{TOKEN_URL}/token", headers={"X-API-Key": API_KEY},
                            json={"voice_id": "demo"})
        assert r.status_code == 200
        mid = (await http.get(f"{WORKER_URL}/health")).json()["active"]
        assert mid == before + 1
        await asyncio.sleep(6)
        after = (await http.get(f"{WORKER_URL}/health")).json()["active"]
        assert after == before
