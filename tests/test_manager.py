import asyncio

import pytest
from prometheus_client import REGISTRY

from server.manager import AtCapacity, DuplicateRoom, SessionManager


async def settle():
    await asyncio.sleep(0.02)


async def test_slot_released_when_session_finishes():
    m = SessionManager(1)
    ev = asyncio.Event()

    async def run():
        await ev.wait()

    m.start("a", run)
    assert m.active == 1
    ev.set()
    await settle()
    assert m.active == 0


async def test_slot_released_when_session_crashes():
    m = SessionManager(1)

    async def run():
        raise RuntimeError("boom")

    m.start("a", run)
    await settle()
    assert m.active == 0
    m.start("b", run)  # slot is reusable


async def test_capacity_enforced():
    m = SessionManager(1)

    async def run():
        await asyncio.Event().wait()

    m.start("a", run)
    with pytest.raises(AtCapacity):
        m.start("b", run)
    await m.shutdown()


async def test_duplicate_room_rejected():
    m = SessionManager(3)

    async def run():
        await asyncio.Event().wait()

    m.start("a", run)
    with pytest.raises(DuplicateRoom):
        m.start("a", run)
    await m.shutdown()


async def test_shutdown_cancels_everything():
    m = SessionManager(3)
    cancelled = []

    async def run():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    m.start("a", run)
    m.start("b", run)
    await asyncio.sleep(0)  # let both tasks start so cancellation lands inside run()
    await m.shutdown()
    assert len(cancelled) == 2 and m.active == 0


async def test_active_gauge_tracks_sessions():
    m = SessionManager(2)

    async def run():
        await asyncio.Event().wait()

    m.start("a", run)
    assert REGISTRY.get_sample_value("vc_active_sessions") == 1
    await m.shutdown()
    assert REGISTRY.get_sample_value("vc_active_sessions") == 0
