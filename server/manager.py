import asyncio
import logging
from typing import Awaitable, Callable

from server import metrics

log = logging.getLogger(__name__)


class AtCapacity(Exception):
    pass


class DuplicateRoom(Exception):
    pass


class SessionManager:
    def __init__(self, max_sessions: int):
        self._max = max_sessions
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def active(self) -> int:
        return len(self._tasks)

    @property
    def capacity(self) -> int:
        return self._max

    def start(self, room: str, run: Callable[[], Awaitable[None]]) -> None:
        if room in self._tasks:
            raise DuplicateRoom(room)
        if len(self._tasks) >= self._max:
            raise AtCapacity(f"{len(self._tasks)}/{self._max} sessions active")
        task = asyncio.get_running_loop().create_task(run(), name=f"session-{room}")
        self._tasks[room] = task
        metrics.ACTIVE.set(len(self._tasks))
        task.add_done_callback(lambda t, room=room: self._finished(room, t))

    def _finished(self, room: str, task: asyncio.Task) -> None:
        self._tasks.pop(room, None)
        metrics.ACTIVE.set(len(self._tasks))
        if not task.cancelled() and task.exception() is not None:
            log.error("session %s crashed", room, exc_info=task.exception())

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
