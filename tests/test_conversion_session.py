import asyncio
from pathlib import Path

import numpy as np
import pytest
from livekit import rtc
from prometheus_client import REGISTRY

from engine.converter import PassthroughConverter
from engine.registry import Voice
from server import session as session_mod
from server.config import Config
from server.session import ConversionSession, LiveKitIo

CFG = Config("ws://x", "ws://x", "devkey", "s" * 40, "app", "int", "http://w", Path("v"), 4, 0.2)
VOICE = Voice("demo", "Demo", 120.0, "test", None, None)


class FakeLocal:
    def __init__(self, fail_data=False):
        self.published, self.data_calls, self.fail_data = [], 0, fail_data

    async def publish_track(self, track, options):
        self.published.append(track)

    async def publish_data(self, payload, **kw):
        self.data_calls += 1
        if self.fail_data:
            raise RuntimeError("transient")


class FakeRoom:
    def __init__(self, participants=True, fail_data=False):
        self.handlers = {}
        self.remote_participants = {"user": object()} if participants else {}
        self.local_participant = FakeLocal(fail_data)
        self.disconnected = False

    def on(self, event):
        def deco(fn):
            self.handlers[event] = fn
            return fn
        return deco

    def emit(self, event, *args):
        self.handlers[event](*args)

    async def connect(self, url, token):
        pass

    async def disconnect(self):
        self.disconnected = True


class FakeSource:
    def __init__(self, fail=False):
        self.frames, self.fail = [], fail

    async def capture_frame(self, frame):
        if self.fail:
            raise RuntimeError("capture failed")
        self.frames.append(bytes(frame.data))


class Ev:
    def __init__(self, data):
        n = len(data) // 2
        self.frame = type("F", (), {"data": data, "num_channels": 1, "samples_per_channel": n})()


class FakeStream:
    def __init__(self, frames):
        self.frames, self.closed = frames, False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for f in self.frames:
            yield Ev(f)
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class FakeIo:
    def __init__(self, room, source, streams):
        self.room, self.source, self.streams = room, source, streams

    def new_room(self):
        return self.room

    def new_source(self):
        return self.source

    def new_output_track(self, source):
        return object()

    def new_stream(self, track):
        return self.streams[track]


def audio_track():
    return type("T", (), {"kind": rtc.TrackKind.KIND_AUDIO})()


def pcm(n=480, v=0.25):
    return (np.full(n, v) * 32767).astype(np.int16).tobytes()


def make(converter=None, room=None, source=None, streams=None):
    room = room or FakeRoom()
    source = source or FakeSource()
    io = FakeIo(room, source, streams or {})
    s = ConversionSession(CFG, "r1", VOICE, converter or PassthroughConverter(), io=io)
    return s, room, source


async def run_bg(s):
    return asyncio.create_task(s.run())


async def test_user_never_joins_closes_after_timeout():
    s, room, _ = make(room=FakeRoom(participants=False))
    await asyncio.wait_for(s.run(), 2)
    assert room.disconnected


async def test_user_leaving_ends_session_and_disconnects():
    s, room, _ = make()
    t = await run_bg(s)
    await asyncio.sleep(0.05)
    room.emit("participant_disconnected", object())
    await asyncio.wait_for(t, 2)
    assert room.disconnected


async def test_connection_drop_ends_session():
    s, room, _ = make()
    t = await run_bg(s)
    await asyncio.sleep(0.05)
    room.emit("disconnected")
    await asyncio.wait_for(t, 2)
    assert room.disconnected


async def test_audio_is_converted_and_published():
    trk = audio_track()
    stream = FakeStream([pcm(480, 0.25), pcm(480, -0.25)])
    s, room, source = make(streams={trk: stream})
    t = await run_bg(s)
    await asyncio.sleep(0.05)
    room.emit("track_subscribed", trk, None, None)
    await asyncio.sleep(0.1)
    assert len(source.frames) == 2 and len(source.frames[0]) == 960
    room.emit("participant_disconnected", object())
    await asyncio.wait_for(t, 2)
    assert stream.closed


async def test_converter_failure_sends_silence_and_counts_error():
    class Boom:
        latency_samples = 0

        def process_block(self, b):
            raise RuntimeError("model exploded")

        def reset(self):
            pass

    before = REGISTRY.get_sample_value("vc_frame_errors_total") or 0
    trk = audio_track()
    s, room, source = make(converter=Boom(), streams={trk: FakeStream([pcm(480)])})
    t = await run_bg(s)
    await asyncio.sleep(0.05)
    room.emit("track_subscribed", trk, None, None)
    await asyncio.sleep(0.1)
    assert source.frames == [bytes(960)]  # silence, never the unconverted voice
    assert REGISTRY.get_sample_value("vc_frame_errors_total") == before + 1
    room.emit("participant_disconnected", object())
    await asyncio.wait_for(t, 2)


async def test_pump_crash_ends_session_instead_of_hanging():
    trk = audio_track()
    s, room, _ = make(source=FakeSource(fail=True), streams={trk: FakeStream([pcm(480)])})
    t = await run_bg(s)
    await asyncio.sleep(0.05)
    room.emit("track_subscribed", trk, None, None)
    await asyncio.wait_for(t, 2)  # would time out if the failure were swallowed
    assert room.disconnected


async def test_second_audio_track_replaces_first_pump():
    t1, t2 = audio_track(), audio_track()
    s1, s2 = FakeStream([]), FakeStream([])
    s, room, _ = make(streams={t1: s1, t2: s2})
    t = await run_bg(s)
    await asyncio.sleep(0.05)
    room.emit("track_subscribed", t1, None, None)
    await asyncio.sleep(0.05)
    room.emit("track_subscribed", t2, None, None)
    await asyncio.sleep(0.05)
    assert s1.closed and not s2.closed  # only one pump feeds the output at a time
    room.emit("participant_disconnected", object())
    await asyncio.wait_for(t, 2)


async def test_stats_publish_failure_does_not_kill_session(monkeypatch):
    monkeypatch.setattr(session_mod, "STATS_INTERVAL_S", 0.01)
    s, room, _ = make(room=FakeRoom(fail_data=True))
    t = await run_bg(s)
    await asyncio.sleep(0.15)
    assert not t.done() and room.local_participant.data_calls >= 2
    room.emit("participant_disconnected", object())
    await asyncio.wait_for(t, 2)


def test_input_stream_is_bounded(monkeypatch):
    seen = {}
    monkeypatch.setattr(session_mod.rtc, "AudioStream", lambda track, **kw: seen.update(kw))
    LiveKitIo().new_stream(object())
    assert seen["capacity"] > 0  # unbounded (0) would let latency grow without limit
