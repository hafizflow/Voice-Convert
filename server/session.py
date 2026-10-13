import asyncio
import json
import logging
import time
from datetime import timedelta

import numpy as np
from livekit import api, rtc

from engine.converter import StreamingConverter
from engine.registry import Voice
from server import metrics
from server.audio import f32_to_pcm16_bytes, pcm16_to_mono_f32
from server.config import Config

log = logging.getLogger(__name__)
SAMPLE_RATE = 48000
STREAM_CAPACITY_FRAMES = 5   # input queue bound (about 50 ms); oldest frames drop first
STATS_INTERVAL_S = 1.0


class FrameProcessor:
    """Pure per-frame logic: PCM16 in, converted mono PCM16 out. No LiveKit imports."""

    def __init__(self, converter: StreamingConverter):
        self._converter = converter
        self.last_inference_ms = 0.0

    def process(self, data, num_channels: int) -> bytes:
        x = pcm16_to_mono_f32(data, num_channels)
        if x.size == 0:
            return b""
        t0 = time.perf_counter()
        y = self._converter.process_block(x)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        metrics.INFERENCE_MS.observe(self.last_inference_ms)
        y = np.asarray(y)
        if y.shape != x.shape:
            raise ValueError(f"converter returned shape {y.shape}, expected {x.shape}")
        y = y.astype(np.float32, copy=False)
        if not np.isfinite(y).all():
            raise ValueError("converter produced non-finite samples")
        return f32_to_pcm16_bytes(y)


class LiveKitIo:
    """Thin seam over the rtc SDK so ConversionSession can be unit-tested without a server."""

    def new_room(self):
        return rtc.Room()

    def new_source(self):
        return rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=200)

    def new_output_track(self, source):
        return rtc.LocalAudioTrack.create_audio_track("converted", source)

    def new_stream(self, track):
        # capacity > 0 makes the SDK drop the oldest frames when we fall behind,
        # so latency cannot grow without bound (spec section 6).
        return rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1,
                               capacity=STREAM_CAPACITY_FRAMES)


class ConversionSession:
    def __init__(self, cfg: Config, room_name: str, voice: Voice, converter: StreamingConverter,
                 io: LiveKitIo | None = None):
        self._cfg = cfg
        self._room_name = room_name
        self._voice = voice
        self._processor = FrameProcessor(converter)
        self._io = io or LiveKitIo()

    def _worker_token(self) -> str:
        return (
            api.AccessToken(self._cfg.livekit_api_key, self._cfg.livekit_api_secret)
            .with_identity(f"converter-{self._room_name}")
            .with_name("converter")
            .with_grants(api.VideoGrants(
                room_join=True, room=self._room_name,
                can_publish=True, can_subscribe=True, can_publish_data=True,
            ))
            .with_ttl(timedelta(hours=1))
            .to_jwt()
        )

    async def run(self) -> None:
        room = self._io.new_room()
        joined = asyncio.Event()
        closed = asyncio.Event()
        source = self._io.new_source()
        tasks: set[asyncio.Task] = set()
        pump: dict[str, asyncio.Task | None] = {"task": None}

        def spawn(coro) -> asyncio.Task:
            t = asyncio.create_task(coro)
            tasks.add(t)

            def done(t: asyncio.Task) -> None:
                tasks.discard(t)
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    log.error("room %s: background task failed, closing session",
                              self._room_name, exc_info=exc)
                    closed.set()

            t.add_done_callback(done)
            return t

        @room.on("participant_connected")
        def _joined(_p):
            joined.set()

        @room.on("participant_disconnected")
        def _left(_p):
            closed.set()

        @room.on("disconnected")
        def _dropped(*_a):
            closed.set()

        @room.on("track_subscribed")
        def _track(track, _pub, _participant):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if pump["task"] is not None:  # only one input feeds the output at a time
                pump["task"].cancel()
            pump["task"] = spawn(self._pump(track, source))

        await room.connect(self._cfg.livekit_internal_url, self._worker_token())
        try:
            if room.remote_participants:
                joined.set()
            out = self._io.new_output_track(source)
            await room.local_participant.publish_track(
                out, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            spawn(self._publish_stats(room))
            try:
                await asyncio.wait_for(joined.wait(), self._cfg.join_timeout_s)
            except asyncio.TimeoutError:
                log.info("room %s: user never joined, closing", self._room_name)
                return
            await closed.wait()
        finally:
            for t in list(tasks):
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await room.disconnect()

    async def _pump(self, track, source) -> None:
        stream = self._io.new_stream(track)
        try:
            async for ev in stream:
                t0 = time.perf_counter()
                frame = ev.frame
                try:
                    out = self._processor.process(frame.data, frame.num_channels)
                except Exception:
                    metrics.FRAME_ERRORS.inc()
                    log.exception("frame processing failed; sending silence")
                    out = bytes(frame.samples_per_channel * 2)
                if not out:
                    continue
                await source.capture_frame(
                    rtc.AudioFrame(out, SAMPLE_RATE, 1, len(out) // 2)
                )
                metrics.SERVER_LATENCY_MS.observe((time.perf_counter() - t0) * 1000.0)
        finally:
            await stream.aclose()

    async def _publish_stats(self, room) -> None:
        while True:
            await asyncio.sleep(STATS_INTERVAL_S)
            payload = json.dumps({
                "voice": self._voice.id,
                "inference_ms": round(self._processor.last_inference_ms, 3),
            }).encode()
            try:
                await room.local_participant.publish_data(payload, reliable=False, topic="stats")
            except Exception:
                log.debug("stats publish failed", exc_info=True)  # stats are best effort
