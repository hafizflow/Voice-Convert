"""Headless client: joins like the browser, sends tone bursts, measures burst round-trip.

Run: python -m tools.smoke_client --token-url http://localhost:8000 --api-key dev-app-key
"""
import argparse
import asyncio
import statistics
import time

import httpx
import numpy as np
from livekit import rtc

SR = 48000
FRAME = 960  # 20 ms


async def run_smoke(token_url: str, api_key: str, voice_id: str, seconds: float = 10.0) -> list[float]:
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.post(f"{token_url}/token", headers={"X-API-Key": api_key},
                            json={"voice_id": voice_id})
        r.raise_for_status()
        t = r.json()

    latencies: list[float] = []
    sent = {"t": None}
    last = {"t": 0.0}
    room = rtc.Room()

    async def receive(track):
        stream = rtc.AudioStream(track, sample_rate=SR, num_channels=1)
        async for ev in stream:
            x = np.frombuffer(ev.frame.data, dtype=np.int16)
            now = time.monotonic()
            if x.size and np.abs(x).max() > 3000 and sent["t"] and now - last["t"] > 1.0:
                latencies.append((now - sent["t"]) * 1000.0)
                last["t"] = now

    @room.on("track_subscribed")
    def _on_track(track, _pub, _participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(receive(track))

    await room.connect(t["url"], t["token"])
    source = rtc.AudioSource(SR, 1, queue_size_ms=100)
    await room.local_participant.publish_track(
        rtc.LocalAudioTrack.create_audio_track("smoke", source),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    tone = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(FRAME) / SR) * 32767).astype(np.int16).tobytes()
    silence = bytes(FRAME * 2)
    start = time.monotonic()
    n = 0
    while time.monotonic() - start < seconds:
        burst = n % 75 == 25  # a 20 ms burst every 1.5 s
        if burst:
            sent["t"] = time.monotonic()
        await source.capture_frame(rtc.AudioFrame(tone if burst else silence, SR, 1, FRAME))
        n += 1
        await asyncio.sleep(max(0.0, start + n * 0.02 - time.monotonic()))
    await room.disconnect()
    return latencies


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token-url", default="http://localhost:8000")
    p.add_argument("--api-key", required=True)
    p.add_argument("--voice", default="demo")
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    res = asyncio.run(run_smoke(a.token_url, a.api_key, a.voice, a.seconds))
    if not res:
        raise SystemExit("no bursts detected: pipeline is not returning audio")
    print(f"bursts={len(res)} min={min(res):.0f} median={statistics.median(res):.0f} "
          f"max={max(res):.0f} ms (round trip, includes Opus + SFU + worker)")
