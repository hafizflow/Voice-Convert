# Voice-Convert: real-time Arabic voice conversion

Status: **skeleton**. The full pipeline (mic → LiveKit → worker → speaker) runs with a
**pass-through** converter. RVC inference is not built yet.

Spec: `docs/superpowers/specs/2026-10-13-realtime-arabic-voice-conversion-design.md`
Plan: `docs/superpowers/plans/2026-10-13-skeleton-pipeline.md`

## Run locally (macOS)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
brew install livekit

# 1. LiveKit. --node-ip 127.0.0.1 is required locally: dev mode binds loopback
#    but otherwise advertises the LAN IP, and WebRTC connections time out.
livekit-server --dev --node-ip 127.0.0.1 &

# 2. Worker and token server
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
       APP_API_KEY=dev-app-key INTERNAL_SECRET=dev-internal WORKER_URL=http://127.0.0.1:8100
uvicorn server.worker_app:build_app --factory --port 8100 &
uvicorn server.token_server:build_app --factory --port 8000 &

# 3. Open http://127.0.0.1:8000, enter the API key, pick a voice, tick "headphones".
#    "Latency test" needs no microphone.
```

Use `127.0.0.1`, not `localhost`: another service may answer on `::1`.

## Tests

```bash
pytest tests --ignore=tests/integration                 # unit tests, no services needed
JOIN_TIMEOUT_S=3 ...start the stack above... 
VC_INTEGRATION=1 pytest tests/integration               # needs the running stack
python -m tools.smoke_client --api-key dev-app-key      # headless round-trip latency
```

## Voices

One directory per voice under `voices/<id>/` containing `voice.yaml`:

```yaml
name: Display name
f0_median_hz: 130
consent_source: "who granted the rights, and how"   # required, non-empty
model: model.pth       # optional until RVC is wired in
index: added.index     # optional
```

Entries without `consent_source`, or with a missing declared file, are skipped with a warning.

## Deploying

`docker-compose.yml` is for Linux hosts and has **not been built or run** (no Docker daemon was
available when it was written). It refuses to start the app with development secrets
(`VC_ENV=production`), binds the worker to loopback, and puts LiveKit's keys in from `.env`.
The metrics endpoint on the worker is unauthenticated: keep it off public interfaces.

## Privacy

Converted audio is returned only to the speaking user's private room. Nothing is recorded,
written to disk, or logged, and LiveKit egress is not configured. The UI always labels the
output "AI practice voice".

## Not built yet

RVC inference, VAD, SOLA chunker, auto pitch shift, watermark, queue-depth metric, GPU image,
TensorRT, mobile client, TURN and autoscaling notes.

## Known measurement

Pass-through round trip on localhost is about 110 to 130 ms (the smoke client). That is the
floor before any inference; it is not caused by the `AudioSource` queues. See the plan ledger
finding: likely stacked WebRTC receive jitter buffers. Investigate before trusting the
latency budget in the spec.
