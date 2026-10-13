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

## Real voice (RVC via Applio, Apple silicon)

```bash
# training (once; ~5 min/epoch for 42 min of audio on an M4)
cd third_party/applio && source ../../.venv-applio/bin/activate && export PYTORCH_ENABLE_MPS_FALLBACK=1
python core.py preprocess --model-name NAME --dataset-path DIR --sample-rate 40000 --cpu-cores 6 --cut-preprocess Automatic --normalization-mode post
python core.py extract    --model-name NAME --f0-method rmvpe --sample-rate 40000 --embedder-model contentvec --gpu - --cpu-cores 6   # --gpu - : MPS fails here
python core.py train      --model-name NAME --vocoder HiFi-GAN --total-epoch 50 --save-every-epoch 10 --save-every-weights --batch-size 8 --sample-rate 40000 --gpu 0 --pretrained
cd ../.. && python -m tools.install_voice --name NAME --display-name "..." --f0-source DIR/file.wav --consent-source "..."

# serve (worker must run in the Applio env, Python 3.12)
source .venv-applio/bin/activate
RVC_BLOCK_MS=200 RVC_F0=fcpe RVC_DEVICE=cpu uvicorn server.worker_app:build_app --factory --port 8100
# offline check and benchmark
python simulate.py IN.wav OUT.wav --model voices/NAME/model.pth --index voices/NAME/added.index
python -m tools.bench_blocks IN.wav --model voices/NAME/model.pth --index voices/NAME/added.index
```

Measured on an Apple M4 (see `tools/bench_blocks.py`): CPU beat MPS for streaming inference,
`fcpe` beat `rmvpe`, and the best config (200 ms blocks) runs at real-time factor about 0.85 with
250 ms of algorithmic latency, before the ~110 ms transport floor. Sub-250 ms end to end is not
reachable on this machine with this engine.

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
