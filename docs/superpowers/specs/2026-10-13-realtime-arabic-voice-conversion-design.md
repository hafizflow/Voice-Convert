# Real-time Arabic Voice Conversion — Design Spec

Date: 2026-10-13
Status: DRAFT, awaiting user review

## 1. Goal

A user speaks or recites any Arabic into a mic and hears it played back **live** in a target
Qari's timbre. Same words, rhythm and melody; only timbre changes.

- Target mic → speaker latency: **< 250 ms**, stretch **< 180 ms** (see §5: this is at the edge; numbers decide).
- Voice conversion: RVC v2 (ContentVec/HuBERT + RMVPE + 40k/48k generator).
- Transport: WebRTC via self-hosted LiveKit, Opus 48 kHz mono, 20 ms frames.
- Server: Python worker, one GPU session per active user.
- Target voice is **config** (`.pth` + `.index` from a `voices/` registry), never hard-coded.

### Priorities (confirmed with user)
1. **Backend correctness and quality come first.** The streaming engine, server and ops are where the effort goes.
2. The frontend is a **thin web test client** for the user to try the system. The mobile app (framework undecided) comes later and reuses the same server API.

## 2. Confirmed decisions

| Topic | Decision |
|---|---|
| Test frontend | Single static `index.html`, vanilla JS, `livekit-client` pinned from CDN, served by the token server. No build step. |
| Backend target | Cloud GPU host + LiveKit (Mumbai or Singapore), matching the final architecture. |
| Local dev machine | macOS, no NVIDIA GPU. Local dev covers LiveKit, token server, pass-through worker and the offline simulator on CPU. Real-time GPU inference is tested on the cloud host. |

### Still open (asked in Phase 0 of the brief)
- Mobile framework: Flutter / React Native / native.
- Hosting provider and GPU budget (drives L4 vs A10G vs 4090 and max sessions).
- Which Qari voice(s) and the audio source + rights for each (`consent_source`).

## 3. Architecture

```
Browser/Mobile ──Opus/WebRTC──▶ LiveKit SFU ──▶ Worker (GPU)
      ▲                                           │ resample 48k→16k, VAD, ring buffer,
      │                                           │ streaming RVC, SOLA, watermark
      └────────────Opus/WebRTC◀── LiveKit SFU ◀───┘ published "converted" track

Token server (FastAPI): auth → LiveKit JWT, /voices, /health, /metrics
```

- One LiveKit room per session. The worker subscribes to the user's mic track and publishes one
  converted track back to the same room. Only that user's participant is permitted to subscribe to it.
- Worker is a **long-lived process** with models preloaded and warmed up. A session only binds a
  voice (loaded index + weights, cached in an LRU) and a stream state.

## 4. Repo structure

```
Voice-Convert/
├── voices/                    # registry: one dir per voice
│   └── <voice_id>/{model.pth, added.index, voice.yaml}   # voice.yaml has name, f0_median, consent_source
├── training/                  # Phase 1: prep_dataset.py, train.py, build_index.py, evaluate.py, config.yaml
├── engine/                    # Phase 2: framework-free streaming core (no LiveKit imports)
│   ├── converter.py           # StreamingConverter interface + Passthrough / RVC implementations
│   ├── chunker.py             # fixed blocks, context window, SOLA crossfade
│   ├── pitch.py               # running median F0 → smoothed semitone shift
│   ├── vad.py                 # Silero VAD + hangover
│   └── watermark.py           # AudioSeal
├── simulate.py                # offline simulator: WAV → same processing as live
├── server/                    # Phase 3
│   ├── token_server.py        # FastAPI: /token /voices /health /metrics + serves web/
│   ├── worker.py              # LiveKit worker
│   ├── session.py             # lifecycle, limits, degradation
│   └── metrics.py             # Prometheus
├── web/index.html             # test frontend
├── optimize/                  # Phase 4: ONNX/TensorRT export, parity + benchmark scripts
├── tools/latency_loopback.py  # Phase 6: chirp end-to-end measurement
├── tests/                     # chunker continuity, pitch shift, parity
├── docker/ + docker-compose.yml
└── docs/
```

Key boundary: **`engine/` knows nothing about networking.** The simulator and the live worker call
the identical `StreamingConverter.process_block()`, so what we verify offline is what runs live.

## 5. Latency budget (estimates, to be replaced by measurement)

| Stage | Est. ms | Notes |
|---|---|---|
| Mic capture + browser buffer | 10–20 | `latency` hint, low-latency input |
| Opus encode (one frame) | 20 | 20 ms frame |
| Uplink one-way, Bangladesh → Mumbai/Singapore | 30–50 | measure RTT/2 from real clients; mobile data is worse |
| Worker jitter buffer | 20 | smaller = more glitches on bad networks |
| Block accumulation (block size B) | 40–100 | the main tunable; algorithmic latency |
| Right-context / SOLA crossfade lookahead | 20–40 | needed for seam quality |
| Inference per block (TRT FP16) | 15–30 | must stay well below B |
| Opus encode out | 20 | |
| Downlink one-way | 30–50 | |
| Client jitter buffer + playout | 25–50 | |
| **Total** | **~230–420** | |

Honest read: with B = 80–100 ms the total lands around 300 ms. **< 250 ms needs B ≈ 40–60 ms, a
short crossfade and a good network, and costs quality and GPU load.** I will not claim 250 until
Phase 2 and 4 produce numbers; the benchmark table (block size vs latency vs quality vs GPU load)
decides the default, per the brief.

## 6. Streaming engine (the core)

- **Fixed-size blocks** of B ms at 16 kHz for the feature path; preallocated ring buffer and work
  arrays; no allocation in the hot path.
- **Sliding context window**: each inference sees `[extra_left | block | extra_right]`; only the
  block's span is emitted. `extra_left` 0.5–2 s tunable.
- **SOLA / crossfade** between consecutive block outputs to eliminate clicks and phase
  discontinuities. Overlap length tunable.
- **Auto pitch shift**: running median F0 of the speaker (voiced frames only, exponentially
  smoothed, slew-rate limited) transposed to the target voice's median F0 from `voice.yaml`.
  Clamped to ±12 semitones; no sudden jumps.
- **VAD** (Silero): skip inference on silence, short hangover, emit silence (and keep SOLA state
  consistent) to avoid noise artifacts and save GPU.
- **Graceful degradation**: if inference time > B, first shrink `extra_left`, then drop the oldest
  queued block and crossfade over the gap. Queue depth is bounded (≤ 2). Latency must never grow
  over a session; a unit/soak test asserts this.
- **Threading**: audio I/O on the asyncio loop; inference in a dedicated thread per session
  (or per-GPU worker pool) fed by a bounded queue.
- **Watermark**: AudioSeal on the output. Risks to verify in Phase 3: it operates at 16 kHz
  (needs resampling vs. the 48 kHz output), it adds latency/compute, and its robustness after
  Opus re-encoding must be measured.

## 7. Dependency health (to verify, not assume)

The RVC ecosystem is fragmented: the original RVC repo, various forks and real-time GUIs. At the
start of Phase 1 I will check which implementations are currently maintained and report the
recommendation (including for RMVPE, ContentVec and index handling) before depending on any of
them. Local Python is 3.14; ML wheels (PyTorch, onnxruntime, livekit) may lag, so the worker runs
in Docker pinned to a supported Python.

## 8. Safeguards (built in, not optional)

- Converted audio goes **only** to the speaking user's participant. No recording, saving, export
  or sharing. No user audio persisted on the server; LiveKit egress disabled; logs carry no audio.
- Inaudible watermark on all converted output.
- UI label: "AI practice voice — not a real recitation by the reciter."
- Every `voice.yaml` requires `consent_source`; the registry loader refuses voices without it.

## 9. Test frontend (`web/index.html`)

- Connect/disconnect, voice picker (from `/voices`), mic and output level meters.
- Headphone warning; toggles for echoCancellation / noiseSuppression / autoGainControl (off by
  default for headphones; recommended on for speaker mode).
- Latency display: WebRTC stats (RTT, jitter buffer) + server-reported inference ms and queue depth.
- Loopback chirp button feeding `tools/latency_loopback.py` logic for true end-to-end measurement.
- Playback only via `<audio>`; no record/download controls. "AI practice voice" label always visible.

## 10. Build order

Backend-first, with the pipeline testable end to end early:

1. **Skeleton**: token server + LiveKit (docker-compose) + **pass-through worker** + web page.
   Proves transport, token flow, meters and loopback latency before any ML.
2. **Phase 1**: dataset prep, training, FAISS index, evaluation (needs your audio).
3. **Phase 2**: offline simulator + `engine/` with tests (chunker continuity, pitch shift).
4. **Phase 3**: real RVC converter in the worker, VAD, degradation, metrics, watermark.
5. **Phase 4**: ONNX → TensorRT FP16, parity checks, benchmarks, max streams per GPU.
6. **Phase 5**: mobile client. **Phase 6**: Docker/deploy/TURN/autoscaling notes.

Each phase is verified with real output (tests, numbers, audio) before the next starts.

## 11. Out of scope

Recording/sharing features, multi-user rooms, non-Arabic content tuning, a production auth system
(token server accepts a simple API key / stub for now), and the mobile client until its framework
is chosen.
