# RVC Engine Integration Plan (Phase 2/3 on Apple M4)

**Goal:** Real RVC conversion behind `StreamingConverter`, usable from the existing worker and web page, plus an offline simulator that processes a WAV exactly like the live path.

**Spec:** `docs/superpowers/specs/2026-10-13-realtime-arabic-voice-conversion-design.md`
**Builds on:** `docs/superpowers/plans/2026-10-13-skeleton-pipeline.md` (done).

## Decisions (rulings, change here if wrong)

1. **Engine = Applio's `rvc.realtime.core.VoiceChanger`**, wrapped, not reimplemented. It already provides fixed 48 kHz blocks, extra left context, SOLA/phase-vocoder crossfade, VAD gate, RMVPE, FAISS index, and MPS handling. Cost if wrong: coupling to a third-party repo that changes weekly; mitigated by pinning the cloned commit (`c6fc349`) and keeping one adapter file.
2. **Worker runs in `.venv-applio` (Python 3.12)** with the server deps installed there; the project `.venv` (3.11) still runs the unit suite for the pure modules.
3. **Our code owns:** re-blocking arbitrary-size frames into fixed blocks (`engine/reblock.py`), the `RVCConverter` adapter, session wiring, the offline simulator, and (later) auto pitch shift. SOLA stays Applio's.
4. **Inference never runs on the event loop.** One single-thread executor per session handles model load and every `process` call (MPS tensors stay on one thread).
5. **Latency is explicit:** `RVCConverter.latency_samples = block + crossfade`. A `Reblocker` prefilled with one block of silence guarantees output length always equals input length.
6. **Applio resolves `rvc/models/...` against the cwd**, so the adapter `chdir`s to the Applio directory once; `voices_dir` is made absolute at config load.

## Tasks (all TDD; tests use a fake changer so the unit suite needs no GPU)

1. `engine/reblock.py` — `Reblocker(block_size, fn)`: `push(x: np.ndarray) -> np.ndarray` (same length as `x`; output delayed by exactly `block_size`; `fn` always receives exactly `block_size` samples).
   Tests: length invariant over random chunk sizes incl. 0 and > block; first `block_size` outputs are zeros; identity `fn` gives `out[B:] == in[:-B]`; `fn` always called with `B` samples.
2. `engine/rvc.py` — `RVCConverter(model_path, index_path, block_ms=100, extra_s=0.5, crossfade_s=0.05, index_rate=0.5, protect=0.33, f0_method="rmvpe", pitch_shift=0, device=None, changer_factory=None)`: implements `StreamingConverter`.
   Tests (fake changer): output length equals input length; `f0_up_key` and `index_rate` forwarded; `latency_samples`; `reset()` clears buffers; non-48 kHz block size rejected only via shape rules of `StreamingConverter`.
3. Session wiring — `ConversionSession` builds its converter inside `run()` on the session executor; every `FrameProcessor.process` runs on that executor; the worker's default factory returns `PassthroughConverter` for voices without a model and `RVCConverter` otherwise.
   Tests: converter factory failure ends the session and disconnects; process runs off the event-loop thread.
4. `simulate.py` — `python -m simulate IN.wav OUT.wav --voice qari012` with block/context/crossfade/index-rate/protect flags; prints per-block ms (mean/p95/max), real-time factor, algorithmic latency, and a boundary-discontinuity check.
   Tests: pass-through run reproduces the input delayed by the reported latency.
5. Auto pitch shift — `engine/pitch.py` `PitchShiftTracker(target_f0_hz)`: running median of voiced F0, smoothed, integer semitones, clamped to ±12, slew-limited.
   Tests: male target / female input shifts up toward target; no jumps larger than 1 semitone per update; unvoiced audio leaves the shift unchanged; clamp.

## Verification

- Unit: `pytest tests --ignore=tests/integration` in `.venv` (3.11) and in `.venv-applio` (3.12).
- Real: `python -m simulate` on `100.mp3` with the trained `qari012` model; real-time factor must be < 1 on MPS or the plan stops and reports.
- Live: worker + token server + LiveKit, then the web page with the voice picker.
