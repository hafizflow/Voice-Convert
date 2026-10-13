# Skeleton Pipeline (Pass-through) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working end-to-end real-time pipeline (browser mic → LiveKit → Python worker → converted track → browser speaker) with a pass-through converter, a token server, a voice registry, metrics, and a test web page, so the RVC engine can later drop in behind one interface.

**Architecture:** Three processes: LiveKit server, a **worker** (FastAPI internal API + asyncio LiveKit sessions, one process so models/GPU are shared and capacity is enforced in-process) and a **token server** (FastAPI, public). The token server asks the worker to start a session in a fresh private room, then returns a LiveKit JWT to the browser. All audio logic sits behind `StreamingConverter` in `engine/`, which has no networking imports.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, `livekit` (rtc), `livekit-api`, prometheus-client, numpy, PyYAML, pytest + pytest-asyncio, vanilla JS + `livekit-client` (CDN, pinned).

**Spec:** `docs/superpowers/specs/2026-10-13-realtime-arabic-voice-conversion-design.md`

**Deviation from spec §3 to review:** the spec says "LiveKit worker"; this plan implements it with the plain `livekit` rtc SDK plus an internal HTTP session API rather than the LiveKit Agents job framework. Reason: Agents runs each job in its own process by default, which fights a shared GPU model, in-process concurrency limits, and a single warmed-up engine. The token server learns about capacity synchronously (HTTP 503 at token time).

## Global Constraints

- Opus, 48 kHz, mono, 20 ms frames end to end (`SAMPLE_RATE = 48000`).
- Target voice is config: a directory under `voices/` with `voice.yaml`; never hard-coded.
- Every voice entry requires a non-empty `consent_source`; the loader refuses entries without it.
- Converted audio goes only to the speaking user's room. No recording, saving or exporting; no audio written to disk or logs; no LiveKit egress configured.
- UI always shows "AI practice voice". This is not a real recitation by the reciter.
- Browser audio capture defaults: noise suppression and AGC off; echo cancellation on only when the user is not on headphones.
- Hot path (`FrameProcessor.process`) must not allocate beyond numpy temporaries and must never emit NaN/Inf.
- Python is pinned to 3.11 (local macOS is 3.14, which many ML wheels do not support).
- Scope exclusions (later plans): RVC inference, VAD, SOLA chunker, pitch shift, watermark, queue-depth metric, GPU, TensorRT, mobile client.

## Review Focus

- A frame with zero samples, odd byte length, or 2 channels must not crash the session (Task 2, Task 4).
- A converter that returns NaN/Inf or the wrong length must produce silence/an error, never corrupted or unconverted audio (Task 4).
- A malformed `voice.yaml`, or one missing `consent_source`, must be skipped without stopping the server (Task 1).
- A user who requests a token and never joins, or who leaves mid-stream, must not leak a session slot (Task 5, Task 8).
- At capacity or with an unknown voice the client must get a clear 503/404, not a token that hangs (Task 6, Task 7).

---

## File Structure

```
.gitignore  pyproject.toml  requirements.txt  requirements-dev.txt  .env.example  README.md
voices/demo/voice.yaml
engine/__init__.py  engine/registry.py  engine/converter.py
server/__init__.py  server/config.py  server/audio.py  server/metrics.py
server/manager.py  server/session.py  server/worker_app.py  server/token_server.py
web/index.html
tools/smoke_client.py
docker/Dockerfile  docker-compose.yml
tests/test_registry.py  tests/test_converter.py  tests/test_audio.py  tests/test_config.py
tests/test_manager.py  tests/test_frame_processor.py  tests/test_worker_app.py
tests/test_token_server.py  tests/integration/test_pipeline.py
```

---

### Task 1: Project scaffold and voice registry

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `engine/__init__.py`, `engine/registry.py`, `voices/demo/voice.yaml`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces: `engine.registry.Voice` (frozen dataclass: `id: str, name: str, f0_median_hz: float, consent_source: str, model_path: Path | None, index_path: Path | None`), `engine.registry.load_registry(root: Path) -> dict[str, Voice]`.

- [ ] **Step 1: Initialise repo and environment**

```bash
cd /Users/hafiz/Dev/Voice-Convert
git init
which uv || brew install uv
uv venv --python 3.11 .venv
source .venv/bin/activate
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
.DS_Store
voices/*/model.pth
voices/*/*.index
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
asyncio_mode = "auto"
markers = ["integration: needs a running LiveKit server, worker and token server (VC_INTEGRATION=1)"]
```

`requirements.txt`:
```
fastapi
uvicorn[standard]
livekit
livekit-api
prometheus-client
pyyaml
numpy
httpx
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest
pytest-asyncio
pyjwt
```

- [ ] **Step 2: Install and freeze**

Run: `uv pip install -r requirements-dev.txt && uv pip freeze > requirements.lock`
Expected: installs cleanly on Python 3.11. `requirements.lock` records exact versions (commit it).

- [ ] **Step 3: Write the failing tests**

`tests/test_registry.py`:
```python
from pathlib import Path

from engine.registry import load_registry


def write_voice(root: Path, name: str, yaml_text: str, files: tuple[str, ...] = ()) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "voice.yaml").write_text(yaml_text)
    for f in files:
        (d / f).write_bytes(b"x")


def test_loads_valid_voice(tmp_path):
    write_voice(
        tmp_path, "qari1",
        "name: Qari One\nf0_median_hz: 130.5\nconsent_source: licensed by X, contract 12\n"
        "model: model.pth\nindex: added.index\n",
        files=("model.pth", "added.index"),
    )
    reg = load_registry(tmp_path)
    v = reg["qari1"]
    assert v.name == "Qari One"
    assert v.f0_median_hz == 130.5
    assert v.model_path == tmp_path / "qari1" / "model.pth"
    assert v.index_path == tmp_path / "qari1" / "added.index"


def test_model_is_optional_for_passthrough_voice(tmp_path):
    write_voice(tmp_path, "demo", "name: Demo\nf0_median_hz: 120\nconsent_source: demo\n")
    v = load_registry(tmp_path)["demo"]
    assert v.model_path is None and v.index_path is None


def test_missing_consent_source_is_skipped(tmp_path):
    write_voice(tmp_path, "bad", "name: Bad\nf0_median_hz: 120\n")
    assert load_registry(tmp_path) == {}


def test_blank_consent_source_is_skipped(tmp_path):
    write_voice(tmp_path, "bad", "name: Bad\nf0_median_hz: 120\nconsent_source: '  '\n")
    assert load_registry(tmp_path) == {}


def test_malformed_yaml_does_not_stop_other_voices(tmp_path):
    write_voice(tmp_path, "broken", "name: [unclosed\n")
    write_voice(tmp_path, "ok", "name: Ok\nf0_median_hz: 100\nconsent_source: own recording\n")
    assert set(load_registry(tmp_path)) == {"ok"}


def test_declared_model_file_missing_is_skipped(tmp_path):
    write_voice(tmp_path, "nomodel", "name: N\nf0_median_hz: 100\nconsent_source: c\nmodel: model.pth\n")
    assert load_registry(tmp_path) == {}


def test_nonexistent_root_gives_empty_registry(tmp_path):
    assert load_registry(tmp_path / "nope") == {}


def test_nonpositive_f0_is_skipped(tmp_path):
    write_voice(tmp_path, "z", "name: Z\nf0_median_hz: 0\nconsent_source: c\n")
    assert load_registry(tmp_path) == {}
```

- [ ] **Step 4: Run to verify failure**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'engine'`.

- [ ] **Step 5: Implement**

`engine/__init__.py`: empty file.

`engine/registry.py`:
```python
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    f0_median_hz: float
    consent_source: str
    model_path: Path | None
    index_path: Path | None


def _load_one(d: Path) -> Voice:
    meta = yaml.safe_load((d / "voice.yaml").read_text())
    if not isinstance(meta, dict):
        raise ValueError("voice.yaml is not a mapping")
    consent = meta.get("consent_source")
    if not isinstance(consent, str) or not consent.strip():
        raise ValueError("consent_source is required")
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    f0 = float(meta["f0_median_hz"])
    if f0 <= 0:
        raise ValueError("f0_median_hz must be positive")
    model = _optional_file(d, meta.get("model"))
    index = _optional_file(d, meta.get("index"))
    return Voice(d.name, name.strip(), f0, consent.strip(), model, index)


def _optional_file(d: Path, rel) -> Path | None:
    if rel is None:
        return None
    p = (d / str(rel)).resolve()
    if d.resolve() not in p.parents or not p.is_file():
        raise ValueError(f"declared file not found inside voice dir: {rel}")
    return d / str(rel)


def load_registry(root: Path) -> dict[str, Voice]:
    voices: dict[str, Voice] = {}
    if not root.is_dir():
        return voices
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            voices[d.name] = _load_one(d)
        except Exception as e:  # one bad voice must not stop the server
            log.warning("skipping voice %s: %s", d.name, e)
    return voices
```

`voices/demo/voice.yaml`:
```yaml
name: Demo (pass-through)
f0_median_hz: 120
consent_source: "Placeholder voice for pipeline testing; no real voice, no model."
```

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_registry.py -v`
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: scaffold project and voice registry with consent_source enforcement"
```

---

### Task 2: StreamingConverter interface and audio helpers

**Files:**
- Create: `engine/converter.py`, `server/__init__.py`, `server/audio.py`
- Test: `tests/test_converter.py`, `tests/test_audio.py`

**Interfaces:**
- Produces: `engine.converter.StreamingConverter` (Protocol: `latency_samples: int`, `process_block(block: np.ndarray) -> np.ndarray`, `reset() -> None`; block is float32 mono shape `(n,)` in [-1, 1] at 48 kHz, output has the same shape). `engine.converter.PassthroughConverter()`.
- Produces: `server.audio.pcm16_to_mono_f32(data: bytes | memoryview, num_channels: int) -> np.ndarray`, `server.audio.f32_to_pcm16_bytes(x: np.ndarray) -> bytes`.

- [ ] **Step 1: Write the failing tests**

`tests/test_converter.py`:
```python
import numpy as np
import pytest

from engine.converter import PassthroughConverter


def test_passthrough_returns_equal_copy():
    c = PassthroughConverter()
    x = np.linspace(-1, 1, 480, dtype=np.float32)
    y = c.process_block(x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.array_equal(x, y)
    y[0] = 9.0
    assert x[0] != 9.0  # no aliasing


def test_passthrough_declares_zero_latency():
    assert PassthroughConverter().latency_samples == 0


@pytest.mark.parametrize("bad", [np.zeros((2, 10), np.float32), np.zeros(10, np.float64)])
def test_passthrough_rejects_wrong_shape_or_dtype(bad):
    with pytest.raises(ValueError):
        PassthroughConverter().process_block(bad)
```

`tests/test_audio.py`:
```python
import numpy as np
import pytest

from server.audio import f32_to_pcm16_bytes, pcm16_to_mono_f32


def pcm(values):
    return np.array(values, dtype=np.int16).tobytes()


def test_mono_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    out = pcm16_to_mono_f32(f32_to_pcm16_bytes(x), 1)
    assert np.allclose(out, x, atol=1e-3)


def test_stereo_is_downmixed_by_average():
    out = pcm16_to_mono_f32(pcm([16384, -16384, 16384, 16384]), 2)
    assert np.allclose(out, [0.0, 0.5], atol=1e-4)


def test_empty_frame_gives_empty_array():
    assert pcm16_to_mono_f32(b"", 1).size == 0


def test_odd_byte_length_raises_value_error():
    with pytest.raises(ValueError):
        pcm16_to_mono_f32(b"\x00\x01\x02", 1)


def test_sample_count_not_multiple_of_channels_raises():
    with pytest.raises(ValueError):
        pcm16_to_mono_f32(pcm([1, 2, 3]), 2)


def test_memoryview_is_accepted():
    assert pcm16_to_mono_f32(memoryview(pcm([100, 200])), 1).size == 2


def test_output_clips_and_removes_nan():
    x = np.array([2.0, -2.0, np.nan, np.inf], dtype=np.float32)
    out = np.frombuffer(f32_to_pcm16_bytes(x), dtype=np.int16)
    assert out[0] == 32767 and out[1] == -32767
    assert out[2] == 0
    assert out[3] == 32767
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_converter.py tests/test_audio.py -v`
Expected: FAIL, import errors.

- [ ] **Step 3: Implement**

`engine/converter.py`:
```python
from typing import Protocol

import numpy as np


class StreamingConverter(Protocol):
    """One instance per session. Blocks are float32 mono, shape (n,), 48 kHz.

    process_block returns an array of the same shape, aligned in time with the
    input shifted by `latency_samples` of algorithmic delay.
    """

    latency_samples: int

    def process_block(self, block: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class PassthroughConverter:
    latency_samples = 0

    def process_block(self, block: np.ndarray) -> np.ndarray:
        if block.ndim != 1 or block.dtype != np.float32:
            raise ValueError("block must be float32 with shape (n,)")
        return block.copy()

    def reset(self) -> None:
        pass
```

`server/__init__.py`: empty file.

`server/audio.py`:
```python
import numpy as np


def pcm16_to_mono_f32(data, num_channels: int) -> np.ndarray:
    if num_channels < 1:
        raise ValueError("num_channels must be >= 1")
    x = np.frombuffer(data, dtype=np.int16)  # raises ValueError on odd byte length
    if x.size % num_channels:
        raise ValueError("sample count is not a multiple of channel count")
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    x = x.reshape(-1, num_channels).astype(np.float32) / 32768.0
    return x.mean(axis=1, dtype=np.float32)


def f32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_converter.py tests/test_audio.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: StreamingConverter interface, pass-through converter, PCM helpers"
```

---

### Task 3: Config and metrics

**Files:**
- Create: `server/config.py`, `server/metrics.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `server.config.Config` (frozen dataclass: `livekit_url, livekit_internal_url, livekit_api_key, livekit_api_secret, app_api_key, internal_secret, worker_url: str; voices_dir: Path; max_sessions: int; join_timeout_s: float`), `server.config.load_config(env: Mapping[str, str] | None = None) -> Config`.
- Produces: `server.metrics.ACTIVE` (Gauge `vc_active_sessions`), `INFERENCE_MS` (Histogram `vc_inference_ms`), `SERVER_LATENCY_MS` (Histogram `vc_server_latency_ms`), `FRAME_ERRORS` (Counter `vc_frame_errors_total`).

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from server.config import load_config

BASE = {
    "LIVEKIT_URL": "ws://localhost:7880",
    "LIVEKIT_API_KEY": "k",
    "LIVEKIT_API_SECRET": "s",
    "APP_API_KEY": "app",
    "INTERNAL_SECRET": "int",
    "WORKER_URL": "http://127.0.0.1:8100",
}


def test_defaults():
    c = load_config(BASE)
    assert c.livekit_internal_url == "ws://localhost:7880"
    assert c.voices_dir == Path("voices")
    assert c.max_sessions == 4
    assert c.join_timeout_s == 30.0


def test_overrides():
    c = load_config({**BASE, "LIVEKIT_INTERNAL_URL": "ws://lk:7880", "MAX_SESSIONS": "9",
                     "JOIN_TIMEOUT_S": "2.5", "VOICES_DIR": "/v"})
    assert (c.livekit_internal_url, c.max_sessions, c.join_timeout_s, c.voices_dir) == (
        "ws://lk:7880", 9, 2.5, Path("/v"))


@pytest.mark.parametrize("missing", list(BASE))
def test_missing_required_var_raises(missing):
    env = {k: v for k, v in BASE.items() if k != missing}
    with pytest.raises(RuntimeError, match=missing):
        load_config(env)


def test_max_sessions_must_be_positive():
    with pytest.raises(RuntimeError, match="MAX_SESSIONS"):
        load_config({**BASE, "MAX_SESSIONS": "0"})
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: server.config`.

- [ ] **Step 3: Implement**

`server/config.py`:
```python
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Config:
    livekit_url: str            # URL browsers connect to
    livekit_internal_url: str   # URL the worker connects to
    livekit_api_key: str
    livekit_api_secret: str
    app_api_key: str            # shared key the test page sends to /token
    internal_secret: str        # token server -> worker
    worker_url: str
    voices_dir: Path
    max_sessions: int
    join_timeout_s: float


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env

    def req(name: str) -> str:
        v = env.get(name, "")
        if not v:
            raise RuntimeError(f"missing required env var {name}")
        return v

    max_sessions = int(env.get("MAX_SESSIONS", "4"))
    if max_sessions < 1:
        raise RuntimeError("MAX_SESSIONS must be >= 1")
    return Config(
        livekit_url=req("LIVEKIT_URL"),
        livekit_internal_url=env.get("LIVEKIT_INTERNAL_URL") or req("LIVEKIT_URL"),
        livekit_api_key=req("LIVEKIT_API_KEY"),
        livekit_api_secret=req("LIVEKIT_API_SECRET"),
        app_api_key=req("APP_API_KEY"),
        internal_secret=req("INTERNAL_SECRET"),
        worker_url=req("WORKER_URL"),
        voices_dir=Path(env.get("VOICES_DIR", "voices")),
        max_sessions=max_sessions,
        join_timeout_s=float(env.get("JOIN_TIMEOUT_S", "30")),
    )
```

`server/metrics.py`:
```python
from prometheus_client import Counter, Gauge, Histogram

_MS_BUCKETS = (0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320)

ACTIVE = Gauge("vc_active_sessions", "Active conversion sessions")
INFERENCE_MS = Histogram("vc_inference_ms", "Per-block converter time in ms", buckets=_MS_BUCKETS)
SERVER_LATENCY_MS = Histogram(
    "vc_server_latency_ms", "Frame received to converted frame handed to publisher, ms", buckets=_MS_BUCKETS
)
FRAME_ERRORS = Counter("vc_frame_errors", "Frames that failed processing (silence was sent)")
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_config.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: env config loader and Prometheus metrics"
```

---

### Task 4: FrameProcessor (hot-path core of a session)

**Files:**
- Create: `server/session.py` (FrameProcessor only in this task; LiveKit glue is added in Task 5)
- Test: `tests/test_frame_processor.py`

**Interfaces:**
- Consumes: `StreamingConverter`, `pcm16_to_mono_f32`, `f32_to_pcm16_bytes`, `metrics.INFERENCE_MS`.
- Produces: `server.session.FrameProcessor(converter)` with `process(data: bytes | memoryview, num_channels: int) -> bytes` (mono PCM16, same sample count as input frame; `b""` for an empty frame; raises `ValueError` if the converter returns the wrong shape), and attribute `last_inference_ms: float`.

- [ ] **Step 1: Write the failing tests**

`tests/test_frame_processor.py`:
```python
import numpy as np
import pytest
from prometheus_client import REGISTRY

from engine.converter import PassthroughConverter
from server.session import FrameProcessor


def pcm(x) -> bytes:
    return (np.asarray(x, dtype=np.float32) * 32767).astype(np.int16).tobytes()


class Fixed:
    latency_samples = 0

    def __init__(self, fn):
        self.fn = fn

    def process_block(self, block):
        return self.fn(block)

    def reset(self):
        pass


def test_passthrough_preserves_audio():
    x = np.sin(np.linspace(0, 20, 960)).astype(np.float32) * 0.5
    out = FrameProcessor(PassthroughConverter()).process(pcm(x), 1)
    back = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767
    assert np.allclose(back, x, atol=1e-3)


def test_empty_frame_returns_empty_bytes():
    assert FrameProcessor(PassthroughConverter()).process(b"", 1) == b""


def test_stereo_frame_is_downmixed_to_mono_length():
    stereo = np.zeros(960 * 2, dtype=np.int16).tobytes()
    out = FrameProcessor(PassthroughConverter()).process(stereo, 2)
    assert len(out) == 960 * 2  # 960 mono int16 samples


def test_wrong_length_from_converter_raises():
    fp = FrameProcessor(Fixed(lambda b: b[:-1]))
    with pytest.raises(ValueError):
        fp.process(pcm(np.zeros(100)), 1)


def test_nan_output_becomes_silence_not_nan():
    fp = FrameProcessor(Fixed(lambda b: np.full_like(b, np.nan)))
    out = np.frombuffer(fp.process(pcm(np.ones(100) * 0.3), 1), dtype=np.int16)
    assert not out.any()


def test_records_inference_time():
    before = REGISTRY.get_sample_value("vc_inference_ms_count") or 0
    fp = FrameProcessor(PassthroughConverter())
    fp.process(pcm(np.zeros(480)), 1)
    assert fp.last_inference_ms >= 0
    assert REGISTRY.get_sample_value("vc_inference_ms_count") == before + 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_frame_processor.py -v`
Expected: FAIL, `ModuleNotFoundError: server.session`.

- [ ] **Step 3: Implement**

`server/session.py`:
```python
import time

import numpy as np

from engine.converter import StreamingConverter
from server import metrics
from server.audio import f32_to_pcm16_bytes, pcm16_to_mono_f32


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
        return f32_to_pcm16_bytes(y.astype(np.float32, copy=False))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_frame_processor.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: FrameProcessor with shape/NaN safety and inference timing"
```

---

### Task 5: SessionManager and the LiveKit ConversionSession

**Files:**
- Create: `server/manager.py`
- Modify: `server/session.py` (append `ConversionSession`)
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: `Config`, `Voice`, `StreamingConverter`, `FrameProcessor`, `metrics`.
- Produces: `server.manager.SessionManager(max_sessions: int)` with `start(room: str, run: Callable[[], Awaitable[None]]) -> None` (raises `AtCapacity` / `DuplicateRoom`), `active: int`, `async shutdown() -> None`. Exceptions `AtCapacity`, `DuplicateRoom`.
- Produces: `server.session.ConversionSession(cfg: Config, room_name: str, voice: Voice, converter: StreamingConverter)` with `async run() -> None` (connects, waits up to `cfg.join_timeout_s` for the user, pumps audio, ends when the user leaves or the connection drops, always disconnects).

- [ ] **Step 1: Write the failing tests for SessionManager**

`tests/test_manager.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_manager.py -v`
Expected: FAIL, `ModuleNotFoundError: server.manager`.

- [ ] **Step 3: Implement SessionManager**

`server/manager.py`:
```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_manager.py -v`
Expected: 6 passed.

- [ ] **Step 5: Implement ConversionSession (LiveKit glue)**

Before writing, check the installed API matches: `python -c "from livekit import rtc, api; print(rtc.AudioStream, rtc.AudioSource, rtc.LocalAudioTrack.create_audio_track, api.AccessToken)"`. If a name differs in the installed version, adapt to the installed API and note the change in the commit message.

Append to `server/session.py` (add the imports at the top of the file):
```python
import asyncio
import json
import logging
from datetime import timedelta

from livekit import api, rtc

from engine.registry import Voice
from server.config import Config

log = logging.getLogger(__name__)
SAMPLE_RATE = 48000


class ConversionSession:
    def __init__(self, cfg: Config, room_name: str, voice: Voice, converter: StreamingConverter):
        self._cfg = cfg
        self._room_name = room_name
        self._voice = voice
        self._processor = FrameProcessor(converter)

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
        room = rtc.Room()
        joined = asyncio.Event()
        closed = asyncio.Event()
        source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=200)
        tasks: set[asyncio.Task] = set()

        def spawn(coro) -> None:
            t = asyncio.create_task(coro)
            tasks.add(t)
            t.add_done_callback(tasks.discard)

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
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                spawn(self._pump(track, source))

        await room.connect(self._cfg.livekit_internal_url, self._worker_token())
        try:
            if room.remote_participants:
                joined.set()
            out = rtc.LocalAudioTrack.create_audio_track("converted", source)
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
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
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
            await asyncio.sleep(1.0)
            payload = json.dumps({
                "voice": self._voice.id,
                "inference_ms": round(self._processor.last_inference_ms, 3),
            }).encode()
            await room.local_participant.publish_data(payload, reliable=False, topic="stats")
```

- [ ] **Step 6: Run the full unit suite**

Run: `pytest tests -v --ignore=tests/integration`
Expected: all pass. (The LiveKit glue is exercised in Task 8's integration test.)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: SessionManager and LiveKit ConversionSession with join timeout and cleanup"
```

---

### Task 6: Worker internal API

**Files:**
- Create: `server/worker_app.py`
- Test: `tests/test_worker_app.py`

**Interfaces:**
- Consumes: `Config`, `load_registry`, `SessionManager`, `ConversionSession`, `PassthroughConverter`.
- Produces: `server.worker_app.create_app(cfg, registry, session_factory=ConversionSession, converter_factory=lambda voice: PassthroughConverter()) -> FastAPI`, and `build_app() -> FastAPI` for `uvicorn --factory`.
- HTTP: `POST /internal/sessions {"room": str, "voice_id": str}` with header `X-Internal-Secret` → 201 / 401 / 404 (unknown voice) / 409 (duplicate room) / 503 (at capacity). `GET /health` → `{"status": "ok", "active": int, "max": int}`. `GET /metrics` → Prometheus text.

- [ ] **Step 1: Write the failing tests**

`tests/test_worker_app.py`:
```python
import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.registry import Voice
from server.config import Config
from server.worker_app import create_app

CFG = Config("ws://x", "ws://x", "k", "s", "app", "int-secret", "http://w", Path("v"), 1, 5.0)
VOICES = {"demo": Voice("demo", "Demo", 120.0, "test", None, None)}
H = {"X-Internal-Secret": "int-secret"}


class FakeSession:
    def __init__(self, cfg, room, voice, converter):
        self.room = room

    async def run(self):
        await asyncio.Event().wait()


@pytest.fixture
def client():
    with TestClient(create_app(CFG, VOICES, session_factory=FakeSession)) as c:
        yield c


def test_wrong_secret_is_401(client):
    r = client.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"},
                    headers={"X-Internal-Secret": "nope"})
    assert r.status_code == 401


def test_unknown_voice_is_404(client):
    r = client.post("/internal/sessions", json={"room": "r1", "voice_id": "ghost"}, headers=H)
    assert r.status_code == 404


def test_create_then_health_shows_active(client):
    r = client.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
    assert r.status_code == 201
    assert client.get("/health").json() == {"status": "ok", "active": 1, "max": 1}


def test_capacity_is_503(client):
    client.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
    r = client.post("/internal/sessions", json={"room": "r2", "voice_id": "demo"}, headers=H)
    assert r.status_code == 503


def test_duplicate_room_is_409():
    cfg = Config("ws://x", "ws://x", "k", "s", "app", "int-secret", "http://w", Path("v"), 5, 5.0)
    with TestClient(create_app(cfg, VOICES, session_factory=FakeSession)) as c:
        c.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
        r = c.post("/internal/sessions", json={"room": "r1", "voice_id": "demo"}, headers=H)
        assert r.status_code == 409


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200 and "vc_active_sessions" in r.text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_worker_app.py -v`
Expected: FAIL, `ModuleNotFoundError: server.worker_app`.

- [ ] **Step 3: Implement**

`server/worker_app.py`:
```python
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from engine.converter import PassthroughConverter
from engine.registry import load_registry
from server.config import Config, load_config
from server.manager import AtCapacity, DuplicateRoom, SessionManager
from server.session import ConversionSession

log = logging.getLogger(__name__)


class SessionRequest(BaseModel):
    room: str
    voice_id: str


def create_app(cfg: Config, registry, session_factory=ConversionSession,
               converter_factory=lambda voice: PassthroughConverter()) -> FastAPI:
    manager = SessionManager(cfg.max_sessions)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await manager.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.post("/internal/sessions", status_code=201)
    async def create_session(body: SessionRequest, x_internal_secret: str = Header(default="")):
        if not hmac.compare_digest(x_internal_secret, cfg.internal_secret):
            raise HTTPException(401, "bad internal secret")
        voice = registry.get(body.voice_id)
        if voice is None:
            raise HTTPException(404, "unknown voice")
        session = session_factory(cfg, body.room, voice, converter_factory(voice))
        try:
            manager.start(body.room, session.run)
        except AtCapacity:
            raise HTTPException(503, "worker at capacity")
        except DuplicateRoom:
            raise HTTPException(409, "room already has a session")
        return {"room": body.room}

    @app.get("/health")
    async def health():
        return {"status": "ok", "active": manager.active, "max": manager.capacity}

    app.mount("/metrics", make_asgi_app())
    return app


def build_app() -> FastAPI:
    cfg = load_config()
    registry = load_registry(cfg.voices_dir)
    log.info("loaded voices: %s", sorted(registry))
    return create_app(cfg, registry)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_worker_app.py -v`
Expected: 6 passed. If `/metrics` returns a 307 redirect rather than 200, change the test to `client.get("/metrics/")`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: worker internal session API with capacity, health and metrics"
```

---

### Task 7: Token server

**Files:**
- Create: `server/token_server.py`
- Test: `tests/test_token_server.py`

**Interfaces:**
- Consumes: `Config`, `Voice`, a `worker_client` with `async start_session(room: str, voice_id: str) -> None`.
- Produces: `server.token_server.create_app(cfg, registry, worker_client) -> FastAPI`, `WorkerFull`, `HttpWorkerClient(base_url, secret)`, `build_app()`.
- HTTP: `GET /health`; `GET /voices` → `[{"id","name"}]`; `POST /token {"voice_id"}` with `X-API-Key` → `{"url","token","room","identity","voice_id"}`, 401 bad key, 404 unknown voice, 503 worker full, 502 worker unreachable; `GET /` serves `web/index.html`.

- [ ] **Step 1: Write the failing tests**

`tests/test_token_server.py`:
```python
from pathlib import Path

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from engine.registry import Voice
from server.config import Config
from server.token_server import WorkerFull, create_app

CFG = Config("ws://lk:7880", "ws://lk:7880", "devkey", "devsecret-devsecret-devsecret-1234",
             "app-key", "int", "http://w", Path("v"), 4, 30.0)
VOICES = {"demo": Voice("demo", "Demo", 120.0, "secret-consent-text", None, None)}
KEY = {"X-API-Key": "app-key"}


class FakeWorker:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def start_session(self, room, voice_id):
        if self.error:
            raise self.error
        self.calls.append((room, voice_id))


def client(worker=None):
    return TestClient(create_app(CFG, VOICES, worker or FakeWorker()))


def test_voices_lists_id_and_name_only():
    r = client().get("/voices")
    assert r.json() == [{"id": "demo", "name": "Demo"}]


def test_token_requires_api_key():
    r = client().post("/token", json={"voice_id": "demo"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_unknown_voice_is_404_and_no_session_started():
    w = FakeWorker()
    r = client(w).post("/token", json={"voice_id": "ghost"}, headers=KEY)
    assert r.status_code == 404 and w.calls == []


def test_token_is_scoped_to_one_private_room():
    w = FakeWorker()
    body = client(w).post("/token", json={"voice_id": "demo"}, headers=KEY).json()
    claims = jwt.decode(body["token"], CFG.livekit_api_secret, algorithms=["HS256"],
                        options={"verify_aud": False})
    assert claims["iss"] == "devkey"
    assert claims["sub"] == body["identity"]
    assert claims["video"]["room"] == body["room"]
    assert claims["video"]["roomJoin"] is True
    assert claims["exp"] - claims["iat"] <= 900
    assert body["url"] == "ws://lk:7880"
    assert w.calls == [(body["room"], "demo")]


def test_each_request_gets_a_new_room():
    c = client()
    a = c.post("/token", json={"voice_id": "demo"}, headers=KEY).json()
    b = c.post("/token", json={"voice_id": "demo"}, headers=KEY).json()
    assert a["room"] != b["room"] and a["identity"] != b["identity"]


def test_worker_full_is_503():
    r = client(FakeWorker(WorkerFull())).post("/token", json={"voice_id": "demo"}, headers=KEY)
    assert r.status_code == 503


def test_worker_unreachable_is_502():
    w = FakeWorker(httpx.ConnectError("down"))
    r = client(w).post("/token", json={"voice_id": "demo"}, headers=KEY)
    assert r.status_code == 502


def test_index_is_served_with_label():
    r = client().get("/")
    assert r.status_code == 200 and "AI practice voice" in r.text
```
(`test_index_is_served_with_label` passes only after Task 9 creates `web/index.html`; expect it to fail until then.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_token_server.py -v`
Expected: FAIL, `ModuleNotFoundError: server.token_server`.

- [ ] **Step 3: Implement**

`server/token_server.py`:
```python
import hmac
import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from livekit import api
from pydantic import BaseModel

from engine.registry import load_registry
from server.config import Config, load_config

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class WorkerFull(Exception):
    pass


class HttpWorkerClient:
    def __init__(self, base_url: str, secret: str):
        self._base = base_url.rstrip("/")
        self._secret = secret

    async def start_session(self, room: str, voice_id: str) -> None:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.post(
                f"{self._base}/internal/sessions",
                json={"room": room, "voice_id": voice_id},
                headers={"X-Internal-Secret": self._secret},
            )
        if r.status_code == 503:
            raise WorkerFull()
        r.raise_for_status()


class TokenRequest(BaseModel):
    voice_id: str


def create_app(cfg: Config, registry, worker_client) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/voices")
    async def voices():
        return [{"id": v.id, "name": v.name} for v in registry.values()]

    @app.post("/token")
    async def token(body: TokenRequest, x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key, cfg.app_api_key):
            raise HTTPException(401, "bad api key")
        if body.voice_id not in registry:
            raise HTTPException(404, "unknown voice")
        room = f"vc-{uuid.uuid4().hex[:12]}"
        identity = f"user-{uuid.uuid4().hex[:8]}"
        try:
            await worker_client.start_session(room, body.voice_id)
        except WorkerFull:
            raise HTTPException(503, "all conversion slots are busy, try again shortly")
        except httpx.HTTPError:
            log.exception("worker unavailable")
            raise HTTPException(502, "conversion worker unavailable")
        jwt_token = (
            api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_metadata(json.dumps({"voice_id": body.voice_id}))
            .with_grants(api.VideoGrants(
                room_join=True, room=room,
                can_publish=True, can_subscribe=True, can_publish_data=False,
            ))
            .with_ttl(timedelta(minutes=15))
            .to_jwt()
        )
        return {"url": cfg.livekit_url, "token": jwt_token, "room": room,
                "identity": identity, "voice_id": body.voice_id}

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    return app


def build_app() -> FastAPI:
    cfg = load_config()
    registry = load_registry(cfg.voices_dir)
    return create_app(cfg, registry, HttpWorkerClient(cfg.worker_url, cfg.internal_secret))
```

- [ ] **Step 4: Run to verify pass (except the index test)**

Run: `pytest tests/test_token_server.py -v`
Expected: 8 pass except `test_index_is_served_with_label`, which fails with a missing file until Task 9. If `with_ttl` does not exist in the installed `livekit-api`, use the equivalent TTL setter it provides and keep the 15-minute expiry (the test pins `exp - iat <= 900`).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: token server with per-session private rooms and worker handshake"
```

---

### Task 8: Smoke client and integration tests

**Files:**
- Create: `tools/smoke_client.py`, `tests/integration/test_pipeline.py`

**Interfaces:**
- Consumes: running LiveKit server, worker (`:8100`), token server (`:8000`).
- Produces: `tools.smoke_client.run_smoke(token_url, api_key, voice_id, seconds=10.0) -> list[float]` (server+network round-trip ms for each detected tone burst).

- [ ] **Step 1: Write the smoke client**

`tools/smoke_client.py`:
```python
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
```

`tests/integration/test_pipeline.py`:
```python
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


async def test_audio_round_trips_with_sane_latency():
    res = await run_smoke(TOKEN_URL, API_KEY, "demo", seconds=10.0)
    assert len(res) >= 3, "converted audio did not come back"
    res.sort()
    assert res[len(res) // 2] < 800, f"median round trip too high: {res}"


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
```

- [ ] **Step 2: Start the stack and run**

```bash
brew install livekit                      # provides livekit-server
livekit-server --dev &                    # ws://localhost:7880, key devkey / secret
export LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
       APP_API_KEY=dev-app-key INTERNAL_SECRET=dev-internal WORKER_URL=http://127.0.0.1:8100 \
       JOIN_TIMEOUT_S=3
uvicorn server.worker_app:build_app --factory --port 8100 &
uvicorn server.token_server:build_app --factory --port 8000 &
python -m tools.smoke_client --api-key dev-app-key
VC_INTEGRATION=1 pytest tests/integration -v
```
Expected: smoke prints `bursts>=5`, median in the low hundreds of ms or below; both integration tests pass. If no bursts return, inspect worker logs first (the likely cause is an rtc API mismatch in `ConversionSession`; fix there).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: headless smoke client and pipeline integration tests"
```

---

### Task 9: Test web page

**Files:**
- Create: `web/index.html`

**Interfaces:**
- Consumes: `GET /voices`, `POST /token`, LiveKit data topic `stats` `{"voice","inference_ms"}`.

- [ ] **Step 1: Verify the pinned client library URL**

Run: `npm view livekit-client versions --json | tail -5` then `curl -sI https://cdn.jsdelivr.net/npm/livekit-client@2.5.7/dist/livekit-client.umd.min.js | head -1`
Expected: `HTTP/2 200`. If it is 404, replace `2.5.7` below with a version from the npm list that is at least two weeks old, and keep it exact.

- [ ] **Step 2: Write the page**

`web/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Convert Test</title>
<style>
  :root { --bg:#0f1115; --fg:#e8e8e8; --muted:#8a8f98; --card:#1a1d24; --line:#2a2e37;
          --accent:#4ade80; --warn:#fbbf24; --bad:#f87171; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f7f9; --fg:#15171c; --muted:#5b616b; --card:#fff; --line:#d9dce2;
            --accent:#15803d; --warn:#b45309; --bad:#b91c1c; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
  main { max-width:560px; margin:0 auto; padding:16px; }
  h1 { font-size:18px; margin:8px 0 12px; }
  .banner { background:var(--card); border:1px solid var(--warn); color:var(--warn);
            border-radius:8px; padding:8px 12px; margin-bottom:12px; font-size:13px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:14px; margin-bottom:12px; }
  label { display:block; font-size:13px; color:var(--muted); margin:8px 0 2px; }
  input[type=text], input[type=password], select { width:100%; padding:8px; border-radius:6px;
          border:1px solid var(--line); background:var(--bg); color:var(--fg); font:inherit; }
  .row { display:flex; gap:8px; flex-wrap:wrap; }
  button { padding:10px 14px; border-radius:8px; border:1px solid var(--line);
           background:var(--card); color:var(--fg); font:inherit; cursor:pointer; }
  button.primary { background:var(--accent); color:#06210f; border-color:var(--accent); font-weight:600; }
  button:disabled { opacity:.45; cursor:default; }
  .check { display:flex; gap:8px; align-items:center; margin:6px 0; font-size:14px; color:var(--fg); }
  .meter { height:10px; background:var(--bg); border:1px solid var(--line); border-radius:5px; overflow:hidden; }
  .meter > div { height:100%; width:0; background:var(--accent); }
  dl { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:0; font-size:14px; }
  dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; }
  #status { font-size:14px; min-height:20px; }
  .bad { color:var(--bad); } .ok { color:var(--accent); }
</style>
</head>
<body>
<main>
  <h1>Real-time voice conversion (test)</h1>
  <div class="banner"><strong>AI practice voice.</strong> This is not a real recitation by the reciter.
    Converted audio plays only to you and is never recorded or saved.</div>

  <div class="card">
    <label for="apikey">App API key</label>
    <input id="apikey" type="password" autocomplete="off" placeholder="dev-app-key">
    <label for="voice">Voice</label>
    <select id="voice"></select>
    <div class="check"><input id="headphones" type="checkbox"><label for="headphones" style="margin:0;color:inherit">
      I am wearing headphones (otherwise echo cancellation is turned on and quality drops)</label></div>
    <details><summary>Audio processing</summary>
      <div class="check"><input id="aec" type="checkbox" checked><label for="aec" style="margin:0;color:inherit">Echo cancellation</label></div>
      <div class="check"><input id="ns" type="checkbox"><label for="ns" style="margin:0;color:inherit">Noise suppression (can distort recitation)</label></div>
      <div class="check"><input id="agc" type="checkbox"><label for="agc" style="margin:0;color:inherit">Auto gain control (can distort recitation)</label></div>
    </details>
    <div class="row" style="margin-top:12px">
      <button id="start" class="primary">Start</button>
      <button id="test">Latency test</button>
      <button id="stop" disabled>Stop</button>
    </div>
    <div id="status"></div>
  </div>

  <div class="card">
    <label>Input level (your mic)</label><div class="meter"><div id="inMeter"></div></div>
    <label>Output level (converted voice)</label><div class="meter"><div id="outMeter"></div></div>
  </div>

  <div class="card">
    <dl>
      <dt>Server inference</dt><dd id="sInf">-</dd>
      <dt>Receive jitter buffer</dt><dd id="sJit">-</dd>
      <dt>Test signal round trip</dt><dd id="sRtt">-</dd>
      <dt>Round-trip history</dt><dd id="sHist">-</dd>
    </dl>
    <p style="font-size:12px;color:var(--muted);margin:8px 0 0">
      The latency test sends a tone burst instead of your mic and times its return. It covers
      encode, network, server and decode, and excludes mic and speaker hardware (add roughly 20 to 60 ms).</p>
  </div>
</main>

<script src="https://cdn.jsdelivr.net/npm/livekit-client@2.5.7/dist/livekit-client.umd.min.js"></script>
<script>
const LK = LivekitClient;
const $ = (id) => document.getElementById(id);
let room = null, ctx = null, timers = [], elements = [], pendingClick = null, rtts = [];

function setStatus(text, cls = '') { const s = $('status'); s.textContent = text; s.className = cls; }

function syncToggles() {
  const hp = $('headphones').checked;
  $('aec').checked = !hp;
  $('ns').checked = false;
  $('agc').checked = false;
}
$('headphones').addEventListener('change', syncToggles);

async function loadVoices() {
  const voices = await (await fetch('/voices')).json();
  const sel = $('voice');
  for (const v of voices) {
    const o = document.createElement('option');
    o.value = v.id; o.textContent = v.name; sel.append(o);
  }
}

function meter(sourceNode, barId) {
  const an = ctx.createAnalyser();
  an.fftSize = 512;
  sourceNode.connect(an);
  const buf = new Float32Array(an.fftSize);
  const bar = $(barId);
  const active = ctx;
  (function tick() {
    if (active !== ctx || !room) { bar.style.width = '0'; return; }
    an.getFloatTimeDomainData(buf);
    let s = 0; for (const v of buf) s += v * v;
    bar.style.width = Math.min(100, Math.round(Math.sqrt(s / buf.length) * 400)) + '%';
    requestAnimationFrame(tick);
  })();
}

async function setupDetector(src) {
  const code = `class D extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c){let m=0;
    for(let k=0;k<c.length;k++){const a=Math.abs(c[k]);if(a>m)m=a}
    if(m>0.1)this.port.postMessage(currentTime)}return true}}registerProcessor('d',D)`;
  const url = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
  await ctx.audioWorklet.addModule(url);
  const node = new AudioWorkletNode(ctx, 'd');
  node.port.onmessage = (e) => {
    if (pendingClick === null) return;
    const ms = (e.data - pendingClick) * 1000;
    pendingClick = null;
    if (ms > 0) {
      rtts.push(ms); rtts = rtts.slice(-20);
      const sorted = [...rtts].sort((a, b) => a - b);
      $('sRtt').textContent = `${ms.toFixed(0)} ms (median ${sorted[Math.floor(sorted.length / 2)].toFixed(0)})`;
      $('sHist').textContent = rtts.map((x) => x.toFixed(0)).join(', ');
    }
  };
  const sink = ctx.createGain(); sink.gain.value = 0;
  src.connect(node); node.connect(sink).connect(ctx.destination);
}

function onTrack(track) {
  if (track.kind !== 'audio') return;
  const el = track.attach();   // plays the converted track; no record/save controls exist
  el.style.display = 'none';
  document.body.append(el); elements.push(el);
  const src = ctx.createMediaStreamSource(new MediaStream([track.mediaStreamTrack]));
  meter(src, 'outMeter');
  setupDetector(src);
  timers.push(setInterval(async () => {
    try {
      const rep = await track.receiver.getStats();
      rep.forEach((s) => {
        if (s.type === 'inbound-rtp' && s.kind === 'audio' && s.jitterBufferEmittedCount) {
          $('sJit').textContent = (s.jitterBufferDelay / s.jitterBufferEmittedCount * 1000).toFixed(0) + ' ms';
        }
      });
    } catch (_) {}
  }, 1000));
}

function onData(payload, _participant, _kind, topic) {
  if (topic !== 'stats') return;
  try {
    const s = JSON.parse(new TextDecoder().decode(payload));
    $('sInf').textContent = s.inference_ms.toFixed(2) + ' ms';
  } catch (_) {}
}

async function publishTestSignal() {
  const dest = ctx.createMediaStreamDestination();
  const osc = ctx.createOscillator(); osc.frequency.value = 1000;
  const g = ctx.createGain(); g.gain.value = 0;
  osc.connect(g).connect(dest); osc.start();
  const track = new LK.LocalAudioTrack(dest.stream.getAudioTracks()[0]);
  await room.localParticipant.publishTrack(track, { source: LK.Track.Source.Microphone });
  timers.push(setInterval(() => {
    const t = ctx.currentTime + 0.05;
    g.gain.setValueAtTime(0.5, t); g.gain.setValueAtTime(0, t + 0.02);
    pendingClick = t;
  }, 1500));
}

async function start(testMode) {
  if (room) return;
  setStatus('Requesting session…');
  $('start').disabled = $('test').disabled = true;
  try {
    const res = await fetch('/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': $('apikey').value },
      body: JSON.stringify({ voice_id: $('voice').value }),
    });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.json()).detail || res.statusText}`);
    const t = await res.json();
    ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
    await ctx.resume();
    room = new LK.Room({
      audioCaptureDefaults: {
        echoCancellation: $('aec').checked, noiseSuppression: $('ns').checked,
        autoGainControl: $('agc').checked, channelCount: 1,
      },
      adaptiveStream: false, dynacast: false,
    });
    room.on(LK.RoomEvent.TrackSubscribed, onTrack);
    room.on(LK.RoomEvent.DataReceived, onData);
    room.on(LK.RoomEvent.Disconnected, () => stop('Disconnected'));
    await room.connect(t.url, t.token);
    if (testMode) {
      await publishTestSignal();
    } else {
      await room.localParticipant.setMicrophoneEnabled(true);
      const pub = room.localParticipant.getTrackPublication(LK.Track.Source.Microphone);
      meter(ctx.createMediaStreamSource(new MediaStream([pub.track.mediaStreamTrack])), 'inMeter');
    }
    $('stop').disabled = false;
    setStatus(testMode ? 'Running latency test…' : 'Live. Speak now.', 'ok');
  } catch (e) {
    setStatus('Failed: ' + e.message, 'bad');
    await stop();
  }
}

async function stop(msg) {
  timers.forEach(clearInterval); timers = [];
  elements.forEach((e) => e.remove()); elements = [];
  const r = room; room = null;
  if (r) { try { await r.disconnect(); } catch (_) {} }
  if (ctx) { try { await ctx.close(); } catch (_) {} ctx = null; }
  pendingClick = null;
  $('start').disabled = $('test').disabled = false;
  $('stop').disabled = true;
  if (msg) setStatus(msg);
}

$('start').addEventListener('click', () => {
  if (!$('headphones').checked && !confirm('Without headphones the converted voice will feed back into the mic. Continue anyway?')) return;
  start(false);
});
$('test').addEventListener('click', () => start(true));
$('stop').addEventListener('click', () => stop('Stopped'));
syncToggles();
loadVoices().catch((e) => setStatus('Could not load voices: ' + e.message, 'bad'));
</script>
</body>
</html>
```

- [ ] **Step 3: Run the token-server index test**

Run: `pytest tests/test_token_server.py -v`
Expected: all 8 pass, including `test_index_is_served_with_label`.

- [ ] **Step 4: Manual verification in a browser**

With the stack from Task 8 running, open `http://localhost:8000`.
1. Voice list shows "Demo (pass-through)".
2. Enter `dev-app-key`, tick headphones, click **Latency test**: the "Test signal round trip" row updates about every 1.5 s with a value in the low hundreds of ms or less; "Server inference" shows a sub-millisecond value.
3. Click **Stop**, then **Start** with headphones on: grant the mic, speak, and hear yourself delayed; the input and output meters move.
4. Wrong API key shows a clear 401 message; stopping and checking `curl localhost:8100/health` shows `active: 0`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: test web page with mic path, latency test, meters and stats"
```

---

### Task 10: Docker and README

**Files:**
- Create: `docker/Dockerfile`, `docker-compose.yml`, `.env.example`, `README.md`

- [ ] **Step 1: Write the files**

`docker/Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY engine ./engine
COPY server ./server
COPY web ./web
COPY voices ./voices
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "server.worker_app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8100"]
```

`docker-compose.yml` (Linux hosts; `network_mode: host` does not work on Docker Desktop for Mac, which is why local Mac development uses `brew install livekit`):
```yaml
services:
  livekit:
    image: livekit/livekit-server:latest
    command: --dev --bind 0.0.0.0
    network_mode: host
  worker:
    build: { context: ., dockerfile: docker/Dockerfile }
    network_mode: host
    env_file: .env
  token:
    build: { context: ., dockerfile: docker/Dockerfile }
    command: ["uvicorn", "server.token_server:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
    network_mode: host
    env_file: .env
```

`.env.example`:
```
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
APP_API_KEY=change-me
INTERNAL_SECRET=change-me-too
WORKER_URL=http://127.0.0.1:8100
MAX_SESSIONS=4
JOIN_TIMEOUT_S=30
```

`README.md`: document the five commands from Task 8 Step 2, the test command (`pytest tests --ignore=tests/integration`), the privacy rules (no recording, no egress, audio never written to disk), the voice registry format with `consent_source`, and the explicit list of what is not built yet (RVC inference, VAD, SOLA, pitch shift, watermark, queue-depth metric, GPU image, mobile client).

- [ ] **Step 2: Verify the image builds**

Run: `docker build -f docker/Dockerfile -t vc-worker .`
Expected: builds successfully. (If Docker is not installed locally, say so in the commit message instead of claiming it was verified.)

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore: Dockerfile, compose for Linux hosts, env example and README"
```

---

## Self-Review

**Spec coverage (skeleton scope):** voice registry with `consent_source` → Task 1. Voice is config, not hard-coded → Tasks 1, 6, 7. Token server issuing LiveKit tokens, `/health`, `/metrics` → Tasks 6, 7. Session lifecycle, concurrency limits → Tasks 5, 6. Prometheus (active sessions, inference ms, server latency) → Task 3 (queue depth deferred: there is no queue until the Phase 3 inference thread exists). "AI practice voice" label, no recording controls, headphones prompt, AEC/NS/AGC toggles, level meters, latency display, voice picker → Task 9. True end-to-end latency tool (tone round trip, excluding hardware) → Tasks 8, 9. Dockerfile, compose → Task 10. Explicitly deferred to later plans: RVC inference, VAD, SOLA/chunker, pitch shift, watermark, TensorRT, mobile, TURN and autoscaling notes.

**Placeholder scan:** no TBD/TODO. Two steps say to adapt to the installed library if a name differs (`ConversionSession` rtc names, `with_ttl`, CDN version); each has a concrete check and a fallback.

**Type consistency:** `Voice`, `Config`, `StreamingConverter`, `FrameProcessor.process(data, num_channels)`, `SessionManager.start(room, run)`, `create_app(cfg, registry, ...)` signatures match across tasks and tests. `Config` positional argument order in tests (`livekit_url, livekit_internal_url, livekit_api_key, livekit_api_secret, app_api_key, internal_secret, worker_url, voices_dir, max_sessions, join_timeout_s`) matches the dataclass field order.

**Known risk:** `ConversionSession` and the browser page are verified only by the Task 8 integration test and the Task 9 manual run, because LiveKit has no in-process fake. That is where API drift would show up first.
