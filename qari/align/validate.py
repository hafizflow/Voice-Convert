"""Validation helpers: defect statistics and per-word clips for listening checks."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from qari.align.quran_align import INTERPOLATED, SPLIT, SUSPECT
from qari.align.store import TimestampStore

Fetch = Callable[[str], tuple[np.ndarray, int]]
GAP_MS = 500
TINY_MS = 40   # a word shorter than this cannot be heard as itself


def stats(store: TimestampStore) -> dict:
    idx = store.index
    s = {"words": len(idx), "non_positive": 0, "overlaps": 0, "open_ended": 0,
         "gaps_over_500ms": 0, "suspect_ayahs": len(store._suspect), "tiny": 0,
         "flags": {"ok": 0, "split": 0, "interpolated": 0, "suspect": 0}}
    for s_, a in sorted({(w.surah, w.ayah) for w in idx.words}):
        first, last = idx.ayah_range(s_, a)
        prev_end = None
        for g in range(first, last + 1):
            w = store.word(g)
            s["flags"]["split" if w.flag == SPLIT else "interpolated" if w.flag == INTERPOLATED
                       else "suspect" if w.flag == SUSPECT else "ok"] += 1
            if w.flag == SUSPECT:
                continue   # no per-word times by design; whole-ayah playback only
            if w.end_ms is None:
                s["open_ended"] += 1
            elif w.end_ms <= w.start_ms:
                s["non_positive"] += 1
            elif w.end_ms - w.start_ms < TINY_MS:
                s["tiny"] += 1
            if prev_end is not None:
                if w.start_ms < prev_end:
                    s["overlaps"] += 1
                elif w.start_ms - prev_end > GAP_MS:
                    s["gaps_over_500ms"] += 1
            prev_end = w.end_ms
    return s


def cut_clip(audio: np.ndarray, sr: int, start_ms: int, end_ms: int | None, pad_ms: int = 0) -> np.ndarray:
    a = max(0, int((start_ms - pad_ms) * sr / 1000))
    b = len(audio) if end_ms is None else min(len(audio), int((end_ms + pad_ms) * sr / 1000))
    return audio[a:b]


def fetch_audio(url: str) -> tuple[np.ndarray, int]:
    import io

    import httpx
    for attempt in range(3):
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 2:
                raise
    data, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return (data.mean(axis=1) if data.ndim > 1 else data), sr


def sample_clips(store: TimestampStore, n_ayahs: int, seed: int, out_dir: Path,
                 fetch: Fetch = fetch_audio, pad_ms: int = 0) -> list[Path]:
    """Cut every word of `n_ayahs` random ayahs into <out_dir>/<surah>_<ayah>_<pos>_<norm>.wav."""
    idx = store.index
    ayahs = sorted({(w.surah, w.ayah) for w in idx.words})
    chosen = random.Random(seed).sample(ayahs, n_ayahs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s, a in sorted(chosen):
        audio, sr = fetch(store.reciter.audio_url(s, a))
        first, last = idx.ayah_range(s, a)
        for g in range(first, last + 1):
            w, t = idx[g], store.word(g)
            path = out_dir / f"{s:03d}_{a:03d}_{w.pos:02d}_{w.norm}.wav"
            sf.write(path, cut_clip(audio, sr, t.start_ms, t.end_ms, pad_ms), sr)
            written.append(path)
    return written


def check_durations(store: TimestampStore, n_ayahs: int, seed: int, fetch: Fetch = fetch_audio) -> list[dict]:
    """For random ayahs compare the last word's end with the real file length.
    A large tail, or an end past the file, means the alignment and the audio disagree."""
    idx = store.index
    ayahs = sorted({(w.surah, w.ayah) for w in idx.words})
    rows = []
    for s, a in sorted(random.Random(seed).sample(ayahs, n_ayahs)):
        audio, sr = fetch(store.reciter.audio_url(s, a))
        _, last = idx.ayah_range(s, a)
        end = store.word(last).end_ms
        dur = round(len(audio) * 1000 / sr)
        rows.append({"surah": s, "ayah": a, "last_end_ms": end, "audio_ms": dur,
                     "tail_ms": None if end is None else dur - end})
    return rows
