"""Local test page for the Hear-the-Qari pipeline (no ASR yet).

    .venv/bin/uvicorn tools.qari_demo:app --port 8200      then open http://127.0.0.1:8200

Type what was recited, or use the mic button (Chrome's built-in Arabic speech recognition as a stand-in
for the real Quran ASR, which is a later sub-project). Matching, the Tanzil text and the Qari segments
are the real code paths. Audio is streamed from the reciter's origin URL; nothing is stored.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qari.align.store import TimestampStore
from qari.match.matcher import Matcher, MatchResult, SessionContext
from qari.reciters.registry import load_reciter
from qari.text.index import WordIndex

ROOT = Path(__file__).resolve().parents[1]
RECITER = "afasy"

app = FastAPI(title="Hear the Qari demo")
_state: dict = {}


def state() -> dict:
    if not _state:
        reciter = load_reciter(RECITER)
        path = ROOT / "data" / "timestamps" / reciter.timestamps_file
        if not path.exists():
            raise HTTPException(500, f"{path} missing: run `python -m tools.build_timestamps --reciter {RECITER}`")
        idx = WordIndex.from_file()
        _state.update(idx=idx, matcher=Matcher(idx), store=TimestampStore.load(path, reciter, idx), reciter=reciter)
    return _state


class MatchRequest(BaseModel):
    text: str
    asr_confidence: float = 1.0
    last_end_gid: int | None = None
    pad_ms: int = 100


def _ref(idx: WordIndex, a: int, b: int) -> str:
    s, e = idx[a], idx[b]
    if (s.surah, s.ayah) == (e.surah, e.ayah):
        return f"{s.surah}:{s.ayah}" + (f" (words {s.pos}-{e.pos})" if (s.pos, e.pos) != (1, idx.ayah_range(s.surah, s.ayah)[1] - idx.ayah_range(s.surah, s.ayah)[0] + 1) else "")
    return f"{s.surah}:{s.ayah} to {e.surah}:{e.ayah}"


def play_plan(st: dict, start: int, end: int, includes_basmala: bool, pad_ms: int) -> dict:
    """Segments the client plays in order, plus per-word timings for highlighting."""
    idx, store = st["idx"], st["store"]
    segs, notes = [], []
    if includes_basmala:
        b = store.basmala_segment()
        segs.append({"audio_url": b.audio_url, "start_ms": 0, "end_ms": None, "label": "Basmala", "words": []})
    for s in store.segments(start, end):
        a_first, a_last = idx.ayah_range(s.surah, s.ayah)
        lo, hi = a_first + s.first_pos - 1, a_first + s.last_pos - 1
        start_ms, end_ms = max(0, s.start_ms - pad_ms), (None if s.end_ms is None else s.end_ms + pad_ms)
        if s.ends_ayah:
            end_ms = None          # the aligned end can undershoot the file: play out the ayah's tail
        if s.quality == "unreliable":
            start_ms, end_ms, lo, hi = 0, None, a_first, a_last
            notes.append(f"{s.surah}:{s.ayah}: word timings unreliable in this ayah, playing the whole ayah")
        elif s.quality == "approx":
            notes.append(f"{s.surah}:{s.ayah}: some word edges are approximate")
        words = [{"surah": idx[g].surah, "ayah": idx[g].ayah, "pos": idx[g].pos, "text": idx[g].uthmani,
                  "start_ms": store.word(g).start_ms, "end_ms": store.word(g).end_ms,
                  "in_range": start <= g <= end} for g in range(lo, hi + 1)]
        segs.append({"audio_url": s.audio_url, "start_ms": start_ms, "end_ms": end_ms,
                     "label": f"{s.surah}:{s.ayah}", "quality": s.quality, "words": words})
    return {"segments": segs, "notes": notes}


def describe(st: dict, r: MatchResult, pad_ms: int) -> dict:
    idx = st["idx"]
    out = {"status": r.status, "confidence": round(r.confidence, 3), "score": round(r.score, 3),
           "confidence_parts": {k: round(v, 3) for k, v in r.confidence_parts.items()},
           "flags": {"istiadha_stripped": r.istiadha_stripped, "includes_basmala": r.includes_basmala,
                     "internal_basmala": r.internal_basmala, "vocative_joined": r.vocative_joined,
                     "used_session": r.used_session, "truncated": r.truncated},
           "range": None, "alternatives": [], "mistakes": []}
    if r.start_gid is not None:
        out["range"] = {"start_gid": r.start_gid, "end_gid": r.end_gid, "ref": _ref(idx, r.start_gid, r.end_gid),
                        "text": idx.text(r.start_gid, r.end_gid)}
        out["play"] = play_plan(st, r.start_gid, r.end_gid, r.includes_basmala, pad_ms)
        for o in r.ops:
            if o.kind in ("missing", "substitution") and o.gid is not None and (o.kind == "missing" or o.sim < 0.85):
                w = idx[o.gid]
                out["mistakes"].append({"type": o.kind, "surah": w.surah, "ayah": w.ayah, "word": w.pos,
                                        "expected": w.uthmani})
    for a in r.alternatives:
        out["alternatives"].append({"start_gid": a.start_gid, "end_gid": a.end_gid, "score": round(a.score, 3),
                                    "ref": _ref(idx, a.start_gid, a.end_gid), "text": idx.text(a.start_gid, a.end_gid)})
    return out


def best_guess(r: MatchResult, last_end_gid: int | None):
    """When the matcher will not commit, choose what to play anyway: highest score; among equals the one
    that continues forward from the previous verse; otherwise the earliest in the Quran."""
    if not r.alternatives:
        return None
    nxt = (last_end_gid + 1) if last_end_gid is not None else None

    def key(c):
        forward = c.start_gid - nxt if nxt is not None and c.start_gid >= nxt - 3 else float("inf")
        return (-round(c.score, 3), forward, c.start_gid)
    return min(r.alternatives, key=key)


@app.post("/api/match")
def match(req: MatchRequest) -> dict:
    st = state()
    t = time.perf_counter()
    session = SessionContext(req.last_end_gid) if req.last_end_gid is not None else None
    r = st["matcher"].match(req.text, req.asr_confidence, session)
    ms = (time.perf_counter() - t) * 1000
    out = describe(st, r, req.pad_ms)
    if r.start_gid is None:
        pick = best_guess(r, req.last_end_gid)
        if pick is not None:
            idx = st["idx"]
            out["range"] = {"start_gid": pick.start_gid, "end_gid": pick.end_gid,
                            "ref": _ref(idx, pick.start_gid, pick.end_gid),
                            "text": idx.text(pick.start_gid, pick.end_gid)}
            out["play"] = play_plan(st, pick.start_gid, pick.end_gid, r.includes_basmala, req.pad_ms)
            out["forced"] = {"reason": r.status, "score": round(pick.score, 3),
                             "candidates": len(r.alternatives)}
    out["timings_ms"] = {"match": round(ms, 1)}
    out["reciter"] = st["reciter"].display_name
    return out


@app.get("/api/play")
def play(start_gid: int, end_gid: int, pad_ms: int = 100) -> dict:
    st = state()
    n = len(st["idx"])
    if not 0 <= start_gid <= end_gid < n:
        raise HTTPException(400, "bad range")
    return {"range": {"start_gid": start_gid, "end_gid": end_gid, "ref": _ref(st["idx"], start_gid, end_gid),
                      "text": st["idx"].text(start_gid, end_gid)},
            "play": play_plan(st, start_gid, end_gid, False, pad_ms)}


@app.get("/")
def page() -> FileResponse:
    return FileResponse(Path(__file__).with_name("qari_demo.html"))
