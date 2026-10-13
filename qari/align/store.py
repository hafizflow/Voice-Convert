"""Compact per-reciter timestamp file and the lookups the API uses.

File: {"reciter", "source", "attribution", "suspect": ["s:a"], "ayahs": {"s:a": [[start_ms, end_ms|null, flag], ...]}}
One triple per word of the ayah, in word order. Padding and waqf extension are applied by the API layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import os
from qari.align.quran_align import INTERPOLATED, SPLIT, AyahTimings, WordTiming

TINY_MS = 40  # a word this short cannot be heard as itself; its source segment is not trusted
from qari.reciters.registry import Reciter
from qari.text.index import WordIndex


@dataclass(frozen=True)
class Segment:
    audio_url: str
    start_ms: int
    end_ms: int | None   # None: play to the end of the file
    surah: int
    ayah: int
    first_pos: int
    last_pos: int
    # "exact": every word is a clean 1:1 source segment.
    # "approx": some word was split or interpolated; its edges can be off by a fraction of a word, pad more.
    # "unreliable": a partial range of an ayah whose source and Tanzil disagree on the words; do not play
    #   it as a sub-range (the whole ayah is fine).
    quality: str = "exact"
    # True when the range ends on the ayah's last word. The aligned end can undershoot the file (measured
    # tails of 0.25 to 6 s after the last word), so the API should pad more here, not cut at end_ms.
    ends_ayah: bool = False


class TimestampStore:
    def __init__(self, reciter: Reciter, index: WordIndex, words: list[WordTiming],
                 suspect: set[tuple[int, int]], source: str = ""):
        self.reciter, self.index, self._words, self._suspect, self.source = reciter, index, words, suspect, source

    @classmethod
    def from_timings(cls, reciter, index, timings: dict[tuple[int, int], AyahTimings], source: str):
        words: list[WordTiming | None] = [None] * len(index)
        suspect = set()
        for s, a in sorted({(w.surah, w.ayah) for w in index.words}):
            first, last = index.ayah_range(s, a)
            t = timings.get((s, a))
            if t is None:
                raise ValueError(f"no timings for {s}:{a}")
            if len(t.words) != last - first + 1:
                raise ValueError(f"{s}:{a} has {len(t.words)} timings for {last - first + 1} words")
            words[first:last + 1] = t.words
            if t.suspect:
                suspect.add((s, a))
        return cls(reciter, index, words, suspect, source)

    @classmethod
    def load(cls, path: Path, reciter: Reciter, index: WordIndex) -> "TimestampStore":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("reciter") != reciter.id:
            raise ValueError(f"{path} is for reciter {d.get('reciter')!r}, not {reciter.id!r}")
        timings = {}
        for s, a in sorted({(w.surah, w.ayah) for w in index.words}):
            rows = d["ayahs"].get(f"{s}:{a}")
            if rows is None:
                raise ValueError(f"timestamp file has no {s}:{a}")
            if any(len(r) != 3 for r in rows):
                raise ValueError(f"{s}:{a}: rows must be [start_ms, end_ms, flag]")
            timings[(s, a)] = AyahTimings([WordTiming(*r) for r in rows], False)
        suspect = {tuple(int(x) for x in k.split(":")) for k in d.get("suspect", [])}
        for k in suspect:
            if k not in timings:
                raise ValueError(f"suspect ayah {k[0]}:{k[1]} is not in the Quran")
            timings[k] = AyahTimings(timings[k].words, True)
        return cls.from_timings(reciter, index, timings, d.get("source", ""))

    def write(self, path: Path) -> None:
        ayahs = {}
        for s, a in sorted({(w.surah, w.ayah) for w in self.index.words}):
            f, l = self.index.ayah_range(s, a)
            ayahs[f"{s}:{a}"] = [[w.start_ms, w.end_ms, w.flag] for w in self._words[f:l + 1]]
        doc = {"reciter": self.reciter.id, "source": self.source,
               "attribution": self.reciter.license["timestamps"],
               "suspect": [f"{s}:{a}" for s, a in sorted(self._suspect)], "ayahs": ayahs}
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)

    def word(self, gid: int) -> WordTiming:
        return self._words[gid]

    def is_suspect(self, surah: int, ayah: int) -> bool:
        return (surah, ayah) in self._suspect

    def segments(self, start_gid: int, end_gid: int) -> list[Segment]:
        """One segment per ayah touched by the inclusive gid range."""
        if not 0 <= start_gid <= end_gid < len(self.index):
            raise ValueError(f"bad gid range {start_gid}..{end_gid} (0..{len(self.index) - 1})")
        out: list[Segment] = []
        gid = start_gid
        while gid <= end_gid:
            w = self.index[gid]
            ayah_first, ayah_last = self.index.ayah_range(w.surah, w.ayah)
            last = min(end_gid, ayah_last)
            ws = self._words[gid:last + 1]
            if (w.surah, w.ayah) in self._suspect:
                whole = gid == ayah_first and last == ayah_last
                start, end = ws[0].start_ms, None   # no trustworthy word edges: play to end of file
                quality = "exact" if whole else "unreliable"
            else:
                start, end = ws[0].start_ms, ws[-1].end_ms
                approx = any(x.flag in (SPLIT, INTERPOLATED) or x.end_ms - x.start_ms < TINY_MS for x in ws)
                quality = "approx" if approx else "exact"
            out.append(Segment(self.reciter.audio_url(w.surah, w.ayah), start, end, w.surah, w.ayah,
                               w.pos, self.index[last].pos, quality, last == ayah_last))
            gid = last + 1
        return out

    def basmala_segment(self) -> Segment:
        first, last = self.index.ayah_range(1, 1)
        return self.segments(first, last)[0]
