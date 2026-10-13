"""Compact per-reciter timestamp file and the lookups the API uses.

File: {"reciter", "source", "attribution", "suspect": ["s:a"], "ayahs": {"s:a": [[start_ms, end_ms|null, flag], ...]}}
One triple per word of the ayah, in word order. Padding and waqf extension are applied by the API layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qari.align.quran_align import AyahTimings, WordTiming
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
        timings = {}
        for s, a in sorted({(w.surah, w.ayah) for w in index.words}):
            rows = d["ayahs"].get(f"{s}:{a}")
            if rows is None:
                raise ValueError(f"timestamp file has no {s}:{a}")
            timings[(s, a)] = AyahTimings([WordTiming(*r) for r in rows], False)
        suspect = {tuple(int(x) for x in k.split(":")) for k in d.get("suspect", [])}
        for k in suspect:
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
        Path(path).write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    def word(self, gid: int) -> WordTiming:
        return self._words[gid]

    def is_suspect(self, surah: int, ayah: int) -> bool:
        return (surah, ayah) in self._suspect

    def segments(self, start_gid: int, end_gid: int) -> list[Segment]:
        """One segment per ayah touched by the inclusive gid range."""
        out: list[Segment] = []
        gid = start_gid
        while gid <= end_gid:
            w = self.index[gid]
            _, ayah_last = self.index.ayah_range(w.surah, w.ayah)
            last = min(end_gid, ayah_last)
            out.append(Segment(self.reciter.audio_url(w.surah, w.ayah), self._words[gid].start_ms,
                               self._words[last].end_ms, w.surah, w.ayah, w.pos, self.index[last].pos))
            gid = last + 1
        return out

    def basmala_segment(self) -> Segment:
        first, last = self.index.ayah_range(1, 1)
        return self.segments(first, last)[0]
