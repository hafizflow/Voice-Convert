"""Convert cpfair/quran-align output (CC BY 4.0) to per-word timings.

Input per ayah: segments [word_start, word_end_exclusive, start_ms, end_ms], word indices 0-based over
the ayah's own words (Basmala prefix excluded, matching qari.text). Known defects in the data, handled
explicitly and flagged so nothing is silently invented:
  - one segment spanning several words        -> SPLIT by character weight
  - a word with no segment                    -> INTERPOLATED between its neighbours
  - a segment with end <= start               -> treated as missing, INTERPOLATED
  - missing words after the last segment      -> start known, end_ms None (play to end of file)
  - indices beyond the ayah's word count      -> ignored and the ayah is marked suspect
"""
from __future__ import annotations

from dataclasses import dataclass

from qari.text.index import WordIndex

OK, SPLIT, INTERPOLATED = 0, 1, 2


@dataclass(frozen=True)
class WordTiming:
    start_ms: int
    end_ms: int | None
    flag: int


@dataclass(frozen=True)
class AyahTimings:
    words: list[WordTiming]
    suspect: bool


def _split(lo: int, hi: int, weights: list[int]) -> list[tuple[int, int]]:
    total = sum(weights)
    bounds, acc = [lo], 0
    for w in weights:
        acc += w
        bounds.append(lo + round((hi - lo) * acc / total))
    return list(zip(bounds[:-1], bounds[1:]))


def convert_ayah(segments: list[list[int]], norms: list[str]) -> AyahTimings:
    n = len(norms)
    weight = [max(1, len(x)) for x in norms]
    got: list[tuple[int, int, int] | None] = [None] * n
    suspect = False
    for ws, we, st, en in segments:
        if ws < 0 or we > n or we <= ws:
            suspect = True
            ws, we = max(ws, 0), min(we, n)
            if we <= ws:
                continue
        if en <= st:
            continue  # defective: leave as missing
        if we - ws == 1:
            got[ws] = (st, en, OK)
        else:
            for k, (s, e) in zip(range(ws, we), _split(st, en, weight[ws:we])):
                got[k] = (s, e, SPLIT)

    out: list[WordTiming] = []
    i = 0
    while i < n:
        if got[i] is not None:
            s, e, f = got[i]
            out.append(WordTiming(s, e, f))
            i += 1
            continue
        j = i
        while j < n and got[j] is None:
            j += 1
        lo = out[-1].end_ms if out and out[-1].end_ms is not None else (out[-1].start_ms if out else 0)
        if j == n:  # trailing run: no next word to bound it
            out.extend(WordTiming(lo, None, INTERPOLATED) for _ in range(i, j))
        else:
            hi = max(got[j][0], lo)
            out.extend(WordTiming(s, e, INTERPOLATED) for s, e in _split(lo, hi, weight[i:j]))
        i = j
    return AyahTimings(out, suspect)


def convert(entries: list[dict], index: WordIndex) -> dict[tuple[int, int], AyahTimings]:
    result = {}
    for e in entries:
        first, last = index.ayah_range(e["surah"], e["ayah"])
        norms = [index[g].norm for g in range(first, last + 1)]
        result[(e["surah"], e["ayah"])] = convert_ayah(e["segments"], norms)
    return result
