"""Convert cpfair/quran-align output (CC BY 4.0) to per-word timings.

Input per ayah: segments [word_start, word_end_exclusive, start_ms, end_ms], word indices 0-based over
the ayah's own words (Basmala prefix excluded, matching qari.text).

Known defects in the data and what is done about each. Nothing is silently invented: every word that is
not a clean 1:1 segment carries a flag, and the store surfaces it on each returned Segment.

  structural mismatch (indices past the ayah's word count, overlapping index ranges): the source and
      Tanzil disagree on the words (10:1 and 13:1 in the Afasy file). No per-word time can be trusted:
      all words are SUSPECT, and only whole-ayah playback is allowed.
  one segment spanning several words      -> SPLIT by character weight
  a word with no usable segment           -> INTERPOLATED. If the gap between its neighbours is wide
      enough it gets the gap. If not (the usual case: the word is inside a neighbour's span) it takes a
      slice of the neighbour with the higher ms-per-character, and that neighbour becomes SPLIT too.
  end <= start, time going backwards      -> unusable segment, treated as missing
  tiny overlaps (<= 30 ms)                -> start clamped to the previous end
"""
from __future__ import annotations

from dataclasses import dataclass

from qari.text.index import WordIndex

OK, SPLIT, INTERPOLATED, SUSPECT = 0, 1, 2, 3
MIN_GAP_MS = 60
OVERLAP_TOL_MS = 30


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


def _suspect(segments, n) -> AyahTimings:
    starts = [st for _, _, st, _ in segments if st >= 0]
    lead = min(starts) if starts else 0
    return AyahTimings([WordTiming(lead, None, SUSPECT)] * n, True)


def convert_ayah(segments: list[list[int]], norms: list[str]) -> AyahTimings:
    n = len(norms)
    weight = [max(1, len(x)) for x in norms]
    segs = sorted(segments, key=lambda s: (s[0], s[1]))
    prev_we = 0
    for ws, we, _, _ in segs:
        if ws < prev_we or we <= ws or we > n or ws < 0:
            return _suspect(segs, n)
        prev_we = we

    words: list[WordTiming | None] = [None] * n
    prev_end = 0
    for ws, we, st, en in segs:
        if st < 0:
            st = 0 if st >= -OVERLAP_TOL_MS else st
        if st < prev_end and prev_end - st <= OVERLAP_TOL_MS:
            st = prev_end
        if st < 0 or st < prev_end or en <= st:
            continue                      # unusable: leave its words missing
        prev_end = en
        if we - ws == 1:
            words[ws] = WordTiming(st, en, OK)
        else:
            for k, (s, e) in zip(range(ws, we), _split(st, en, weight[ws:we])):
                words[k] = WordTiming(s, e, SPLIT)
    if all(w is None for w in words):
        return _suspect(segs, n)

    i = 0
    while i < n:
        if words[i] is not None:
            i += 1
            continue
        j = i
        while j < n and words[j] is None:
            j += 1
        prev = words[i - 1] if i > 0 else None
        nxt = words[j] if j < n else None
        run = list(range(i, j))
        gap_lo = prev.end_ms if prev else 0
        gap_hi = nxt.start_ms if nxt else None
        if gap_hi is not None and gap_hi - gap_lo >= MIN_GAP_MS:
            for k, (s, e) in zip(run, _split(gap_lo, gap_hi, [weight[k] for k in run])):
                words[k] = WordTiming(s, e, INTERPOLATED)
        else:
            # the missing words live inside a neighbour's span: share it by character weight
            use_prev = prev is not None and (nxt is None or
                (prev.end_ms - prev.start_ms) / weight[i - 1] >= (nxt.end_ms - nxt.start_ms) / weight[j])
            if use_prev:
                group, host = [i - 1] + run, words[i - 1]
            else:
                group, host = run + [j], words[j]
            parts = _split(host.start_ms, host.end_ms, [weight[k] for k in group])
            for k, (s, e) in zip(group, parts):
                words[k] = WordTiming(s, e, SPLIT if k in (i - 1, j) else INTERPOLATED)
        i = j
    return AyahTimings(words, False)


def convert(entries: list[dict], index: WordIndex) -> dict[tuple[int, int], AyahTimings]:
    result = {}
    for e in entries:
        first, last = index.ayah_range(e["surah"], e["ayah"])
        norms = [index[g].norm for g in range(first, last + 1)]
        result[(e["surah"], e["ayah"])] = convert_ayah(e["segments"], norms)
    return result
