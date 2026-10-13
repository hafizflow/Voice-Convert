"""Synthetic evaluation of the matcher. No ASR exists yet, so queries are generated from the Tanzil
text with ASR-like damage and labelled with the true range. The numbers bound the matcher's logic,
not end-to-end accuracy (real ASR errors are not uniform character noise)."""
from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass

from qari.match.align import skeleton
from qari.match.matcher import Matcher
from qari.text.index import WordIndex

CONFUSIONS = {"ث": "س", "س": "ث", "ذ": "ز", "ز": "ذ", "ض": "ظ", "ظ": "ض", "ع": "ء", "ء": "ع",
              "ه": "ح", "ح": "ه", "ص": "س", "ط": "ت", "ت": "ط", "ق": "ك", "ك": "ق"}
ISTIADHA = "اعوذ بالله من الشيطان الرجيم"
BASMALA = "بسم الله الرحمن الرحيم"
CATEGORIES = ["full_verse", "half_verse", "span", "multi_verse", "cross_surah"]
NOISES = ["clean", "drop1", "sub1", "extra1", "drop1+sub1", "prefix"]


@dataclass
class Query:
    category: str
    noise: str
    text: str
    start: int
    end: int


def _ayahs(idx: WordIndex):
    return sorted({(w.surah, w.ayah) for w in idx.words})


def make_range(idx: WordIndex, category: str, rng: random.Random) -> tuple[int, int]:
    ayahs = _ayahs(idx)
    for _ in range(1000):
        s, a = rng.choice(ayahs)
        first, last = idx.ayah_range(s, a)
        n = last - first + 1
        if category == "full_verse" and n >= 3:
            return first, last
        if category == "half_verse" and n >= 6:
            return (first, first + n // 2 - 1) if rng.random() < 0.5 else (first + n // 2, last)
        if category == "span":
            k = rng.randint(3, 15)
            if k <= n:
                lo = rng.randint(first, last - k + 1)
                return lo, lo + k - 1
        if category == "multi_verse":
            k = rng.randint(2, 4)
            end_ayah = (s, a + k - 1)
            if end_ayah in set(ayahs):
                return first, idx.ayah_range(*end_ayah)[1]
        if category == "cross_surah" and s < 114:
            sf, sl = idx.surah_range(s)
            nf, nl = idx.surah_range(s + 1)
            lo = max(sf, sl - rng.randint(2, 6))
            hi = min(nl, nf + rng.randint(2, 6))
            return lo, hi
    raise RuntimeError(category)


def damage(words: list[str], noise: str, rng: random.Random, vocab: list[str]) -> tuple[list[str], int, int]:
    """Returns (words, words_trimmed_at_start, words_trimmed_at_end): a dropped edge word shrinks the
    range the user actually recited, so the truth must shrink with it."""
    w = list(words)
    trim_lo = trim_hi = 0
    if "drop1" in noise and len(w) > 4:
        i = rng.randrange(len(w))
        del w[i]
        trim_lo, trim_hi = int(i == 0), int(i == len(words) - 1)
    if "sub1" in noise:
        i = rng.randrange(len(w))
        chars = [c for c in w[i] if c in CONFUSIONS]
        if chars:
            c = rng.choice(chars)
            w[i] = w[i].replace(c, CONFUSIONS[c], 1)
        elif len(w[i]) > 2:
            w[i] = w[i][:-1]
    if "extra1" in noise:
        w.insert(rng.randrange(len(w) + 1), rng.choice(vocab))
    return w, trim_lo, trim_hi


def generate(idx: WordIndex, per_cell: int, seed: int) -> list[Query]:
    rng = random.Random(seed)
    vocab = sorted({w.norm for w in idx.words})
    out = []
    for cat in CATEGORIES:
        for noise in NOISES:
            for _ in range(per_cell):
                lo, hi = make_range(idx, cat, rng)
                words = [idx[g].norm for g in range(lo, hi + 1)]
                words, trim_lo, trim_hi = damage(words, noise, rng, vocab)
                lo, hi = lo + trim_lo, hi - trim_hi
                text = " ".join(words)
                if noise == "prefix":
                    text = ISTIADHA + " " + BASMALA + " " + text if rng.random() < 0.5 else BASMALA + " " + text
                out.append(Query(cat, noise, text, lo, hi))
    return out


def run(matcher: Matcher, queries: list[Query]) -> dict:
    idx = matcher.index
    keys = matcher.ngrams.keys
    cells: dict = defaultdict(lambda: defaultdict(int))
    wrong: list = []
    scored: list = []   # (confidence, correct) for every matched query
    times = []
    for q in queries:
        t = time.perf_counter()
        r = matcher.match(q.text)
        times.append((time.perf_counter() - t) * 1000)
        c = cells[(q.category, q.noise)]
        c["n"] += 1
        truth_key = tuple(keys[q.start:q.end + 1])
        if r.status == "matched":
            exact = (r.start_gid, r.end_gid) == (q.start, q.end)
            near = abs(r.start_gid - q.start) <= 1 and abs(r.end_gid - q.end) <= 1
            same_text = tuple(keys[r.start_gid:r.end_gid + 1]) == truth_key
            c["matched"] += 1
            scored.append((r.confidence, near))   # right words at the wrong place is NOT correct
            c["exact"] += exact
            c["within1"] += near
            if not near and same_text:
                c["same_text_elsewhere"] += 1
            elif not near:
                c["wrong_confident"] += 1
                wrong.append((q, r))
        elif r.status == "ambiguous":
            c["ambiguous"] += 1
            c["ambiguous_has_truth"] += any(abs(a.start_gid - q.start) <= 1 and abs(a.end_gid - q.end) <= 1
                                            or a.text_key == truth_key for a in r.alternatives)
        else:
            c[r.status] += 1
    times.sort()
    return {"cells": cells, "wrong": wrong, "scored": scored, "ms_p50": times[len(times) // 2], "ms_p95": times[int(len(times) * .95)],
            "n": len(queries)}


def format_report(res: dict) -> str:
    cols = ["n", "exact", "within1", "same_text_elsewhere", "wrong_confident", "ambiguous", "ambiguous_has_truth",
            "not_found", "too_short"]
    lines = ["| category | noise | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    tot = defaultdict(int)
    for (cat, noise), c in sorted(res["cells"].items(), key=lambda kv: (CATEGORIES.index(kv[0][0]), NOISES.index(kv[0][1]))):
        lines.append(f"| {cat} | {noise} | " + " | ".join(str(c.get(k, 0)) for k in cols) + " |")
        for k in cols:
            tot[k] += c.get(k, 0)
    lines.append("| **all** | | " + " | ".join(str(tot[k]) for k in cols) + " |")
    n = tot["n"]
    lines.append("")
    lines.append(f"top-1 within ±1 word: {tot['within1'] / n:.1%}   exact boundaries: {tot['exact'] / n:.1%}   "
                 f"confident-wrong: {tot['wrong_confident']} ({tot['wrong_confident'] / n:.2%})   "
                 f"match latency p50/p95: {res['ms_p50']:.1f}/{res['ms_p95']:.1f} ms (CPU, this machine)")
    return "\n".join(lines)


def threshold_table(res: dict, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)) -> str:
    """If the app only auto-plays at confidence >= t: how many queries play, and how many of those are wrong."""
    n = res["n"]
    lines = ["| auto-play if confidence >= | auto-played | of all queries | wrong among played |", "|---|---|---|---|"]
    for t in thresholds:
        played = [ok for conf, ok in res["scored"] if conf >= t]
        wrong = sum(1 for ok in played if not ok)
        lines.append(f"| {t:.1f} | {len(played)} | {len(played) / n:.1%} | {wrong} ({wrong / max(1, len(played)):.2%}) |")
    return "\n".join(lines)
