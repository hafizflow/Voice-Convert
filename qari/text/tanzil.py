"""Parse Tanzil's `surah|ayah|text` file and build the global word list.

Tanzil prepends the Basmala to the first ayah of every surah except 1 (where it is
ayah 1) and 9 (which has none). We strip it so surah text is the surah's own words;
the Basmala is represented once, as the four words of 1:1.
"""
from __future__ import annotations

from qari.text.index import Word
from qari.text.normalize import normalize

BASMALA_WORDS = 4
NO_BASMALA_PREFIX = frozenset({1, 9})


def parse_tanzil(text: str) -> list[tuple[int, int, str]]:
    rows = []
    for line in text.splitlines():
        line = line.lstrip("﻿").strip()
        if not line or line.startswith("#"):
            continue
        surah, ayah, body = line.split("|", 2)
        rows.append((int(surah), int(ayah), body.strip()))
    return rows


def build_words(rows: list[tuple[int, int, str]]) -> tuple[Word, ...]:
    basmala_norm: list[str] | None = None
    words: list[Word] = []
    prev_surah, prev_ayah = 0, 0
    for surah, ayah, body in rows:
        if surah == prev_surah:
            if ayah != prev_ayah + 1:
                raise ValueError(f"ayah gap at {surah}:{ayah}")
        elif ayah != 1 or surah < prev_surah:
            raise ValueError(f"bad surah/ayah order at {surah}:{ayah}")
        prev_surah, prev_ayah = surah, ayah

        tokens = body.split()
        if (surah, ayah) == (1, 1):
            basmala_norm = [normalize(t) for t in tokens]
            if len(basmala_norm) != BASMALA_WORDS:
                raise ValueError("1:1 is not the four-word Basmala")
        if ayah == 1 and surah not in NO_BASMALA_PREFIX and surah != 1:
            if basmala_norm is None:
                raise ValueError("1:1 must precede other surahs")
            if [normalize(t) for t in tokens[:BASMALA_WORDS]] != basmala_norm:
                raise ValueError(f"{surah}:1 does not start with the Basmala")
            tokens = tokens[BASMALA_WORDS:]
        for pos, tok in enumerate(tokens, start=1):
            norm = normalize(tok)
            if not norm:
                raise ValueError(f"{surah}:{ayah} word {pos} normalizes to nothing")
            words.append(Word(len(words), surah, ayah, pos, tok, norm))
    return tuple(words)
