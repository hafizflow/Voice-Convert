"""Detect and remove the isti'adha and the Basmala from recognized words."""
from __future__ import annotations

from qari.match.align import similarity, skeleton

ISTIADHA = [skeleton(w) for w in "اعوذ بالله من الشيطان الرجيم".split()]
BASMALA = [skeleton(w) for w in "بسم الله الرحمن الرحيم".split()]


def _positional_hits(words: list[str], pattern: list[str], threshold: float) -> int:
    return sum(similarity(w, p) >= threshold for w, p in zip(words, pattern))


def strip_istiadha(words: list[str]) -> tuple[list[str], bool]:
    """Strip a leading isti'adha, through the last word that lines up with it (the reciter may skip one).
    Needs أعوذ first and 3 isti'adha words in order, so a verse that merely begins with أعوذ بالله
    (2:67) is left alone."""
    if not words or similarity(words[0], ISTIADHA[0]) < 0.7:
        return words, False
    j, last, hits = 0, -1, 0
    for i, w in enumerate(words[:len(ISTIADHA) + 1]):
        for t in (0, 1):                      # the next pattern word, or one skipped
            if j + t < len(ISTIADHA) and similarity(w, ISTIADHA[j + t]) >= 0.7:
                j, last, hits = j + t + 1, i, hits + 1
                break
        else:
            break
    if hits >= 3:
        return words[last + 1:], True
    return words, False


def is_basmala_at(words: list[str], start: int) -> bool:
    window = words[start:start + len(BASMALA)]
    return len(window) == len(BASMALA) and _positional_hits(window, BASMALA, 0.75) >= 3 and \
        similarity(window[0], BASMALA[0]) >= 0.75


def strip_basmala(words: list[str]) -> tuple[list[str], bool, list[str] | None]:
    """Returns (words without a leading Basmala, leading_found, words_with_internal_Basmala_removed).
    The third value is None when there is no internal Basmala. It is only an *alternative reading*:
    27:30 contains the Basmala as part of the verse, so the caller must decide by alignment whether
    removing it helps (a recitation that runs from one surah into the next) or hurts (27:30)."""
    leading = is_basmala_at(words, 0)
    if leading:
        words = words[len(BASMALA):]
    for i in range(len(words) - len(BASMALA) + 1):
        if is_basmala_at(words, i):
            return words, leading, words[:i] + words[i + len(BASMALA):]
    return words, leading, None
