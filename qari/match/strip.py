"""Detect and remove the isti'adha and the Basmala from recognized words."""
from __future__ import annotations

from qari.match.align import similarity, skeleton

ISTIADHA = [skeleton(w) for w in "اعوذ بالله من الشيطان الرجيم".split()]
BASMALA = [skeleton(w) for w in "بسم الله الرحمن الرحيم".split()]


def _positional_hits(words: list[str], pattern: list[str], threshold: float) -> int:
    return sum(similarity(w, p) >= threshold for w, p in zip(words, pattern))


def strip_istiadha(words: list[str]) -> tuple[list[str], bool]:
    """Strip a leading isti'adha. Needs the first word to be أعوذ and 3 of the 5 positions to agree,
    so a verse that merely begins with أعوذ بالله (e.g. 2:67) is left alone."""
    head = words[:len(ISTIADHA)]
    if len(head) >= 4 and similarity(head[0], ISTIADHA[0]) >= 0.7 and _positional_hits(head, ISTIADHA, 0.7) >= 3:
        return words[len(head):], True
    return words, False


def is_basmala_at(words: list[str], start: int) -> bool:
    window = words[start:start + len(BASMALA)]
    return len(window) == len(BASMALA) and _positional_hits(window, BASMALA, 0.75) >= 3 and \
        similarity(window[0], BASMALA[0]) >= 0.75


def strip_basmala(words: list[str]) -> tuple[list[str], bool, bool]:
    """Returns (words, leading_basmala_found, internal_basmala_found). An internal Basmala sits between
    two surahs in a continuous recitation; it is removed so the surah text on both sides still matches."""
    leading = is_basmala_at(words, 0)
    if leading:
        words = words[len(BASMALA):]
    internal = False
    for i in range(len(words) - len(BASMALA) + 1):
        if is_basmala_at(words, i):
            words = words[:i] + words[i + len(BASMALA):]
            internal = True
            break
    return words, leading, internal
